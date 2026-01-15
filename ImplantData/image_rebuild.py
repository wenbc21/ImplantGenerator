import os
import argparse
import gc
import csv
import json
import numpy as np
import time
import SimpleITK as sitk
import pydicom
from pydicom.uid import ImplicitVRLittleEndian, CTImageStorage, generate_uid
from skimage.transform import rescale
from data_utils import *


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--data_path', type=str, default='ImplantData/data/UpperAnterior')
    parser.add_argument('--spacing', type=float, default=0.3)
    parser.add_argument('--results_path', type=str, default='results/ImplantGeneration_UpperAnterior')
    parser.add_argument('--location_path', type=str, default='results/ImplantLocation_UpperAnterior/location.json')

    return parser


def image_rebuild(args) :
    cbct_data_path = [item.path for item in os.scandir(f"{args.data_path}/CBCT") if item.is_dir()]
    cbct_data_path.sort()
    predict_path = [item.path for item in os.scandir(f"{args.results_path}/standard_implant") if item.is_file()]
    predict_path.sort()

    cbct_index = {}
    for it in range(len(cbct_data_path)) :
        cbct_index[os.path.split(cbct_data_path[it])[-1][:3]] = cbct_data_path[it]

    with open(args.location_path, 'r') as file:
        location = json.load(file)
    
    metadata, _, _ = get_dataset_metadata(args.data_path)
    
    rebuild_path = os.path.join(args.results_path, "rebuild_std")
    os.makedirs(os.path.join(rebuild_path, "rebuild_nii"), exist_ok=True)
    os.makedirs(os.path.join(rebuild_path, "rebuild_nii_predict"), exist_ok=True)
    os.makedirs(os.path.join(rebuild_path, "rebuild_dicom"), exist_ok=True)
    
    for it in range(len(predict_path)) :
        # get dicom file and pcd file
        data_name = os.path.basename(predict_path[it]).split('.')[0]
        data_id = data_name[-3:]
        dicom_dir = cbct_index[data_id]
        dicom = get_dcm_3d_array(dicom_dir)
        dicom = window_transform_3d(dicom, metadata[data_id]["width"], metadata[data_id]["window"]).astype(np.uint8)
        predict = sitk.ReadImage(predict_path[it])
        predict = sitk.GetArrayFromImage(predict)

        centroid = np.array(location[data_name])
        if metadata[data_id]["spacing"] != args.spacing:
            resize_rate = args.spacing / metadata[data_id]["spacing"]
            predict = rescale(predict, resize_rate, order=1, preserve_range=True)
            centroid *= resize_rate

        # nii will be translated by window size and width
        rebuild_nii(dicom, predict, centroid, data_name, rebuild_path)

        # dicom retain the original values
        rebuild_dicom(dicom_dir, predict, centroid, data_name, rebuild_path)

        gc.collect()
        print(f"{data_name} Done!")


def rebuild_nii(dicom, predict, centroid, data_name, result_dir) :
    # space transfer
    midx, midy, midz = int(centroid[0]), int(centroid[1]), int(centroid[2])
    patch_size = predict.shape
    predict = np.array(np.where(predict == 1))
    predict[0] += (midx - patch_size[0] // 2)
    predict[1] += (midy - patch_size[1] // 2)
    predict[2] += (midz - patch_size[2] // 2)
    predict = predict.T

    rebuild_nii_predict = np.zeros_like(dicom, dtype=np.uint8)

    # rebuild dicom
    for p in predict :
        dicom[p[0]][p[1]][p[2]] = 255
        rebuild_nii_predict[p[0]][p[1]][p[2]] = 1
    
    # write nii file
    rebuild_nii = sitk.GetImageFromArray(dicom)
    sitk.WriteImage(rebuild_nii, os.path.join(result_dir, "rebuild_nii", f"{data_name}.nii.gz"))

    # write nii file
    rebuild_nii_predict = sitk.GetImageFromArray(rebuild_nii_predict)
    sitk.WriteImage(rebuild_nii_predict, os.path.join(result_dir, "rebuild_nii_predict", f"{data_name}.nii.gz"))


def rebuild_dicom(dicom_dir, predict, centroid, data_name, result_dir):
    # read dicom
    dicoms = []
    for f in os.listdir(dicom_dir):
        if f.endswith(".dcm"):
            dcm = pydicom.dcmread(os.path.join(dicom_dir, f), force=True)
            if not hasattr(dcm, "file_meta"):
                dcm.file_meta = pydicom.dataset.FileMetaDataset()
            dcm.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
            dcm.file_meta.MediaStorageSOPClassUID = CTImageStorage
            dcm.file_meta.MediaStorageSOPInstanceUID = generate_uid()
            dcm.file_meta.ImplementationClassUID = generate_uid()
            dcm.is_little_endian = True
            dcm.is_implicit_VR = True
            dicoms.append(dcm)

    dicoms.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    dicom_image = np.stack([d.pixel_array for d in dicoms]).astype(np.int16)
    max_dicom = np.max(dicom_image)
    
    # space transfer
    midx, midy, midz = int(centroid[0]), int(centroid[1]), int(centroid[2])
    patch_size = predict.shape
    predict = np.array(np.where(predict == 1))
    predict[0] += (midx - patch_size[0] // 2)
    predict[1] += (midy - patch_size[1] // 2)
    predict[2] += (midz - patch_size[2] // 2)
    predict = predict.T
    
    for p in predict :
        dicom_image[p[0]][p[1]][p[2]] = max_dicom

    # write dicom
    new_series_uid = generate_uid()
    os.makedirs(os.path.join(result_dir, "rebuild_dicom", data_name), exist_ok=True)
    for i, dcm in enumerate(dicoms):
        dcm.PixelData = dicom_image[i].astype(np.int16).tobytes()
        dcm.Rows, dcm.Columns = dicom_image[i].shape
        dcm.BitsAllocated = 16
        dcm.BitsStored = 16
        dcm.HighBit = 15
        dcm.PixelRepresentation = 1
        dcm.SeriesInstanceUID = new_series_uid
        dcm.SOPInstanceUID = dcm.file_meta.MediaStorageSOPInstanceUID
        dcm.InstanceNumber = i + 1
        dcm.ImageType = ["DERIVED", "SECONDARY"]

        dcm.save_as(
            os.path.join(result_dir, "rebuild_dicom", data_name, f"rebuild_{i:03d}.dcm"),
            write_like_original=False
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Rebuild image files', parents=[get_args_parser()])
    args = parser.parse_args()

    image_rebuild(args)
