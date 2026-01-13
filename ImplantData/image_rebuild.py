import os
import argparse
import gc
import csv
import json
import numpy as np
import time
import SimpleITK as sitk
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
    # Read DICOM series
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise RuntimeError("No DICOM series found")

    filenames = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
    reader.SetFileNames(filenames)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()
    dcm_3d_array = sitk.GetArrayFromImage(image)
    max_dicom = np.max(dcm_3d_array)

    # space transfer
    midx, midy, midz = int(centroid[0]), int(centroid[1]), int(centroid[2])
    patch_size = predict.shape
    predict = np.array(np.where(predict == 1))
    predict[0] += (midx - patch_size[0] // 2)
    predict[1] += (midy - patch_size[1] // 2)
    predict[2] += (midz - patch_size[2] // 2)
    predict = predict.T
    
    # rebuild dicom
    for p in predict :
        dcm_3d_array[p[0]][p[1]][p[2]] = max_dicom
    new_img = sitk.GetImageFromArray(dcm_3d_array)
    new_img.SetSpacing(image.GetSpacing())
    # new_img.CopyInformation(image)

    # Write the 3D image as a series
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")

    # Copy some of the tags and add the relevant tags indicating the change.
    direction = image.GetDirection()
    series_tag_values = [
        ("0008|0031", modification_time),  # Series Time
        ("0008|0021", modification_date),  # Series Date
        ("0008|0008", "DERIVED\\SECONDARY"),  # Image Type
        (
            "0020|000e",
            "1.2.826.0.1.3680043.2.1125." + modification_date + ".1" + modification_time,
        ),  # Series Instance UID
        (
            "0020|0037",
            "\\".join(
                map(
                    str,
                    (
                        direction[0],
                        direction[3],
                        direction[6],
                        direction[1],
                        direction[4],
                        direction[7],
                    ),
                )
            ),
        ),  # Image Orientation
        # (Patient)
        ("0008|103e", "Rebuild DICOM"),  # Series Description
    ]

    # Write slices to output directory
    os.makedirs(os.path.join(result_dir, "rebuild_dicom", data_name), exist_ok=True)
    for i in range(new_img.GetDepth()) :
        image_slice = new_img[:, :, i]

        # Tags shared by the series.
        list(
            map(
                lambda tag_value: image_slice.SetMetaData(tag_value[0], tag_value[1]),
                series_tag_values,
            )
        )

        # Slice specific tags.
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))
        image_slice.SetMetaData("0008|0060", "CT")
        image_slice.SetMetaData(
            "0020|0032",
            "\\".join(map(str, new_img.TransformIndexToPhysicalPoint((0, 0, i)))),
        )
        image_slice.SetMetaData("0020|0013", str(i))

        # Write to the output directory and add the extension dcm, to force writing in DICOM format.
        writer.SetFileName(os.path.join(result_dir, "rebuild_dicom", data_name, f"rebuild_{str(i).zfill(3)}.dcm"))
        writer.Execute(image_slice)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Rebuild image files', parents=[get_args_parser()])
    args = parser.parse_args()

    image_rebuild(args)