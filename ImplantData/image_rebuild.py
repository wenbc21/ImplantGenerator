import os
import argparse
import gc
import csv
import json
import numpy as np
import pydicom
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
    predict_path = [item.path for item in os.scandir(f"{args.results_path}/predict") if item.is_file()]
    predict_path.sort()

    cbct_index = {}
    for it in range(len(cbct_data_path)) :
        cbct_index[os.path.split(cbct_data_path[it])[-1][:3]] = cbct_data_path[it]

    with open(args.location_path, 'r') as file:
        location = json.load(file)

    data_id = []
    spacing_dict = {}
    window_dict = {}
    width_dict = {}
    with open(f'{args.data_path}/metadata.csv', mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            id = row['ID'].zfill(3)
            data_id.append(id)
            spacing_dict[id] = float(row['Spacing'])
            window_dict[id] = float(row['Window'])
            width_dict[id] = float(row['Width'])
    
    os.makedirs(os.path.join(args.results_path, "rebuild_nii"), exist_ok=True)
    os.makedirs(os.path.join(args.results_path, "rebuild_dicom"), exist_ok=True)
    
    for it in range(22, len(predict_path)) :
        # get dicom file and pcd file
        data_name = os.path.basename(predict_path[it]).split('.')[0]
        data_id = data_name[-3:]
        dicom_dir = cbct_index[data_id]
        dicom = get_dcm_3d_array(dicom_dir)
        if spacing_dict[data_id] != args.spacing:
            dicom = rescale(dicom, spacing_dict[data_id] / args.spacing, order=1, preserve_range=True)
        dicom = window_transform_3d(dicom, window_width=width_dict[data_id], window_center=window_dict[data_id]).astype(np.uint8)

        centroid = np.array(location[data_name]).astype(int)
        patch_size = 96

        predict = sitk.ReadImage(predict_path[it])
        predict = sitk.GetArrayFromImage(predict)

        # nii will be translated by window size and width
        rebuild_nii(dicom, predict, centroid, patch_size, data_name, args.results_path)

        # dicom retain the original values
        rebuild_dicom(dicom_dir, predict, centroid, patch_size, data_name, args.results_path)

        gc.collect()
        print(f"{data_name} Done!")


def rebuild_nii(dicom, predict, centroid, patch_size, data_name, result_dir) :
    # space transfer
    midx, midy, midz = int(centroid[0]), int(centroid[1]), int(centroid[2])
    predict = np.array(np.where(predict == 1))
    predict[0] += (midx - patch_size // 2)
    predict[1] += (midy - patch_size // 2)
    predict[2] += (midz - patch_size // 2)
    predict = predict.T

    # rebuild dicom
    for p in predict :
        dicom[p[0]][p[1]][p[2]] = 255
    
    # write nii file
    rebuild_nii = sitk.GetImageFromArray(dicom)
    sitk.WriteImage(rebuild_nii, os.path.join(result_dir, "rebuild_nii", f"{data_name}.nii.gz"))


def rebuild_dicom(dicom_dir, predict, centroid, patch_size, data_name, result_dir):
    # rebuild dicom file for each item
    dicom_slice = [item.path for item in os.scandir(dicom_dir) if item.is_file()]
    dicom_slice.sort()
    dicom_slice = dicom_slice
    dicom = get_dcm_3d_array(dicom_dir)
    
    # space transfer
    midx, midy, midz = int(centroid[0]), int(centroid[1]), int(centroid[2])
    predict = np.array(np.where(predict == 1))
    predict[0] += (midx - patch_size // 2)
    predict[1] += (midy - patch_size // 2)
    predict[2] += (midz - patch_size // 2)
    predict = predict.T

    # read original dicom
    dicom_size = dicom.shape
    dicom = np.zeros(dicom_size, dtype = np.int16)
    for i in range(dicom_size[0]) :
        data = pydicom.dcmread(dicom_slice[i])
        databytes = data.PixelData
        image_np = np.frombuffer(databytes, dtype=np.int16).reshape(dicom_size[1:])
        dicom[i] = image_np
    
    # rebuild dicom
    max_dicom = np.max(dicom)
    for p in predict :
        dicom[p[0]][p[1]][p[2]] = max_dicom

    # write dicom file
    os.makedirs(os.path.join(result_dir, "rebuild_dicom", data_name), exist_ok=True)
    for i in range(dicom_size[0]) :
        data = pydicom.dcmread(dicom_slice[i])
        data.pixel_array.data = dicom[i]
        data.PixelData = dicom[i].tobytes()
        data.save_as(os.path.join(result_dir, "rebuild_dicom", data_name, f"predict_{str(i).zfill(3)}.dcm"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Rebuild image files', parents=[get_args_parser()])
    args = parser.parse_args()

    image_rebuild(args)