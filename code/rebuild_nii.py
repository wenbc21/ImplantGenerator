import SimpleITK as sitk
import os
import argparse
import numpy as np
from get_dicom import *
from get_stl import *
import json


def get_args_parser():
    parser = argparse.ArgumentParser('Rebuild nii files from nnU-Net results', add_help=False)
    parser.add_argument('--dicom_path', type=str, default="./data/data/01_mimics/")
    parser.add_argument('--mid_path', type=str, default="./data/data/labeled_midpoint")
    parser.add_argument('--split_path', type=str, default='./data/data/splits.json')
    parser.add_argument('--implant_path', type=str, default='./data/ImplantDataProcess/post_2')
    parser.add_argument('--target_path', type=str, default='./data/data_trans/postprocessed_2')

    return parser


def rebuild_nii(args) :

    # read raw data
    dicom_dirs = [item.path for item in os.scandir(args.dicom_path) if item.is_dir()]
    mid_dirs = [item.path for item in os.scandir(args.mid_path) if item.is_file()]
    implant_dirs = [item.path for item in os.scandir(args.implant_path) if item.is_file()]
    dicom_dirs.sort()
    mid_dirs.sort()
    implant_dirs.sort()

    # assort dataset base on random generated split file
    with open(args.split_path) as f:
        test_split = [int(i)-1 for i in json.load(f)['test']]
    dicom_dirs = [dicom_dirs[i] for i in test_split]
    assert len(dicom_dirs) == len(implant_dirs) and len(dicom_dirs) == len(mid_dirs), "dicom files not compatible with nii files!"

    # rebuild nii file for each item
    for it in range(len(dicom_dirs)) :
        # get dicom file and implant file
        dicom_dir = list([item.path for item in os.scandir(dicom_dirs[it]) if item.is_dir()])[0]
        dicom = get_dicom(dicom_dir)
        implant = sitk.ReadImage(implant_dirs[it])
        implant = sitk.GetArrayFromImage(implant)
        implant = np.array(np.where(implant == 1))

        # get labeled mid point
        with open(mid_dirs[it]) as f:
            mid = json.load(f)
        midx = dicom.shape[0] - mid['midx']
        midy = mid['midz']
        midz = mid['midy']
        
        # space transfer
        implant[0] += (midx - 48)
        implant[1] += (midy - 48)
        implant[2] += (midz - 48)
        implant = implant.T

        # rebuild dicom
        max_dicom = np.max(dicom)
        for p in implant :
            dicom[p[0]][p[1]][p[2]] = max_dicom
        dicom = dicom[::-1, :, :]
        
        # write nii file
        os.makedirs(args.target_path, exist_ok=True)
        os.makedirs(f"{args.target_path}/rebuild_nii", exist_ok=True)
        rebuild_nii = sitk.GetImageFromArray(dicom)
        sitk.WriteImage(rebuild_nii, f"{args.target_path}/rebuild_nii/{str(it+1).zfill(3)}.nii.gz")
            
        print(f"{str(it+1).zfill(3)} done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Rebuild nii files from nnU-Net results', parents=[get_args_parser()])
    args = parser.parse_args()

    rebuild_nii(args)