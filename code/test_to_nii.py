import SimpleITK as sitk
import os
import argparse
from get_dicom import *
from get_stl import *
from get_pcd import *
import json


def get_args_parser():
    parser = argparse.ArgumentParser('Transform CBCT and Implant data into test set', add_help=False)
    parser.add_argument('--patch_size', type=int, default=96)
    parser.add_argument('--dicom_path', type=str, default="./data/data/01_mimics/")
    parser.add_argument('--pcd_path', type=str, default="./data/data/02_pointcloud/")
    parser.add_argument('--mid_path', type=str, default="./data/data/labeled_midpoint")
    parser.add_argument('--split_path', type=str, default='./data/data/splits.json')
    parser.add_argument('--images_path', type=str, default='./data/nii/labeled/raw')
    parser.add_argument('--labels_path', type=str, default='./data/nii/labeled/label')

    return parser


def test_to_nii(args) :

    # read raw data
    dicom_dirs = [item.path for item in os.scandir(args.dicom_path) if item.is_dir()]
    pcd_dirs = [item.path for item in os.scandir(args.pcd_path) if item.is_file()]
    mid_dirs = [item.path for item in os.scandir(args.mid_path) if item.is_file()]
    dicom_dirs.sort()
    pcd_dirs.sort()
    mid_dirs.sort()

    # assort dataset base on random generated split file
    with open(args.split_path) as f:
        test_split = [int(i)-1 for i in json.load(f)['test']]
    dicom_dirs = [dicom_dirs[i] for i in test_split]
    pcd_dirs = [pcd_dirs[i] for i in test_split]
    assert len(dicom_dirs) == len(pcd_dirs), "dicom files not compatible with pcd files!"

    # prepare results directory
    os.makedirs(args.images_path, exist_ok=True)
    os.makedirs(args.labels_path, exist_ok=True)

    # make dataset for each item
    for it in range(len(dicom_dirs)) :
        # get dicom file and pcd file
        dicom_dir = list([item.path for item in os.scandir(dicom_dirs[it]) if item.is_dir()])[0]
        pcd_dir = pcd_dirs[it]
        dicom = get_dicom(dicom_dir)
        pcd = get_pcd(pcd_dir, dicom.shape)

        # get labeled mid point
        with open(mid_dirs[it]) as f:
            mid = json.load(f)
        midx = dicom.shape[0] - mid['midx']
        midy = mid['midz']
        midz = mid['midy']

        # get patch nearby mid point
        half_patch = int(args.patch_size / 2)
        midx_part = (midx - half_patch, midx + half_patch)
        midy_part = (midy - half_patch, midy + half_patch)
        midz_part = (midz - half_patch, midz + half_patch)
        dicom_part = dicom[midx_part[0]:midx_part[1], midy_part[0]:midy_part[1], midz_part[0]:midz_part[1]]
        pcd_part = pcd[midx_part[0]:midx_part[1], midy_part[0]:midy_part[1], midz_part[0]:midz_part[1]]
        
        # write images
        dicom_part = sitk.GetImageFromArray(dicom_part)
        pcd_part = sitk.GetImageFromArray(pcd_part)
        sitk.WriteImage(dicom_part, f"{args.images_path}/IMPLANT_{str(it+1).zfill(3)}_0000.nii.gz")
        sitk.WriteImage(pcd_part, f"{args.labels_path}/IMPLANT_{str(it+1).zfill(3)}.nii.gz")
        
        print(f"{str(it+1).zfill(3)} done! midx:{midx} midy:{midy} midz:{midz}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Transform CBCT and Implant data into test set', parents=[get_args_parser()])
    args = parser.parse_args()

    test_to_nii(args)
