import SimpleITK as sitk
import os
import argparse
from get_dicom import *
from get_stl import *
from get_pcd import *
import random
import json


def get_args_parser():
    parser = argparse.ArgumentParser('Transform CBCT and Implant data into training set', add_help=False)
    parser.add_argument('--augment_size', type=int, default=10)
    parser.add_argument('--displacement', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=96)
    parser.add_argument('--dicom_path', type=str, default="./data/data/01_mimics/")
    parser.add_argument('--pcd_path', type=str, default="./data/data/02_pointcloud/")
    parser.add_argument('--split_path', type=str, default='./data/data/splits.json')
    parser.add_argument('--images_path', type=str, default='./data/nii/train/raw')
    parser.add_argument('--labels_path', type=str, default='./data/nii/train/label')

    return parser


def train_to_nii(args) :

    # read raw data
    dicom_dirs = [item.path for item in os.scandir(args.dicom_path) if item.is_dir()]
    pcd_dirs = [item.path for item in os.scandir(args.pcd_path) if item.is_file()]
    dicom_dirs.sort()
    pcd_dirs.sort()

    # assort dataset base on random generated split file
    with open(args.split_path) as f:
        train_split = [int(i)-1 for i in json.load(f)['train']]
    dicom_dirs = [dicom_dirs[i] for i in train_split]
    pcd_dirs = [pcd_dirs[i] for i in train_split]
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
        
        # get central position of cylinder
        cylinder = np.array(np.where(pcd == 1))
        midx = (np.min(cylinder[0]) + np.max(cylinder[0])) // 2
        midy = (np.min(cylinder[1]) + np.max(cylinder[1])) // 2
        midz = (np.min(cylinder[2]) + np.max(cylinder[2])) // 2
        
        # augmentation
        for jt in range(args.augment_size) :
            displacement = args.displacement
            half_patch = int(args.patch_size / 2)

            # random displacement for [x,y,z] axis
            randomx = random.randint(0, displacement * 2) - displacement
            randomy = random.randint(0, displacement * 2) - displacement
            randomz = random.randint(0, displacement * 2) - displacement
            midx_part = (midx + randomx - half_patch, midx + randomx + half_patch)
            midy_part = (midy + randomy - half_patch, midy + randomy + half_patch)
            midz_part = (midz + randomz - half_patch, midz + randomz + half_patch)
            dicom_part = dicom[midx_part[0]:midx_part[1], midy_part[0]:midy_part[1], midz_part[0]:midz_part[1]]
            pcd_part = pcd[midx_part[0]:midx_part[1], midy_part[0]:midy_part[1], midz_part[0]:midz_part[1]]
            
            # random flip for [x,y,z] axis
            random_flip = random.randint(0, 3)
            dicom_part = np.rot90(dicom_part, k = random_flip, axes = (0, 1))
            pcd_part = np.rot90(pcd_part, k = random_flip, axes = (0, 1))
            random_flip = random.randint(0, 3)
            dicom_part = np.rot90(dicom_part, k = random_flip, axes = (0, 2))
            pcd_part = np.rot90(pcd_part, k = random_flip, axes = (0, 2))
            random_flip = random.randint(0, 3)
            dicom_part = np.rot90(dicom_part, k = random_flip, axes = (1, 2))
            pcd_part = np.rot90(pcd_part, k = random_flip, axes = (1, 2))
            
            # write images
            dicom_part = sitk.GetImageFromArray(dicom_part)
            pcd_part = sitk.GetImageFromArray(pcd_part)
            image_number = str(it * args.augment_size + jt + 1).zfill(4)
            sitk.WriteImage(dicom_part, f"{args.images_path}/IMPLANT_{image_number}_0000.nii.gz")
            sitk.WriteImage(pcd_part, f"{args.labels_path}/IMPLANT_{image_number}.nii.gz")
        
        print(f"{str(it+1).zfill(4)} done! midx:{midx} midy:{midy} midz:{midz}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Transform CBCT and Implant data into training set', parents=[get_args_parser()])
    args = parser.parse_args()

    train_to_nii(args)
