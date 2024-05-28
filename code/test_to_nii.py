import SimpleITK as sitk
import os
from get_dicom import *
from get_stl import *
from get_pcd import *
import json


def test_to_nii(params) :

    dicom_dirs = [item.path for item in os.scandir(params['dicom_path']) if item.is_dir()]
    pcd_dirs = [item.path for item in os.scandir(params['pcd_path']) if item.is_file()]
    mid_dirs = [item.path for item in os.scandir(params['mid_path']) if item.is_file()]
    dicom_dirs.sort()
    pcd_dirs.sort()
    mid_dirs.sort()

    with open(params['split_path']) as f:
        test_split = [int(i)-1 for i in json.load(f)['test']]
    dicom_dirs = [dicom_dirs[i] for i in test_split]
    pcd_dirs = [pcd_dirs[i] for i in test_split]

    assert len(dicom_dirs) == len(pcd_dirs), "dicom files not compatible with pcd files!"

    os.makedirs(params['images_path'], exist_ok=True)
    os.makedirs(params['labels_path'], exist_ok=True)

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
        half_patch = int(params['patch_size'] / 2)
        midx_part = (midx - half_patch, midx + half_patch)
        midy_part = (midy - half_patch, midy + half_patch)
        midz_part = (midz - half_patch, midz + half_patch)
        dicom_part = dicom[midx_part[0]:midx_part[1], midy_part[0]:midy_part[1], midz_part[0]:midz_part[1]]
        pcd_part = pcd[midx_part[0]:midx_part[1], midy_part[0]:midy_part[1], midz_part[0]:midz_part[1]]
        
        # write images
        dicom_part = sitk.GetImageFromArray(dicom_part)
        pcd_part = sitk.GetImageFromArray(pcd_part)
        sitk.WriteImage(dicom_part, f"{params['images_path']}/IMPLANT_{str(it+1).zfill(3)}_0000.nii.gz")
        sitk.WriteImage(pcd_part, f"{params['labels_path']}/IMPLANT_{str(it+1).zfill(3)}.nii.gz")
        
        print(f"{str(it+1).zfill(3)} done! midx:{midx} midy:{midy} midz:{midz}")


if __name__ == '__main__':

    params = {
        'dicom_path' : './data/data/01_mimics/',
        'pcd_path' : './data/data/02_pointcloud/',
        'mid_path' : './data/data/labeled_midpoint',
        'split_path' : "./data/data/splits.json",
        'patch_size' : 96,
        'images_path' : './data/nii/labeled/raw',
        'labels_path' : './data/nii/labeled/label'
    }

    test_to_nii(params)
