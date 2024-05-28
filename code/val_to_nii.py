import SimpleITK as sitk
import os
from get_dicom import *
from get_stl import *
from get_pcd import *
import random
import json


def val_to_nii(params) :

    dicom_dirs = [item.path for item in os.scandir(params['dicom_path']) if item.is_dir()]
    pcd_dirs = [item.path for item in os.scandir(params['pcd_path']) if item.is_file()]
    dicom_dirs.sort()
    pcd_dirs.sort()

    with open(params['split_path']) as f:
        val_split = [int(i)-1 for i in json.load(f)['val']]
    dicom_dirs = [dicom_dirs[i] for i in val_split]
    pcd_dirs = [pcd_dirs[i] for i in val_split]

    assert len(dicom_dirs) == len(pcd_dirs), "dicom files not compatible with pcd files!"

    os.makedirs(params['images_path'], exist_ok=True)
    os.makedirs(params['labels_path'], exist_ok=True)
    
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
        
        for jt in range(params['augment_size']) :
            displacement = params['displacement']
            half_patch = int(params['patch_size'] / 2)

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
            image_number = str(it * params['augment_size'] + jt + params['train_data_size'] + 1).zfill(4)
            sitk.WriteImage(dicom_part, f"{params['images_path']}/IMPLANT_{image_number}_0000.nii.gz")
            sitk.WriteImage(pcd_part, f"{params['labels_path']}/IMPLANT_{image_number}.nii.gz")
        
        print(f"{str(it+1).zfill(4)} done! midx:{midx} midy:{midy} midz:{midz}")


if __name__ == '__main__':

    params = {
        'dicom_path' : './data/data/01_mimics/',
        'pcd_path' : './data/data/02_pointcloud/',
        'split_path' : "./data/data/splits.json",
        'augment_size' : 10,
        'displacement' : 16,
        'patch_size' : 96,
        'images_path' : './data/nii/val/raw',
        'labels_path' : './data/nii/val/label',
        'train_data_size' : 1220
    }

    val_to_nii(params)
