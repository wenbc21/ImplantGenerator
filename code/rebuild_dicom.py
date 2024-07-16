import SimpleITK as sitk
import os
import cv2
import numpy as np
from get_dicom import *
from get_stl import *
from get_pcd import *
import pydicom
import json


def rebuild_dicom(params) :

    # 文件路径
    dicom_dirs = [item.path for item in os.scandir(params['dicom_path']) if item.is_dir()]
    mid_dirs = [item.path for item in os.scandir(params['mid_path']) if item.is_file()]
    implant_dirs = [item.path for item in os.scandir(params['implant_path']) if item.is_file()]
    dicom_dirs.sort()
    mid_dirs.sort()
    implant_dirs.sort()

    with open(params['split_path']) as f:
        test_split = [int(i)-1 for i in json.load(f)['test']]
    dicom_dirs = [dicom_dirs[i] for i in test_split]

    assert len(dicom_dirs) == len(implant_dirs) and len(dicom_dirs) == len(mid_dirs), "dicom files not compatible with nii files!"
    
    for it in range(len(dicom_dirs)) :

        # get dicom file and implant file
        dicom_dir = list([item.path for item in os.scandir(dicom_dirs[it]) if item.is_dir()])[0]
        dicom_slice = [item.path for item in os.scandir(dicom_dir) if item.is_file()]
        dicom_slice.sort()
        dicom_slice = dicom_slice[:-1]
        implant = sitk.ReadImage(implant_dirs[it])
        implant = sitk.GetArrayFromImage(implant)
        implant = np.array(np.where(implant == 1))

        # read dicom
        dicom = get_dcm_3d_array(dicom_dir)
        dicom_size = dicom.shape
        dicom = np.zeros(dicom_size, dtype = np.int16)
        for i in range(dicom_size[0]) :
            data = pydicom.dcmread(dicom_slice[i])
            databytes = data.PixelData
            image_np = np.frombuffer(databytes, dtype=np.int16).reshape(dicom_size[1:])
            dicom[i] = image_np
        dicom = dicom[::-1, :, :]

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
        
        # write dicom file
        os.makedirs(params['target_path'], exist_ok=True)
        os.makedirs(f"{params['target_path']}/rebuild_dicom/{str(it+1).zfill(3)}", exist_ok=True)
        for i in range(dicom_size[0]) :
            data = pydicom.dcmread(dicom_slice[i])
            data.pixel_array.data = dicom[i]
            data.PixelData = dicom[i].tobytes()
            data.save_as(f"{params['target_path']}/rebuild_dicom/{str(it+1).zfill(3)}/predict_{str(i).zfill(3)}.dcm")
        
        print(f"{str(it+1).zfill(3)} done!")


if __name__ == '__main__':

    params = {
        'dicom_path' : './data/data/01_mimics/',
        'mid_path' : './data/data/labeled_midpoint',
        'implant_path' : './data/ImplantDataProcess/post_2',
        'split_path' : "./data/data/splits.json",
        'target_path' : "./data/data_trans/postprocessed_2"
    }

    rebuild_dicom(params)
