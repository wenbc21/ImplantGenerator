import SimpleITK as sitk
import os
import argparse
import numpy as np
from get_dicom import *
from get_stl import *
import pydicom
import random


def get_args_parser():
    parser = argparse.ArgumentParser('Rebuild dicom files from nnU-Net results', add_help=False)
    parser.add_argument('--random_seed', type=int, default=21)
    parser.add_argument('--data_path', type=str, default="datasets/data")
    parser.add_argument('--label_path', type=str, default='datasets/implant_dataset/labelsTs')
    parser.add_argument('--predict_path', type=str, default='datasets/implant_dataset/predict')
    parser.add_argument('--results_path', type=str, default='datasets/rebuild/rebuild_dicom')

    return parser


def rebuild_dicom(args) :

    # read raw data
    data_path = [item.path for item in os.scandir(args.data_path) if item.is_dir()]
    label_dirs = [item.path for item in os.scandir(args.label_path) if item.is_file()]
    predict_dirs = [item.path for item in os.scandir(args.predict_path) if item.is_file()]
    data_path.sort()
    label_dirs.sort()
    predict_dirs.sort()

    data_id = list(range(len(data_path)))
    random.seed(args.random_seed)
    random.shuffle(data_id)
    test_split = data_id[round(0.8*len(data_id)):]
    test_split.sort()
    
    os.makedirs(args.results_path, exist_ok=True)

    # rebuild dicom file for each item
    for it in range(len(test_split)) :
        predict = sitk.ReadImage(predict_dirs[it])
        predict = sitk.GetArrayFromImage(predict)

        dicom_dir = list([item.path for item in os.scandir(data_path[test_split[it]]) if item.is_dir()])[0]
        dicom_slice = [item.path for item in os.scandir(dicom_dir) if item.is_file()]
        dicom_slice.sort()
        dicom_slice = dicom_slice[:-1]
        stl_file = list([item.path for item in os.scandir(data_path[test_split[it]]) if item.is_file() and item.path.endswith(".stl")])[0]
        dicom = get_dcm_3d_array(dicom_dir)
        label = get_stl(stl_file, dicom.shape)

        # get central position of cylinder
        cylinder = np.array(np.where(label == 1))
        midx = (np.min(cylinder[0]) + np.max(cylinder[0])) // 2
        midy = (np.min(cylinder[1]) + np.max(cylinder[1])) // 2
        midz = (np.min(cylinder[2]) + np.max(cylinder[2])) // 2
        
        # space transfer
        predict = np.array(np.where(predict == 1))
        predict[0] += (midx - 48)
        predict[1] += (midy - 48)
        predict[2] += (midz - 48)
        predict = predict.T


        # # get dicom file and implant file
        # dicom_dir = list([item.path for item in os.scandir(dicom_dirs[it]) if item.is_dir()])[0]
        # dicom_slice = [item.path for item in os.scandir(dicom_dir) if item.is_file()]
        # dicom_slice.sort()
        # dicom_slice = dicom_slice[:-1]
        # implant = sitk.ReadImage(implant_dirs[it])
        # implant = sitk.GetArrayFromImage(implant)
        # implant = np.array(np.where(implant == 1))

        # read original dicom
        dicom = get_dcm_3d_array(dicom_dir)
        dicom_size = dicom.shape
        dicom = np.zeros(dicom_size, dtype = np.int16)
        for i in range(dicom_size[0]) :
            data = pydicom.dcmread(dicom_slice[i])
            databytes = data.PixelData
            image_np = np.frombuffer(databytes, dtype=np.int16).reshape(dicom_size[1:])
            dicom[i] = image_np
        dicom = dicom[::-1, :, :]

        # # get labeled mid point
        # with open(mid_dirs[it]) as f:
        #     mid = json.load(f)
        # midx = dicom.shape[0] - mid['midx']
        # midy = mid['midz']
        # midz = mid['midy']

        # # space transfer
        # implant[0] += (midx - 48)
        # implant[1] += (midy - 48)
        # implant[2] += (midz - 48)
        # implant = implant.T
        
        # rebuild dicom
        max_dicom = np.max(dicom)
        for p in predict :
            dicom[p[0]][p[1]][p[2]] = max_dicom
        dicom = dicom[::-1, :, :]
        
        # write dicom file
        os.makedirs(f"{args.results_path}/{str(it+1).zfill(3)}", exist_ok=True)
        for i in range(dicom_size[0]) :
            data = pydicom.dcmread(dicom_slice[i])
            data.pixel_array.data = dicom[i]
            data.PixelData = dicom[i].tobytes()
            data.save_as(f"{args.results_path}/{str(it+1).zfill(3)}/predict_{str(i).zfill(3)}.dcm")
        
        print(f"{str(it+1).zfill(3)} done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Rebuild dicom files from nnU-Net results', parents=[get_args_parser()])
    args = parser.parse_args()

    rebuild_dicom(args)
