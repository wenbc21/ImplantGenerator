import os
import numpy as np
import argparse
from tqdm import tqdm
import SimpleITK as sitk
from data_utils import *
import json

def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--dataset_dir', type=str, default='ImplantData/datasets/ImplantGeneration/UpperAnterior')
    parser.add_argument('--location_path', type=str, default='results/ImplantLocation_UpperAnterior/location.json')

    return parser


def make_dataset(args):
    images_ts = [item.path for item in os.scandir(os.path.join(args.dataset_dir, "imagesTs")) if item.is_file()]
    labels_ts = [item.path for item in os.scandir(os.path.join(args.dataset_dir, "labelsTs")) if item.is_file()]
    images_ts.sort()
    labels_ts.sort()

    with open(args.location_path, 'r') as file:
        location = json.load(file)
    
    os.makedirs(os.path.join(args.dataset_dir, "imagesInfer"), exist_ok=True)
    os.makedirs(os.path.join(args.dataset_dir, "labelsInfer"), exist_ok=True)
    for it in range(len(images_ts)):
        data_name = os.path.split(labels_ts[it])[-1].split('.')[0]

        images = sitk.ReadImage(images_ts[it])
        images = sitk.GetArrayFromImage(images)
        labels = sitk.ReadImage(labels_ts[it])
        labels = sitk.GetArrayFromImage(labels)

        c = np.array(location[data_name]).astype(int)
        images = images[c[0]-48:c[0]+48, c[1]-48:c[1]+48, c[2]-48:c[2]+48]
        labels = labels[c[0]-48:c[0]+48, c[1]-48:c[1]+48, c[2]-48:c[2]+48]
        
        images = sitk.GetImageFromArray(images)
        labels = sitk.GetImageFromArray(labels)
        sitk.WriteImage(images, os.path.join(args.dataset_dir, "imagesInfer", f"{data_name}_0000.nii.gz"))
        sitk.WriteImage(labels, os.path.join(args.dataset_dir, "labelsInfer", f"{data_name}.nii.gz"))

        print(data_name, "done")


if __name__ == '__main__':

    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()

    make_dataset(args)
