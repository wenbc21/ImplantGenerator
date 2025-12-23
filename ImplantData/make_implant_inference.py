import os
import numpy as np
import argparse
import SimpleITK as sitk
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
        data_name = os.path.basename(labels_ts[it]).split('.')[0]

        image = sitk.ReadImage(images_ts[it])
        image = sitk.GetArrayFromImage(image)
        label = sitk.ReadImage(labels_ts[it])
        label = sitk.GetArrayFromImage(label)

        centroids = location[data_name]
        for i in range(len(centroids)) :
            c = np.array(centroids[i]).astype(int)
            image_roi = image[c[0]-48:c[0]+48, c[1]-48:c[1]+48, c[2]-48:c[2]+48]
            label_roi = label[c[0]-48:c[0]+48, c[1]-48:c[1]+48, c[2]-48:c[2]+48]
            
            image_roi = sitk.GetImageFromArray(image_roi)
            label_roi = sitk.GetImageFromArray(label_roi)
            sitk.WriteImage(image_roi, os.path.join(args.dataset_dir, "imagesInfer", f"{data_name}_{i}_0000.nii.gz"))
            sitk.WriteImage(label_roi, os.path.join(args.dataset_dir, "labelsInfer", f"{data_name}_{i}.nii.gz"))

            print(f"{data_name}_{i}, image shape: {image.shape}, ROI: {c}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()

    make_dataset(args)
