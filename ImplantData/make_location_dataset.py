import SimpleITK as sitk
import os
import random
import argparse
import json
import gc
import csv
from skimage.transform import rescale
from data_utils import *
import cv2


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--data_path', type=str, default='ImplantData/data/UpperPosterior')
    parser.add_argument('--task', type=str, default='UpperPosterior')
    parser.add_argument('--spacing', type=float, default=0.3)
    parser.add_argument('--random_seed', type=int, default=21)
    parser.add_argument('--results_path', type=str, default='ImplantData/datasets/ImplantLocation')

    return parser


def make_dataset(args) :

    cbct_data_path = [item.path for item in os.scandir(f"{args.data_path}/CBCT") if item.is_dir()]
    cbct_data_path.sort()
    stl_data_path = [item.path for item in os.scandir(f"{args.data_path}/STL") if item.is_file()]
    stl_data_path.sort()
    assert len(cbct_data_path) == len(stl_data_path), "data number not aligned!"

    dataset_name = args.task
    results_path = os.path.join(args.results_path, dataset_name)

    metadata, train_split, val_split = get_dataset_metadata(args.data_path, dataset_name, args.random_seed)
    splits_final = {"train": [], "val": []}
    dataset_configs = {}

    for d in ["imagesTr", "imagesTs", "labelsTr", "labelsTs", "mip"]:
        os.makedirs(os.path.join(results_path, d), exist_ok=True)
    
    for it in range(len(cbct_data_path)) :
        # get dicom file and stl file
        cbct_path = cbct_data_path[it]
        stl_path = stl_data_path[it]
        data_name = os.path.split(cbct_path)[-1][:3]
        item_name = f"IMPLANT_{dataset_name}_{data_name}"
        if data_name in train_split :
            is_test = False
        if data_name in val_split :
            is_test = True
        
        dicom = get_dcm_3d_array(cbct_path)
        if metadata[data_name]["spacing"] != args.spacing:
            dicom = rescale(dicom, metadata[data_name]["spacing"] / args.spacing, order=1, preserve_range=True)
        cs_upper, cs_lower, cs_front, cs_rear, cs_left, cs_right = get_cross_section(
            dicom, True, data_name, results_path, metadata[data_name]["mip_window"], metadata[data_name]["mip_width"])
        dicom = window_transform_3d(dicom, metadata[data_name]["width"], metadata[data_name]["window"]).astype(np.uint8)
        
        cylinders = get_stl(stl_path)
        cylinders = cylinder_transform(
            cylinders, 
            dicom.shape, 
            args.spacing
        )

        implant = np.zeros_like(dicom, dtype=np.uint8)
        for cylinder_i, cylinder in enumerate(cylinders) :
            upper_center, lower_center, centroid, radius, length, direction = cylinder
            implant |= cylinder_render(centroid, dicom.shape, direction, length, radius)
            print("upper", upper_center, "lower", lower_center)
            print("centroid", centroid, "direction", direction)
            print("radius", radius, "length", length)

        dataset_configs[item_name] = {
            "region": [cs_upper, cs_lower, cs_front, cs_rear, cs_left, cs_right],
            "class_label": metadata[data_name]["fdi"],
        }
        
        # get image
        dicom_part = dicom[cs_upper:cs_lower, cs_front:cs_rear, cs_left:cs_right]
        implant_part = implant[cs_upper:cs_lower, cs_front:cs_rear, cs_left:cs_right]
        dicom_part = rescale(dicom_part, 0.5, order=1, preserve_range=True)
        implant_part = rescale(implant_part, 0.5, order=1, preserve_range=True)
        implant_part = np.round(implant_part).astype(np.uint8)
        
        # write images
        dicom_part_img = sitk.GetImageFromArray(dicom_part)
        implant_part_img = sitk.GetImageFromArray(implant_part)
        if is_test:
            sitk.WriteImage(dicom_part_img, os.path.join(results_path, f"imagesTs", f"{item_name}_0000.nii.gz"))
            sitk.WriteImage(implant_part_img, os.path.join(results_path, f"labelsTs", f"{item_name}.nii.gz"))
            splits_final["val"].append(item_name)
        else :
            sitk.WriteImage(dicom_part_img, os.path.join(results_path, f"imagesTr", f"{item_name}_0000.nii.gz"))
            sitk.WriteImage(implant_part_img, os.path.join(results_path, f"labelsTr", f"{item_name}.nii.gz"))
            splits_final["train"].append(item_name)
        
        os.makedirs(os.path.join(results_path, "combine_slices", item_name), exist_ok=True)
        for i in range(dicom_part.shape[0]) :
            cross_sec_img = dicom_part[i, :, :]
            cross_sec_implant = implant_part[i, :, :]
            cross_sec_img[cross_sec_implant == 1] = 255
            cv2.imwrite(os.path.join(results_path, "combine_slices", item_name, f"cross_section_{i}.png"), cross_sec_img)
        
        print(item_name, metadata[data_name]["spacing"], dicom_part.shape, os.path.split(cbct_path)[-1])
        print()
        gc.collect()
    
    with open(os.path.join(results_path, "splits_final.json"), 'w', encoding='utf-8') as sf:
        json.dump(splits_final, sf, indent=4)
    with open(os.path.join(results_path, "metadata.json"), 'w', encoding='utf-8') as sf:
        json.dump(dataset_configs, sf, indent=4)

if __name__ == '__main__':

    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()

    make_dataset(args)
