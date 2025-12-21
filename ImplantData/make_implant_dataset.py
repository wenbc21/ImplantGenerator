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
    parser.add_argument('--patch_size', type=int, default=96)
    parser.add_argument('--displacement', type=int, default=16)
    parser.add_argument('--random_seed', type=int, default=21)
    parser.add_argument('--results_path', type=str, default='ImplantData/datasets/ImplantGeneration')

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

    for d in ["imagesTr", "imagesTs", "labelsTr", "labelsTs"]:
        os.makedirs(os.path.join(results_path, d), exist_ok=True)
    
    for it in range(len(cbct_data_path)) :
        # get dicom file and stl file
        cbct_path = cbct_data_path[it]
        stl_path = stl_data_path[it]
        data_name = os.path.split(cbct_path)[-1][:3]
        if data_name in train_split :
            is_test = False
        if data_name in val_split :
            is_test = True
        
        dicom = get_dcm_3d_array(cbct_path)
        if metadata[data_name]["spacing"] != args.spacing:
            dicom = rescale(dicom, metadata[data_name]["spacing"] / args.spacing, order=1, preserve_range=True)
        dicom = window_transform_3d(dicom, metadata[data_name]["width"], metadata[data_name]["window"]).astype(np.uint8)
        
        cylinders = get_stl(stl_path)
        cylinders = cylinder_transform(
            cylinders, 
            dicom.shape, 
            args.spacing
        )
        
        for cylinder_i, cylinder in enumerate(cylinders) :
            item_name = f"IMPLANT_{dataset_name}_{data_name}_{cylinder_i}"
            upper_center, lower_center, centroid, radius, length, direction = cylinder
            patch_size = args.patch_size + args.displacement * 2
            half_patch = int(args.patch_size / 2) + args.displacement
            implant_part = cylinder_render(half_patch, patch_size, direction, length, radius)

            midx = int(centroid[0])
            midy = int(centroid[1])
            midz = int(centroid[2])

            # rigorously this should be half_patch, but 48 is enough and efficient
            pad_size = 48
            dicom_pad = np.pad(dicom, pad_width=[(pad_size,)]*3, mode='constant', constant_values=0)
            dicom_part = dicom_pad[
                int(midx - half_patch + pad_size): int(midx + half_patch + pad_size), 
                int(midy - half_patch + pad_size): int(midy + half_patch + pad_size), 
                int(midz - half_patch + pad_size): int(midz + half_patch + pad_size), 
            ]

            print("upper", upper_center, "lower", lower_center)
            print("centroid", centroid, "direction", direction)
            print("radius", radius, "length", length)
            
            dataset_configs[item_name] = {
                "upper_center": upper_center.tolist(),
                "lower_center": lower_center.tolist(),
                "centroid": centroid.tolist(),
                "radius": float(radius),
                "length": float(length),
                "direction": direction.tolist(), 
                "patch_size": patch_size,
                "class_label": metadata[data_name]["fdi"],
            }

            # write images
            dicom_part_img = sitk.GetImageFromArray(dicom_part)
            implant_part_img = sitk.GetImageFromArray(implant_part)
            sitk.WriteImage(dicom_part_img, os.path.join(results_path, f"imagesTr", f"{item_name}_0000.nii.gz"))
            sitk.WriteImage(implant_part_img, os.path.join(results_path, f"labelsTr", f"{item_name}.nii.gz"))
            
            if is_test:
                implant = cylinder_render(centroid, dicom.shape, direction, length, radius)
                dicom_img = sitk.GetImageFromArray(dicom)
                implant_img = sitk.GetImageFromArray(implant)
                sitk.WriteImage(dicom_img, os.path.join(results_path, f"imagesTs", f"{item_name}_0000.nii.gz"))
                sitk.WriteImage(implant_img, os.path.join(results_path, f"labelsTs", f"{item_name}.nii.gz"))
                splits_final["val"].append(item_name)
            else :
                splits_final["train"].append(item_name)

            os.makedirs(os.path.join(results_path, "combine_slices", item_name), exist_ok=True)
            for i in range(dicom_part.shape[0]) :
                cross_sec_img = dicom_part[i, :, :]
                cross_sec_implant = implant_part[i, :, :]
                cross_sec_img[cross_sec_implant == 1] = 255
                cv2.imwrite(os.path.join(results_path, "combine_slices", item_name, f"cross_section_{i}.png"), cross_sec_img)
            
            print(item_name, metadata[data_name]["spacing"], os.path.split(cbct_path)[-1])
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
