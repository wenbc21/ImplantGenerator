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
    parser.add_argument('--data_path', type=str, default='ImplantData/data/UpperAnterior')
    parser.add_argument('--task', type=str, default='UpperAnterior')
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
    args.results_path = os.path.join(args.results_path, dataset_name)

    data_id = []
    train_split = []
    val_split = []
    spacing_dict = {}
    window_dict = {}
    width_dict = {}
    mip_window_dict = {}
    mip_width_dict = {}
    fdi_dict = {}
    with open(f'{args.data_path}/metadata.csv', mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        has_is_test = 'isTest' in reader.fieldnames
        for row in reader:
            id = row['ID'].zfill(3)
            data_id.append(id)
            spacing_dict[id] = float(row['Spacing'])
            window_dict[id] = float(row['Window'])
            width_dict[id] = float(row['Width'])
            mip_window_dict[id] = float(row['MIPWindow'])
            mip_width_dict[id] = float(row['MIPWidth'])
            fdi_dict[id] = row['FDI']
            if has_is_test :
                if int(row["isTest"]) == 1:
                    val_split.append(id)
                else :
                    train_split.append(id)
    if not has_is_test :
        random.seed(args.random_seed)
        random.shuffle(data_id)
        train_split = data_id[:round(0.8*len(data_id))]
        val_split = data_id[round(0.8*len(data_id)):]
    
    splits_final = {"train":[],"val":[]}
    metadata = {}

    os.makedirs(os.path.join(args.results_path, "mip"), exist_ok=True)
    os.makedirs(os.path.join(args.results_path, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(args.results_path, "imagesTs"), exist_ok=True)
    os.makedirs(os.path.join(args.results_path, "labelsTr"), exist_ok=True)
    os.makedirs(os.path.join(args.results_path, "labelsTs"), exist_ok=True)
    
    for it in range(len(cbct_data_path)) :
        # get dicom file and stl file
        cbct_path = cbct_data_path[it]
        stl_path = stl_data_path[it]
        data_name = os.path.split(cbct_path)[-1][:3]
        if data_name in train_split :
            splits_final["train"].append(f"IMPLANT_{dataset_name}_{data_name}")
            is_test = False
        if data_name in val_split :
            splits_final["val"].append(f"IMPLANT_{dataset_name}_{data_name}")
            is_test = True
        
        dicom = get_dcm_3d_array(cbct_path)
        if spacing_dict[data_name] != args.spacing:
            dicom = rescale(dicom, spacing_dict[data_name] / args.spacing, order=1, preserve_range=True)
        cs_upper, cs_lower, cs_front, cs_rear, cs_left, cs_right = get_cross_section(
            dicom, True, data_name, args.results_path, mip_window_dict[data_name], mip_width_dict[data_name])
        dicom = window_transform_3d(dicom, window_width=width_dict[data_name], window_center=window_dict[data_name]).astype(np.uint8)
        upper_center, lower_center, centroid, radius, length, direction = get_stl(stl_path)
        upper_center, lower_center, centroid, radius, length, direction = cylinder_transform(
            [upper_center, lower_center, centroid, radius, length, direction], 
            dicom.shape, 
            args.spacing
        )
        implant = cylinder_render(centroid, dicom.shape, direction, length, radius)

        print("upper", upper_center, "lower", lower_center)
        print("centroid", centroid, "direction", direction)
        print("radius", radius, "length", length)

        metadata[f"IMPLANT_{dataset_name}_{data_name}"] = {
            "region": [cs_upper, cs_lower, cs_front, cs_rear, cs_left, cs_right],
            "class_label": fdi_dict[data_name],
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
            sitk.WriteImage(dicom_part_img, os.path.join(args.results_path, f"imagesTs", f"IMPLANT_{dataset_name}_{data_name}_0000.nii.gz"))
            sitk.WriteImage(implant_part_img, os.path.join(args.results_path, f"labelsTs", f"IMPLANT_{dataset_name}_{data_name}.nii.gz"))
        else :
            sitk.WriteImage(dicom_part_img, os.path.join(args.results_path, f"imagesTr", f"IMPLANT_{dataset_name}_{data_name}_0000.nii.gz"))
            sitk.WriteImage(implant_part_img, os.path.join(args.results_path, f"labelsTr", f"IMPLANT_{dataset_name}_{data_name}.nii.gz"))
        
        os.makedirs(os.path.join(args.results_path, "combine_slices", data_name), exist_ok=True)
        for i in range(dicom_part.shape[0]) :
            cross_sec_img = dicom_part[i, :, :]
            cross_sec_implant = implant_part[i, :, :]
            cross_sec_img[cross_sec_implant == 1] = 255
            cv2.imwrite(os.path.join(args.results_path, "combine_slices", data_name, f"cross_section_{i}.png"), cross_sec_img)
        
        print(data_name, spacing_dict[data_name], dicom_part.shape, implant_part.shape, os.path.split(cbct_path)[-1])
        print()
        gc.collect()
    
    with open(os.path.join(args.results_path, "splits_final.json"), 'w', encoding='utf-8') as sf:
        json.dump(splits_final, sf, indent=4)
    with open(os.path.join(args.results_path, "metadata.json"), 'w', encoding='utf-8') as sf:
        json.dump(metadata, sf, indent=4)

if __name__ == '__main__':

    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()

    make_dataset(args)
