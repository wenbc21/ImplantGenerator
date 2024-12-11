import SimpleITK as sitk
import os
from get_dicom import *
from get_stl import *
from get_cross_section import get_cross_section
import random
import argparse
import json
import gc
from skimage.transform import rescale


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--data_path', type=str, default='datasets/cbct_stl')
    parser.add_argument('--augment_size', type=int, default=10)
    parser.add_argument('--displacement', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=96)
    parser.add_argument('--random_seed', type=int, default=21)
    parser.add_argument('--results_path', type=str, default='results')

    return parser


def train_to_nii(args) :

    data_path = [item.path for item in os.scandir(args.data_path) if item.is_dir()]
    data_path.sort()
    
    data_id = list(range(len(data_path)))
    random.seed(args.random_seed)
    random.shuffle(data_id)
    train_split = data_id[:round(0.64*len(data_id))]
    val_split = data_id[round(0.64*len(data_id)):round(0.8*len(data_id))]
    test_split = data_id[round(0.8*len(data_id)):]
    splits_final = []
    split_dic = {"train":[],"val":[]}

    os.makedirs(os.path.join(args.results_path, "mip"), exist_ok=True)
    os.makedirs(os.path.join(args.results_path, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(args.results_path, "imagesTs"), exist_ok=True)
    os.makedirs(os.path.join(args.results_path, "labelsTr"), exist_ok=True)
    os.makedirs(os.path.join(args.results_path, "labelsTs"), exist_ok=True)
    
    for it in range(len(data_path)) :
        # get dicom file and pcd file
        split_name = "Ts" if it in test_split else "Tr"
        if it in train_split :
            split_dic["train"].append(f"IMPLANT_{str(it+1).zfill(4)}")
        if it in val_split :
            split_dic["val"].append(f"IMPLANT_{str(it+1).zfill(4)}")
        
        dicom_dir = list([item.path for item in os.scandir(data_path[it]) if item.is_dir()])[0]
        stl_file = list([item.path for item in os.scandir(data_path[it]) if item.is_file() and item.path.endswith(".stl")])[0]
        dicom = get_dcm_3d_array(dicom_dir)
        implant = get_stl(stl_file, dicom.shape)
        cs_upper, cs_lower, cs_front, cs_rear, cs_left, cs_right = get_cross_section(dicom, if_vis=True, number=it+1, results_path=args.results_path)
        
        # get image
        dicom = get_dcm_3d_array(dicom_dir)
        dicom = window_transform_3d(dicom, window_width=4000, window_center=1000).astype(np.uint8)
        dicom_part = dicom[cs_upper:cs_lower, cs_front:cs_rear, cs_left:cs_right]
        implant_part = implant[cs_upper:cs_lower, cs_front:cs_rear, cs_left:cs_right]
        dicom_part = rescale(dicom_part, 0.5, order=1, preserve_range=True)
        implant_part = rescale(implant_part, 0.5, order=1, preserve_range=True)
        implant_part = np.round(implant_part).astype(np.uint8)
        print(it+1, dicom.shape, dicom_part.shape, os.path.split(dicom_dir)[-1])
        
        os.makedirs(os.path.join(args.results_path, "roi_slices", str(it+1).zfill(4)), exist_ok=True)
        os.makedirs(os.path.join(args.results_path, "implant_slices", str(it+1).zfill(4)), exist_ok=True)
        for i in range(dicom_part.shape[0]) :
            cross_sec_img = dicom_part[i, :, :]  # 从三维矩阵中找出横断面切片
            cross_sec_img = cross_sec_img.astype("uint8")  # 转换类型
            cv2.imwrite(os.path.join(args.results_path, "roi_slices", str(it+1).zfill(4), f"cross_section_{i}.png"), cross_sec_img)  # 保存横断面
            
            cross_sec_img = implant_part[i, :, :]  # 从三维矩阵中找出横断面切片
            cross_sec_img = cross_sec_img.astype("uint8")  # 转换类型
            cv2.imwrite(os.path.join(args.results_path, "implant_slices", str(it+1).zfill(4), f"cross_section_{i}.png"), cross_sec_img * 255)  # 保存横断面
        
        # write images
        dicom_part = sitk.GetImageFromArray(dicom_part)
        implant_part = sitk.GetImageFromArray(implant_part)
        sitk.WriteImage(dicom_part, os.path.join(args.results_path, f"images{split_name}", f"IMPLANT_{str(it+1).zfill(4)}_0000.nii.gz"))
        sitk.WriteImage(implant_part, os.path.join(args.results_path, f"labels{split_name}", f"IMPLANT_{str(it+1).zfill(4)}.nii.gz"))
        
        gc.collect()
    
    for i in range(5) :
        splits_final.append(split_dic)
    with open(os.path.join(args.results_path, "splits_final.json"), 'w', encoding='utf-8') as sf:
        json.dump(splits_final, sf, indent=4)

if __name__ == '__main__':

    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()

    train_to_nii(args)
