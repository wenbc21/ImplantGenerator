import SimpleITK as sitk
import os
from get_dicom import *
from get_stl import *
import random
import argparse
import json
import gc


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--data_path', type=str, default='datasets/data')
    parser.add_argument('--augment_size', type=int, default=10)
    parser.add_argument('--displacement', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=96)
    parser.add_argument('--random_seed', type=int, default=21)
    parser.add_argument('--results_path', type=str, default='datasets/implant_dataset')

    return parser


def make_dataset(args) :

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

    # os.makedirs(os.path.join(args.results_path, "mip"), exist_ok=True)
    os.makedirs(os.path.join(args.results_path, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(args.results_path, "imagesTs"), exist_ok=True)
    os.makedirs(os.path.join(args.results_path, "labelsTr"), exist_ok=True)
    os.makedirs(os.path.join(args.results_path, "labelsTs"), exist_ok=True)
    
    for it in range(len(data_path)) :
        # get dicom file and pcd file
        dicom_dir = list([item.path for item in os.scandir(data_path[it]) if item.is_dir()])[0]
        stl_file = list([item.path for item in os.scandir(data_path[it]) if item.is_file() and item.path.endswith(".stl")])[0]
        dicom = get_dcm_3d_array(dicom_dir)
        implant = get_stl(stl_file, dicom.shape)

        # get central position of cylinder
        cylinder = np.array(np.where(implant == 1))
        midx = (np.min(cylinder[0]) + np.max(cylinder[0])) // 2
        midy = (np.min(cylinder[1]) + np.max(cylinder[1])) // 2
        midz = (np.min(cylinder[2]) + np.max(cylinder[2])) // 2

        half_patch = int(args.patch_size / 2)
        midx_part = (midx - half_patch, midx + half_patch)
        midy_part = (midy - half_patch, midy + half_patch)
        midz_part = (midz - half_patch, midz + half_patch)
        dicom_part = dicom[midx_part[0]:midx_part[1], midy_part[0]:midy_part[1], midz_part[0]:midz_part[1]]
        implant_part = implant[midx_part[0]:midx_part[1], midy_part[0]:midy_part[1], midz_part[0]:midz_part[1]]
        
        # get image
        dicom = get_dcm_3d_array(dicom_dir)
        dicom = window_transform_3d(dicom, window_width=4000, window_center=1000).astype(np.uint8)
        dicom_part = dicom[midx_part[0]:midx_part[1], midy_part[0]:midy_part[1], midz_part[0]:midz_part[1]]
        implant_part = implant[midx_part[0]:midx_part[1], midy_part[0]:midy_part[1], midz_part[0]:midz_part[1]]
        print(it+1, dicom.shape, dicom_part.shape, os.path.split(dicom_dir)[-1])
        
        os.makedirs(os.path.join(args.results_path, "dicom_slices", str(it+1).zfill(4)), exist_ok=True)
        os.makedirs(os.path.join(args.results_path, "implant_slices", str(it+1).zfill(4)), exist_ok=True)
        os.makedirs(os.path.join(args.results_path, "combine_slices", str(it+1).zfill(4)), exist_ok=True)
        for i in range(dicom_part.shape[0]) :
            cross_sec_dicom = dicom_part[i, :, :]  # 从三维矩阵中找出横断面切片
            cross_sec_dicom = cross_sec_dicom.astype("uint8")  # 转换类型
            cv2.imwrite(os.path.join(args.results_path, "dicom_slices", str(it+1).zfill(4), f"cross_section_{i}.png"), cross_sec_dicom)  # 保存横断面
            
            cross_sec_implant = implant_part[i, :, :]  # 从三维矩阵中找出横断面切片
            cross_sec_implant = cross_sec_implant.astype("uint8")  # 转换类型
            cv2.imwrite(os.path.join(args.results_path, "implant_slices", str(it+1).zfill(4), f"cross_section_{i}.png"), cross_sec_implant * 255)  # 保存横断面

            cross_sec_dicom[cross_sec_implant == 1] = 255
            cv2.imwrite(os.path.join(args.results_path, "combine_slices", str(it+1).zfill(4), f"cross_section_{i}.png"), cross_sec_dicom)  # 保存横断面
        
        # write images
        dicom_part = sitk.GetImageFromArray(dicom_part)
        implant_part = sitk.GetImageFromArray(implant_part)
        if it in test_split :
            sitk.WriteImage(dicom_part, os.path.join(args.results_path, "imagesTs", f"IMPLANT_{str(it+1).zfill(4)}_0000.nii.gz"))
            sitk.WriteImage(implant_part, os.path.join(args.results_path, "labelsTs", f"IMPLANT_{str(it+1).zfill(4)}.nii.gz"))
            continue
        else :
            sitk.WriteImage(dicom_part, os.path.join(args.results_path, "imagesTr", f"IMPLANT_{str(it*args.augment_size+1).zfill(4)}_0000.nii.gz"))
            sitk.WriteImage(implant_part, os.path.join(args.results_path, "labelsTr", f"IMPLANT_{str(it* args.augment_size+1).zfill(4)}.nii.gz"))
            if it in train_split :
                split_dic["train"].append(f"IMPLANT_{str(it*args.augment_size+1).zfill(4)}")
            if it in val_split :
                split_dic["val"].append(f"IMPLANT_{str(it*args.augment_size+1).zfill(4)}")

        # augmentation
        for jt in range(args.augment_size - 1) :
            displacement = args.displacement

            # random displacement for [x,y,z] axis
            randomx = random.randint(0, displacement * 2) - displacement
            randomy = random.randint(0, displacement * 2) - displacement
            randomz = random.randint(0, displacement * 2) - displacement
            midx_part = (midx + randomx - half_patch, midx + randomx + half_patch)
            midy_part = (midy + randomy - half_patch, midy + randomy + half_patch)
            midz_part = (midz + randomz - half_patch, midz + randomz + half_patch)
            dicom_part = dicom[midx_part[0]:midx_part[1], midy_part[0]:midy_part[1], midz_part[0]:midz_part[1]]
            implant_part = implant[midx_part[0]:midx_part[1], midy_part[0]:midy_part[1], midz_part[0]:midz_part[1]]
            
            # random flip for [x,y,z] axis
            random_flip = random.randint(0, 3)
            dicom_part = np.rot90(dicom_part, k = random_flip, axes = (0, 1))
            implant_part = np.rot90(implant_part, k = random_flip, axes = (0, 1))
            random_flip = random.randint(0, 3)
            dicom_part = np.rot90(dicom_part, k = random_flip, axes = (0, 2))
            implant_part = np.rot90(implant_part, k = random_flip, axes = (0, 2))
            random_flip = random.randint(0, 3)
            dicom_part = np.rot90(dicom_part, k = random_flip, axes = (1, 2))
            implant_part = np.rot90(implant_part, k = random_flip, axes = (1, 2))
            
            # write images
            dicom_part = sitk.GetImageFromArray(dicom_part)
            implant_part = sitk.GetImageFromArray(implant_part)
            image_number = str(it * args.augment_size + jt + 2).zfill(4)
            sitk.WriteImage(dicom_part, os.path.join(args.results_path, "imagesTr", f"IMPLANT_{image_number}_0000.nii.gz"))
            sitk.WriteImage(implant_part, os.path.join(args.results_path, "labelsTr", f"IMPLANT_{image_number}.nii.gz"))

            if it in train_split :
                split_dic["train"].append(f"IMPLANT_{image_number}")
            if it in val_split :
                split_dic["val"].append(f"IMPLANT_{image_number}")
        
        gc.collect()
    
    for i in range(5) :
        splits_final.append(split_dic)
    with open(os.path.join(args.results_path, "splits_final.json"), 'w', encoding='utf-8') as sf:
        json.dump(splits_final, sf, indent=4)

if __name__ == '__main__':

    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()

    make_dataset(args)
