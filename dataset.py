import os
import glob
import json
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from monai import transforms, data
from torch.utils.data import Dataset


class ImplantDataset(Dataset):
    def __init__(self, datalist, metadata, transform=None, cache=False) -> None:
        super().__init__()
        self.transform = transform
        self.datalist = datalist
        self.metadata = metadata
        self.cache = cache
        if cache:
            self.cache_data = []
            for i in tqdm(range(len(datalist)), total=len(datalist)):
                d = self.read_data(datalist[i])
                self.cache_data.append(d)

    def read_data(self, data_path):
        
        image_path = data_path[0]
        label_path = data_path[1]
        item_name = os.path.basename(label_path).split(".")[0]

        image_data = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        seg_data = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        seg_data = np.expand_dims(seg_data, axis=0).astype(np.int32)

        info = {
            "name": item_name,
            "class_label": self.metadata[item_name]["class_label"]
        }
        if "region" in self.metadata[item_name]:
            info["region"] = self.metadata[item_name]["region"]

        return {
            "image": image_data,
            "label": seg_data,
            "info": info
        }

    def __getitem__(self, i):
        if self.cache:
            image = self.cache_data[i]
        else :
            image = self.read_data(self.datalist[i])
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.datalist)


def get_implant_dataset(data_dir, cache=True, is_train=True):
    
    with open(f"{data_dir}/splits_final.json", 'r') as file:
        data_split = json.load(file)
    with open(f"{data_dir}/metadata.json", 'r') as file:
        metadata = json.load(file)

    all_images = sorted(glob.glob(f"{data_dir}/imagesTr/*.nii.gz"))
    all_labels =  sorted(glob.glob(f"{data_dir}/labelsTr/*.nii.gz"))
    train_files = []
    val_files = []
    for i in range(len(all_images)) :
        if os.path.basename(all_labels[i]).split('.')[0] in data_split["train"]:
            train_files.append([all_images[i], all_labels[i]])
        elif os.path.basename(all_labels[i]).split('.')[0] in data_split["val"]:
            val_files.append([all_images[i], all_labels[i]])
        else: 
            print("Data", os.path.basename(all_labels[i]), "not in any split.")
            exit()

    all_images = sorted(glob.glob(f"{data_dir}/imagesInfer/*.nii.gz"))
    all_labels =  sorted(glob.glob(f"{data_dir}/labelsInfer/*.nii.gz"))
    test_files = [[all_images[i], all_labels[i]] for i in range(len(all_images))]

    train_transform = transforms.Compose([
        transforms.ScaleIntensityRanged(
            keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True
        ),
        transforms.RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=4,
        ),
        # only sagittal flip is enabled because of tooth prior
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.2),
        transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.2),
        transforms.ToTensord(keys=["image", "label"],),
    ])
    val_transform = transforms.Compose([
        transforms.ScaleIntensityRanged(
            keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True
        ),
        transforms.RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=8,
        ),
        transforms.ToTensord(keys=["image", "label"]),
    ])
    test_transform = transforms.Compose([
        transforms.ScaleIntensityRanged(
            keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True
        ),
        transforms.ToTensord(keys=["image", "label"]),
    ])
    
    if is_train:
        train_ds = ImplantDataset(train_files, metadata, transform=train_transform, cache=cache)
        val_ds = ImplantDataset(val_files, metadata, transform=val_transform, cache=cache)
        return train_ds, val_ds
    else:
        return ImplantDataset(test_files, metadata, transform=test_transform)


def get_location_dataset(data_dir, cache=True, is_train=True):
    
    with open(f"{data_dir}/splits_final.json", 'r') as file:
        data_split = json.load(file)
    with open(f"{data_dir}/metadata.json", 'r') as file:
        metadata = json.load(file)

    all_images = sorted(glob.glob(f"{data_dir}/imagesTr/*.nii.gz"))
    all_labels =  sorted(glob.glob(f"{data_dir}/labelsTr/*.nii.gz"))
    train_files = []
    val_files = []
    for i in range(len(all_images)) :
        if os.path.basename(all_labels[i]).split('.')[0] in data_split["train"]:
            train_files.append([all_images[i], all_labels[i]])
        elif os.path.basename(all_labels[i]).split('.')[0] in data_split["val"]:
            val_files.append([all_images[i], all_labels[i]])
        else: 
            print("Data", os.path.basename(all_labels[i]), "not in any split.")
            exit()

    all_images = sorted(glob.glob(f"{data_dir}/imagesTs/*.nii.gz"))
    all_labels =  sorted(glob.glob(f"{data_dir}/labelsTs/*.nii.gz"))
    test_files = [[all_images[i], all_labels[i]] for i in range(len(all_images))]

    train_transform = transforms.Compose([
        transforms.ScaleIntensityRanged(
            keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True
        ),
        transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.2),
        transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.2),
        transforms.ToTensord(keys=["image", "label"],),
    ])
    test_transform = transforms.Compose([
        transforms.ScaleIntensityRanged(
            keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True
        ),
        transforms.ToTensord(keys=["image", "label"]),
    ])

    if is_train:
        train_ds = ImplantDataset(train_files, metadata, transform=train_transform, cache=cache)
        val_ds = ImplantDataset(test_files, metadata, transform=test_transform, cache=cache)
        return train_ds, val_ds
    else:
        return ImplantDataset(test_files, metadata, transform=test_transform)
