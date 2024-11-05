# ImplantDataProcess

The repository of thesis Automatic Planning of Implant Surgery for Mandibular Posterior Teeth based on Deep Learning

## How to use

1. Run [dataset_split](code/dataset_split.py) to make a random split of the dataset
2. Run [train_to_nii](code/train_to_nii.py), [val_to_nii](code/val_to_nii.py), [test_to_nii](code/test_to_nii.py) to preprocess data (transform raw CBCT and implant data into nii.gz format)
3. Install [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) to train a model and inference on the test set
4. Run [loss_segment](code/loss_segment.py) and [loss_spacial](code/loss_spacial.py) to calculate metrics for the results
5. (Optional) Run [fit_cylinder](code/fit_cylinder.py) to get a standard cylinder
6. Run [rebuild_nii](code/rebuild_nii.py) and [rebuild_dicom](code/rebuild_dicom.py) to rebuild augmented images for doctor usage

## Other files
Other than these, you may check out: 
[get_dicom](code/get_dicom.py) for reading dicom images, 
[get_pcd](code/get_pcd.py) for reading point cloud format implants, 
[get_stl](code/get_stl.py) for reading stl format implants (which is not done yet), 
[nii_to_image](code/nii_to_image.py) to view a nii image at any time to check your intermediate result.
