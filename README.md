# ImplantDataProcess

The official repository of my bachelor thesis 
```
Automatic Planning of Implant Surgery for Mandibular Posterior Teeth based on Deep Learning
```

## How to use

1. Run [make_region_dataset](make_region_dataset.py) and [make_implant_dataset](make_implant_dataset.py) to preprocess data (transform raw CBCT and implant data into nii.gz format)
2. Install [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) to train a model and inference on the test set
3. Run [loss_segment](loss_segment.py) and [loss_spacial](loss_spacial.py) to calculate metrics for the results
4. (Optional) Run [fit_cylinder](fit_cylinder.py) to get a standard cylinder
5. Run [rebuild_nii](rebuild_nii.py) and [rebuild_dicom](rebuild_dicom.py) to rebuild augmented images for doctor usage

## Other files
Other than these, you may check out: 
[get_dicom](get_dicom.py) for reading dicom images, 
[get_pcd](get_pcd.py) for reading point cloud format implants, 
[get_stl](get_stl.py) for reading stl format implants, 
[nii_to_image](nii_to_image.py) to check your intermediate result.
