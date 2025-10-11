# ImplantGenerator

Virtual dental implant placement in cone-beam computed tomography (CBCT) is a prerequisite for digital implant surgery, carrying clinical significance. This study aims to achieve intelligent virtual dental implant placement through a 3-dimensional (3D) segmentation strategy, generating virtual implant from the edentulous region of CBCT and employed an approximation module for mathematical optimization. The tool demonstrated good performance in predicting both the dimension and position of the virtual implants, showing significant clinical application potential in implant planning.

## Setup Instructions
```bash
conda create -n impgen python=3.8
conda activate impgen
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Usage
1. Make datasets by [make_implant_dataset.py](ImplantData/make_implant_dataset.py) and [make_location_dataset.py](ImplantData/make_location_dataset.py).
2. Use [train.py](train.py) (or [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)) to train location and generation models.
3. Use [test.py](test.py) (or [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)) to do inference.
4. (Optional) Use [postprocessing.py](ImplantData/postprocessing.py) to get a standard cylinder.
5. Use [evaluation.py](ImplantData/evaluation.py) to evaluate generated implants.
6. Use [image_rebuild.py](ImplantData/image_rebuild.py) to rebuild augmented images (nii.gz and dicom) for further usage.
