import os
import argparse
import SimpleITK as sitk
from data_utils import *


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--data_path', type=str, default='ImplantData/UpperAnterior')
    parser.add_argument('--spacing', type=float, default=0.3)
    parser.add_argument('--results_path', type=str, default='results/ImplantGeneration_UNet_UpperAnterior')

    return parser


def main(args) :
    predict_path = [item.path for item in os.scandir(f"{args.results_path}/predict") if item.is_file()]
    predict_path.sort()
    
    os.makedirs(os.path.join(args.results_path, "standard_implant"), exist_ok=True)
    
    for it in range(len(predict_path)) :
        data_name = os.path.basename(predict_path[it])

        predict = sitk.ReadImage(predict_path[it])
        predict = sitk.GetArrayFromImage(predict)

        center, direction, radius, length = get_cylinder_param(predict)
        standard_cylinder = cylinder_render(center, predict.shape[0], direction, length/0.3, radius/0.3)

        standard_cylinder = sitk.GetImageFromArray(standard_cylinder)
        sitk.WriteImage(standard_cylinder, os.path.join(args.results_path, "standard_implant", data_name))

        print(f"{data_name} Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Make standard cylinder from segmentation results', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)