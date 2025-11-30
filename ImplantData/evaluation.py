import os
import csv
import numpy as np
import argparse
from tqdm import tqdm
import SimpleITK as sitk
from medpy.metric.binary import dc, jc, hd95, asd
import surface_distance as surdist
from data_utils import *
from tabulate import tabulate


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--data_path', type=str, default='ImplantData/datasets/ImplantLocation/UpperAnterior/labelsTs')
    parser.add_argument('--results_path', type=str, default='results/ImplantLocation_UpperAnterior/predict')

    return parser


def evaluate(args):
    # get predict and groundtruth
    label_dirs = [item.path for item in os.scandir(args.data_path) if item.is_file()]
    predict_dirs = [item.path for item in os.scandir(args.results_path) if item.is_file()]
    label_dirs.sort()
    predict_dirs.sort()

    # Prepare CSV file
    table_data = []
    headers = [
        "Name", "Dice", "HD95", "ASD", "IOU", "SurIOU", "SurDice",
        "LengthPre", "LengthGT", "DiameterPre", "DiameterGT", 
        "DiameterLoss", "LengthLoss", "ImpPos", "EndPos", "Angle", "ImpDep", "EndDep",
        "X_Dist", "Y_Dist", "Z_Dist", "EuclideanDist"
    ]
    csv_file = os.path.join(os.path.dirname(args.results_path), 'evaluation_metrics.csv')
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(headers)
    
    for it in tqdm(range(len(label_dirs))):
        data_name = os.path.split(label_dirs[it])[-1].split('.')[0]
        
        # get files
        label = sitk.ReadImage(label_dirs[it])
        label = sitk.GetArrayFromImage(label)
        predict = sitk.ReadImage(predict_dirs[it])
        predict = sitk.GetArrayFromImage(predict)
        predict = keep_largest_component(predict)

        Dice, HD95, ASD, IOU, surdistance, suriou, surdice = get_segmentation_metrics(label, predict)
        diameter, length, inplant_position, endpoint_position, angle, \
            implant_depth, endpoint_depth, radius_p, length_p, radius_g, length_g, \
            x_dist, y_dist, z_dist, euc_dist = get_spatial_metrics(label, predict)
        
        # Format metrics to 4 decimal places
        metrics = [
            f"{Dice:.4f}", f"{HD95:.4f}", f"{ASD:.4f}", f"{IOU:.4f}", 
            f"{suriou:.4f}", f"{surdice:.4f}",
            f"{length_p:.4f}", f"{length_g:.4f}", f"{radius_p*2:.4f}", f"{radius_g*2:.4f}", 
            f"{diameter:.4f}", f"{length:.4f}", f"{inplant_position:.4f}", 
            f"{endpoint_position:.4f}", f"{angle:.4f}",
            f"{implant_depth:.4f}", f"{endpoint_depth:.4f}",
            f"{x_dist:.4f}", f"{y_dist:.4f}", f"{z_dist:.4f}", f"{euc_dist:.4f}",
        ]
        
        # Add to table data
        table_data.append([data_name] + metrics)
        
        # Write to CSV
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([data_name] + metrics)
    
    print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".4f"))


def get_segmentation_metrics(label, predict, spacing=0.3):
    # calculate segmentation loss of cylinder
    dice = dc(predict, label)
    HD95 = hd95(predict, label, 0.3)
    ASD = asd(predict, label, 0.3)
    IOU = jc(predict, label)
    predict = predict.astype(bool)
    label = label.astype(bool)
    surdistance = surdist.compute_surface_distances(label, predict, spacing_mm=(0.3, 0.3, 0.3))
    suriou = np.average(surdist.compute_surface_overlap_at_tolerance(surdistance, 1))
    surdice = surdist.compute_surface_dice_at_tolerance(surdistance, 1)
    return dice, HD95, ASD, IOU, surdistance, suriou, surdice


def get_spatial_metrics(label, predict):
    # compute cylinder parameters for predict and groundtruth
    center_p, direction_p, radius_p, length_p = get_cylinder_param(predict, approx=True)
    center_g, direction_g, radius_g, length_g = get_cylinder_param(label)
    
    # get centers of the top and bottom faces
    upper_center_p = center_p + length_p * direction_p * 0.5 / 0.3
    bottom_center_p = center_p - length_p * direction_p * 0.5 / 0.3
    upper_center_g = center_g + length_g * direction_g * 0.5 / 0.3
    bottom_center_g = center_g - length_g * direction_g * 0.5 / 0.3
    
    # calculate spacial loss of cylinder
    diameter = radius_p * 2 - radius_g * 2
    length = length_p - length_g
    inplant_position = np.linalg.norm(upper_center_p - upper_center_g) * 0.3
    endpoint_position = np.linalg.norm(bottom_center_p - bottom_center_g) * 0.3
    angle = np.arccos(np.dot(direction_g, direction_p)) * 180 / np.pi
    implant_lateral1 = (upper_center_p[1] - upper_center_g[1]) * 0.3
    implant_lateral2 = (upper_center_p[2] - upper_center_g[2]) * 0.3
    endpoint_lateral1 = (bottom_center_p[1] - bottom_center_g[1]) * 0.3
    endpoint_lateral2 = (bottom_center_p[2] - bottom_center_g[2]) * 0.3
    implant_depth = (upper_center_p[0] - upper_center_g[0]) * 0.3
    endpoint_depth = (bottom_center_p[0] - bottom_center_g[0]) * 0.3

    # distance loss
    x_dist_vxl = center_p[0] - center_g[0]
    y_dist_vxl = center_p[1] - center_g[1]
    z_dist_vxl = center_p[2] - center_g[2]
    euc_dist_vxl = np.linalg.norm(center_p - center_g)

    return diameter, length, inplant_position, endpoint_position, \
        angle, implant_depth, endpoint_depth, radius_p, length_p, radius_g, length_g, \
        x_dist_vxl, y_dist_vxl, z_dist_vxl, euc_dist_vxl


if __name__ == '__main__':

    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()

    evaluate(args)
