import SimpleITK as sitk
import os
from medpy.metric.binary import dc, jc, hd95, asd
import numpy as np
import surface_distance as surdist


if __name__ == '__main__':
    
    # get predict and groundtruth   
    label_path = 'datasets/implant_dataset/labelsTs'
    predict_path = 'datasets/implant_dataset/predict'    
    label_dirs = [item.path for item in os.scandir(label_path) if item.is_file()]
    predict_dirs = [item.path for item in os.scandir(predict_path) if item.is_file()]
    label_dirs.sort()
    predict_dirs.sort()

    dice_list = []
    hd95_list = []
    asd_list = []
    iou_list = []
    suriou_list = []
    surdice_list = []
        
    print("                 dice       IOU       95%HD       ASD       surIOU      surDice")
        
    for it in range(len(label_dirs)) :
        # get files
        label_dir = label_dirs[it]
        predict_dir = predict_dirs[it]
        label = sitk.ReadImage(label_dir)
        label = sitk.GetArrayFromImage(label)
        predict = sitk.ReadImage(predict_dir)
        predict = sitk.GetArrayFromImage(predict)
        
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

        # record and save
        dice_list.append(dice)
        hd95_list.append(HD95)
        asd_list.append(ASD)
        iou_list.append(IOU)
        suriou_list.append(suriou)
        surdice_list.append(surdice)
        
        dice = "%.6f" % dice
        HD95 = "%.6f" % HD95
        ASD = "%.6f" % ASD
        IOU = "%.6f" % IOU
        suriou = "%.6f" % suriou
        surdice = "%.6f" % surdice
        
        print(f"predict {str(it+1).zfill(3)}: | {dice} | {IOU} | {HD95} | {ASD} | {suriou} | {surdice}")

    print(f"average: | {np.mean(np.array(dice_list))} | {np.mean(np.array(iou_list))} | {np.mean(np.array(hd95_list))} | {np.mean(np.array(asd_list))} | {np.mean(np.array(suriou_list))} | {np.mean(np.array(surdice_list))}")
    
    loss_segment_list = np.array([dice_list, iou_list, hd95_list, asd_list, suriou_list, surdice_list])
    np.savetxt("./loss_segment.csv", loss_segment_list.T, fmt="%.6f", delimiter=',')
