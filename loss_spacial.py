import SimpleITK as sitk
import numpy as np
from get_dicom import *
from get_stl import *
from get_cylinder_param import *
    

if __name__ == '__main__':
    
    # get predict and groundtruth
    label_path = 'datasets/implant_dataset/labelsTs'
    predict_path = 'datasets/implant_dataset/predict'
    label_dirs = [item.path for item in os.scandir(label_path) if item.is_file()]
    predict_dirs = [item.path for item in os.scandir(predict_path) if item.is_file()]
    label_dirs.sort()
    predict_dirs.sort()
    
    loss_spacial = []
    length_width = []

    print(" diameter    length  inplant_position    endpoint_position   angle   implant_lateral1    implant_lateral2    endpoint_lateral1   endpoint_lateral2   implant_depth   endpoint_depth")
    
    for it in range(len(label_dirs)) :
        # get files
        label_dir = label_dirs[it]
        predict_dir = predict_dirs[it]
        label = sitk.ReadImage(label_dir)
        label = sitk.GetArrayFromImage(label)
        predict = sitk.ReadImage(predict_dir)
        predict = sitk.GetArrayFromImage(predict)
        
        # compute cylinder parameters for predict and groundtruth
        center_p, direction_p, radius_p, length_p = get_cylinder_param(predict[::-1, :, :])
        center_g, direction_g, radius_g, length_g = get_cylinder_param(label[::-1, :, :])
        
        # get centers of the top and bottom faces
        upper_center_p = center_p + length_p * direction_p * 0.5 / 0.3
        bottom_center_p = center_p - length_p * direction_p * 0.5 / 0.3
        upper_center_g = center_g + length_g * direction_g * 0.5 / 0.3
        bottom_center_g = center_g - length_g * direction_g * 0.5 / 0.3
        assert upper_center_g[0] > bottom_center_g[0] or upper_center_p[0] > bottom_center_p[0], "wrong cylinder shape!"
        
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
        
        # record and save
        spacial_loss_list = [
            diameter, 
            length, 
            inplant_position, 
            endpoint_position, 
            angle, 
            implant_lateral1, 
            implant_lateral2, 
            endpoint_lateral1, 
            endpoint_lateral2, 
            implant_depth, 
            endpoint_depth
        ]
        loss_spacial.append(spacial_loss_list)

        length_width.append([
            radius_g * 2, 
            radius_p * 2,
            length_g,
            length_p
        ])
        print(f"predict {str(it+1).zfill(3)}: ", ' | '.join([str(float('{:.6f}'.format(i))) for i in spacial_loss_list]))

    loss_spacial = np.array(loss_spacial)
    length_width = np.array(length_width)
    mae = np.average(np.maximum(loss_spacial, -loss_spacial), axis = 0)
    print("MAE : ", " | ".join([str(float('{:.6f}'.format(i))) for i in mae]))

    np.savetxt("./loss_spacial.csv", loss_spacial, fmt = "%.6f", delimiter=',')
    np.savetxt("./length_width.csv", length_width, fmt = "%.6f", delimiter=',')
