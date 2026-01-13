import SimpleITK as sitk
import numpy as np
from scipy import ndimage
import trimesh
import stl
import cv2
import copy
import csv
import random
import torch


def get_dataset_metadata(data_path, random_seed=21):
    with open(f'{data_path}/metadata.csv', mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        metadata = {
            row['ID'].zfill(3): {
                'spacing': float(row['Spacing']),
                'window': float(row['Window']),
                'width': float(row['Width']),
                'mip_window': float(row['MIPWindow']),
                'mip_width': float(row['MIPWidth']),
                'fdi': float(row['FDI']),
            }
            for row in rows
        }
        data_id = list(metadata.keys())

        if 'isTest' in reader.fieldnames:
            train_split = [row['ID'].zfill(3) for row in rows if row['isTest'] == "0"]
            val_split = [row['ID'].zfill(3) for row in rows if row['isTest'] == "1"]
        else:
            random.seed(random_seed)
            random.shuffle(data_id)
            train_split = data_id[:round(0.8*len(data_id))]
            val_split = data_id[round(0.8*len(data_id)):]

        return metadata, train_split, val_split


def get_dcm_3d_array(dicom_dir) :
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    dcm_3d_array = sitk.GetArrayFromImage(image)
    return dcm_3d_array


def window_transform_2d(dcm_2d_array, window_width, window_center, normal=False):
    min_window = float(window_center) - 0.5 * float(window_width)
    new_2d_array = (dcm_2d_array - min_window) / float(window_width)
    new_2d_array[new_2d_array < 0] = 0
    new_2d_array[new_2d_array > 1] = 1
    if not normal:
        new_2d_array = (new_2d_array * 255).astype('uint8')
    return new_2d_array


def window_transform_3d(dcm_3d_array, window_width, window_center, low_slice_num=0, high_slice_num=0, normal=False) :
    if(high_slice_num == 0):
        high_slice_num = len(dcm_3d_array)
    new_dcm_3d_array = np.zeros_like(dcm_3d_array)
    
    for slice_num in range(low_slice_num, high_slice_num):
        new_dcm_3d_array[slice_num] = window_transform_2d(dcm_3d_array[slice_num], window_width, window_center, normal)

    return new_dcm_3d_array


def get_stl(stl_path):
    mesh = trimesh.load(stl_path)
    components = mesh.split()

    cylinder_mesh = stl.mesh.Mesh.from_file(stl_path)
    all_v = cylinder_mesh.points.reshape((-1, 3))
    uni_v, c_v = np.unique(all_v, axis=0, return_counts=True)

    center_indices = np.argsort(c_v)[-2*len(components):]
    circle_centers = uni_v[center_indices]
    other_vertices = np.delete(uni_v, center_indices, axis=0)

    # group all circle
    distances = np.linalg.norm(other_vertices[:, np.newaxis] - circle_centers[np.newaxis, :], axis=2)
    nearest_center_idx = np.argmin(distances, axis=1)

    circles = []
    for i in range(circle_centers.shape[0]):
        center = circle_centers[i]
        vertices = other_vertices[nearest_center_idx == i]
        cov_matrix = np.cov((vertices - center).T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        axis_idx = np.argmin(eigenvalues)
        axis = eigenvectors[:, axis_idx]
        circles.append({
            "center": center,
            "vertices": vertices,
            "axis": axis
        })

    # pair circles into cylinder
    for ci in range(len(circles)) :
        distance = []
        for cj in range(len(circles)) :
            if ci == cj :
                distance.append(np.inf)
            else :
                c = circles[ci]["center"]
                a = circles[cj]["axis"] / np.linalg.norm(circles[cj]["axis"])
                distance.append(np.linalg.norm(np.cross(c - circles[cj]["center"], a)))
        circles[ci]["pair_idx"] = np.argmin(np.array(distance))

    # easy check valid pairing
    center_groups = []
    added_groups = []
    for ci in range(len(circles)) :
        cj = circles[ci]["pair_idx"]
        assert circles[cj]["pair_idx"] == ci, f"circle {ci} and {cj} not paired correctly"
        if ci not in added_groups and cj not in added_groups :
            center_groups.append({
                "c0": circles[ci]["center"],
                "e0": circles[ci]["vertices"],
                "c1": circles[cj]["center"],
                "e1": circles[cj]["vertices"],
            })
            added_groups.append(ci)
            added_groups.append(cj)
    
    cylinders = []
    for cg in center_groups:
        centroid = (cg["c0"] + cg["c1"]) / 2
        if cg["c0"][2] > cg["c1"][2] :
            upper_center = cg["c0"]
            lower_center = cg["c1"]
            upper_distance = np.linalg.norm(cg["e0"] - upper_center, axis=1)
            lower_distance = np.linalg.norm(cg["e1"] - lower_center, axis=1)
        else :
            upper_center = cg["c1"]
            lower_center = cg["c0"]
            upper_distance = np.linalg.norm(cg["e1"] - upper_center, axis=1)
            lower_distance = np.linalg.norm(cg["e0"] - lower_center, axis=1)
        
        radius = np.average(np.stack((upper_distance, lower_distance), axis=0))
        length = np.linalg.norm(upper_center - lower_center, axis=0)
        direction = (upper_center - lower_center) / length

        cylinders.append([
            upper_center, lower_center, centroid, radius, length, direction
        ])

    return cylinders


def cylinder_transform(cylinders, world_size, spacing) :
    def transform_to_world(point, world_size, spacing) :
        return np.array([
            world_size[0] * 0.5 + point[2] / spacing,
            world_size[1] * 0.5 + point[1] / spacing,
            world_size[2] * 0.5 + point[0] / spacing
        ])
    cylinders_tr = []
    for cylinder_cfg in cylinders :
        upper_center, lower_center, centroid, radius, length, direction = cylinder_cfg
        upper_center_tr = transform_to_world(upper_center, world_size, spacing)
        lower_center_tr = transform_to_world(lower_center, world_size, spacing)
        centroid_tr = transform_to_world(centroid, world_size, spacing)
        radius_tr = radius / spacing
        length_tr = np.linalg.norm(upper_center_tr - lower_center_tr, axis=0)
        direction_tr = (upper_center_tr - lower_center_tr) / length_tr
        cylinders_tr.append([
            upper_center_tr, lower_center_tr, centroid_tr, radius_tr, length_tr, direction_tr
        ])
    return cylinders_tr


def cylinder_render(center, image_size, direction, length, radius):
    if isinstance(image_size, int):
        image_size = np.array([image_size, image_size, image_size])
    xx, yy, zz = np.mgrid[0:image_size[0], 0:image_size[1], 0:image_size[2]]
    v = np.stack([xx, yy, zz], axis=-1) - center
    projection = np.dot(v, direction)  # v·d
    perpendicular_dist_sq = np.sum(v**2, axis=-1) - projection**2
    length_cond = np.abs(projection) <= length / 2
    implant_part = (perpendicular_dist_sq <= radius**2) & length_cond
    implant_part = implant_part.astype(np.uint8)
    return implant_part


def get_cylinder_param(cylinder_voxel, approx=False) :
    # get cylinder point locations
    cylinder = np.array(np.where(cylinder_voxel == 1), dtype=float).T
    
    # fit the main axis
    output = cv2.fitLine(cylinder, distType = cv2.DIST_L2, param = 0, reps = 1e-2, aeps = 1e-2)
    center = np.array(output[3:]).T[0]
    direction = np.array(output[:3]).T[0]
    
    # get radius
    direction = direction / np.linalg.norm(direction)
    relative_vectors = cylinder - center
    cross_products = np.cross(relative_vectors, direction)
    distances = np.linalg.norm(cross_products, axis=1)
    radius = np.percentile(distances, 90) - 0.25 if approx else np.max(distances)
    
    # get length
    projection = np.dot(cylinder - center, direction)
    distances = np.abs(projection)
    length = np.percentile(distances, 95) * 2 + 1 if approx else np.max(distances) * 2
    
    return center, direction, radius*0.3, length*0.3


def keep_largest_component(mask):
    labeled, num_features = ndimage.label(mask)
    if num_features <= 1:
        return mask
    
    sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    largest_label = np.argmax(sizes) + 1
    largest_mask = (labeled == largest_label)

    return largest_mask.astype(mask.dtype)


def get_cross_section(dicom, if_vis, number, results_path, window, width) :
    dcm_3d_array = window_transform_3d(dicom, window_width=width, window_center=window).astype(np.uint8)

    # Maximum Intensity Projection
    mip_img = np.max(dcm_3d_array, axis=2)
    mip_img[mip_img != 255] = 0

    # open and close
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    mip_img = cv2.morphologyEx(mip_img, cv2.MORPH_OPEN, kernel, iterations=1)
    mip_img = cv2.morphologyEx(mip_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # get roi slice base on corner detection
    y_coords = np.where(mip_img == 255)[0]
    y_coords = y_coords[(y_coords > 50) & (y_coords < 300)]
    average_y = np.mean(y_coords)

    bi_img, coordinates = harris_corner_detection(mip_img, gamma=0.25)
    index = np.argsort(coordinates[:, 1])
    coordinates = coordinates[index]
    del_idx = np.array([], dtype=np.int16)
    for i in range(coordinates.shape[0]):
        x = int(coordinates[i, 0])
        y = int(coordinates[i, 1])
        if dicom.shape[0] < 210:
            break
        if np.count_nonzero(mip_img[x-1, (y-1):(y+2)]) >= 2 or x >= 250 or x < 50:
            del_idx = np.append(del_idx, int(i))
    
    coordinates = np.delete(coordinates, del_idx, axis=0)
    coordinates = coordinates.astype(int)
    ori_coordinates = coordinates
    
    # find anomalies
    coordinates_x = coordinates[:, 1]
    coordinates_y = coordinates[:, 0]
    data_std = np.std(coordinates_y)
    data_mean = np.mean(coordinates_y)
    anomaly = data_std * 3

    coordinates = []
    for num in coordinates_y:
        if num <= data_mean + anomaly and num >= data_mean - anomaly :
            coordinates.append(num)

    cs_upper, cs_lower = np.min(coordinates), np.max(coordinates)
    cs_front, cs_rear = np.min(coordinates_x), np.max(coordinates_x)
    cs_y = int(average_y)
    # get roi slice base on corner detection
    x_coords = np.where(mip_img == 255)[1]
    cs_front = int(np.min(x_coords))
    
    # 96 * 2
    cs_front -= 16
    if cs_front < 0 :
        cs_front = 0
    cs_rear = cs_front + 192
    
    # 96 * 2
    cs_upper = cs_y - 96
    cs_lower = cs_y + 96
    if cs_upper < 0 :
        cs_upper = 0
        cs_lower = 192

    if if_vis :
        mip_img = cv2.cvtColor(mip_img, cv2.COLOR_GRAY2BGR)
        cv2.line(mip_img, (0, cs_upper), (mip_img.shape[1], cs_upper), (255, 0, 0), 2)
        cv2.line(mip_img, (0, cs_lower), (mip_img.shape[1], cs_lower), (255, 0, 0), 2)
        cv2.line(mip_img, (cs_rear, 0), (cs_rear, mip_img.shape[0]), (255, 0, 0), 2)
        cv2.line(mip_img, (cs_front, 0), (cs_front, mip_img.shape[0]), (255, 0, 0), 2)
        for coor in ori_coordinates:
            cv2.circle(mip_img, coor[::-1], radius=3, color=(0, 0, 255), thickness=-1)
        cv2.imwrite(f"{results_path}/mip/{str(number).zfill(4)}_mip_sagittal.png", mip_img)
    
    dcm_3d_array = dcm_3d_array[cs_upper:cs_lower, cs_front:cs_rear, :]
    
    # Maximum Intensity Projection
    mip_img = np.max(dcm_3d_array, axis=1)
    mip_img[mip_img != 255] = 0
    
    teeth_area = np.max(mip_img, axis=0)
    non_zero_area = np.where(teeth_area != 0)[0]
    teeth_min, teeth_max = non_zero_area.min(), non_zero_area.max()
    
    cs_left = (teeth_min + teeth_max) // 2 - 128
    cs_right = (teeth_min + teeth_max) // 2 + 128
    
    if if_vis :
        mip_img = cv2.cvtColor(mip_img, cv2.COLOR_GRAY2BGR)
        cv2.line(mip_img, (cs_left, 0), (cs_left, mip_img.shape[0]), (255, 0, 0), 2)
        cv2.line(mip_img, (cs_right, 0), (cs_right, mip_img.shape[0]), (255, 0, 0), 2)
        cv2.imwrite(f"{results_path}/mip/{str(number).zfill(4)}_mip_coronal.png", mip_img)
        
    return cs_upper, cs_lower, cs_front, cs_rear, int(cs_left), int(cs_right)


def harris_corner_detection(img, gamma):
    img = img.astype("uint8")
    img = np.float32(img)
    width, height = img.shape
    
    Harris_detector = cv2.cornerHarris(img, 2, 3, 0.04) 

    dst = Harris_detector
    thres = gamma * dst.max() 
    gray_img = copy.deepcopy(img)
    gray_img[dst <= thres] = 0
    gray_img[dst > thres] = 255
    gray_img = gray_img.astype("uint8")

    coor = np.array([])
    for i in range(width):
        for j in range(height):
            if gray_img[i][j] == 255: 
                coor = np.append(coor, [i, j], axis=0) 
    
    coor = np.reshape(coor, (-1, 2)) 
    return gray_img, coor


def differentiable_cylinder(cylinder_params, patch_size=96, sharpness=200) :
    half_patch = patch_size / 2
    B = cylinder_params.shape[0]
    device = cylinder_params.device

    # Unpack and decode parameters
    radius = cylinder_params[:, 0] * 1.5 + 5.5           # (B,)
    length = cylinder_params[:, 1] * 10.0 + 35.0         # (B,)
    direction = cylinder_params[:, 2:5]                  # (B, 3)
    centroid = cylinder_params[:, 5:] * half_patch       # (B, 3)

    # Normalize direction
    direction = direction / (direction.norm(dim=1, keepdim=True) + 1e-8)  # (B, 3)

    # Create voxel grid (D, H, W, 3)
    coords = torch.linspace(0, patch_size - 1, patch_size, device=device)
    zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing='ij')  # (D, H, W)
    zz = zz.unsqueeze(0).expand(B, -1, -1, -1) - centroid[:, 0].view(B, 1, 1, 1)  # B x D x H x W
    yy = yy.unsqueeze(0).expand(B, -1, -1, -1) - centroid[:, 1].view(B, 1, 1, 1)
    xx = xx.unsqueeze(0).expand(B, -1, -1, -1) - centroid[:, 2].view(B, 1, 1, 1)
    grid = torch.stack([xx, yy, zz], dim=-1)  # (B, D, H, W, 3)
    v = grid - half_patch  # shift origin to center

    # Project v onto direction: scalar projection
    projection = (v * direction.view(B, 1, 1, 1, 3)).sum(dim=-1)  # (B, D, H, W)

    # Perpendicular distance squared
    v_squared = (v ** 2).sum(dim=-1)  # (B, D, H, W)
    proj_squared = projection ** 2   # (B, D, H, W)
    perp_dist_sq = v_squared - proj_squared  # (B, D, H, W)

    # Length and radius soft masks
    radius_sq = radius.view(B, 1, 1, 1) ** 2
    length_half = length.view(B, 1, 1, 1) / 2.0

    soft_radial_mask = torch.sigmoid((radius_sq - perp_dist_sq) * sharpness)
    soft_axial_mask = torch.sigmoid((length_half - torch.abs(projection)) * sharpness)

    # Final mask
    mask = soft_radial_mask * soft_axial_mask  # (B, D, H, W)

    return mask.unsqueeze(1)
