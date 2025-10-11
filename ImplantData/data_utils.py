import SimpleITK as sitk
import numpy as np
from scipy import ndimage
from stl import mesh
import cv2
import copy


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
    cylinder_mesh = mesh.Mesh.from_file(stl_path)
        
    all_v = cylinder_mesh.points.reshape((-1, 3))
    uni_v, c_v = np.unique(all_v, axis=0, return_counts=True)
    # print("data name", data_name)
    # print("STL feat.", all_v.shape, "unique vert.", uni_v.shape, "max rep.", np.max(c_v), "min rep.", np.min(c_v))
    
    top_2_indices = np.argsort(c_v)[-2:]
    circle_centers = uni_v[top_2_indices]
    other_vertices = np.delete(uni_v, top_2_indices, axis=0)
    
    centroid = np.average(circle_centers, axis = 0)
    if circle_centers[0][2] > circle_centers[1][2] :
        upper_center = circle_centers[0]
        lower_center = circle_centers[1]
    else :
        upper_center = circle_centers[1]
        lower_center = circle_centers[0]
    
    dist_to_center1 = np.linalg.norm(other_vertices - upper_center, axis=1)
    dist_to_center2 = np.linalg.norm(other_vertices - lower_center, axis=1)
    
    closer_to_center1 = dist_to_center1 < dist_to_center2
    group1_indices = np.where(closer_to_center1)[0]
    group2_indices = np.where(~closer_to_center1)[0]
    
    upper_vertices = other_vertices[group1_indices]
    upper_distance = np.linalg.norm(upper_vertices - upper_center, axis=1)
    lower_vertices = other_vertices[group2_indices]
    lower_distance = np.linalg.norm(lower_vertices - lower_center, axis=1)
    
    radius = np.average(np.stack((upper_distance, lower_distance), axis=0))
    length = np.linalg.norm(upper_center - lower_center, axis=0)
    direction = (upper_center - lower_center) / length

    return upper_center, lower_center, centroid, radius, length, direction


def cylinder_transform(cylinder_cfg, world_size, spacing) :
    def transform_to_world(point, world_size, spacing) :
        return np.array([
            world_size[0] * 0.5 + point[2] / spacing,
            world_size[1] * 0.5 + point[1] / spacing,
            world_size[2] * 0.5 + point[0] / spacing
        ])
    upper_center, lower_center, centroid, radius, length, direction = cylinder_cfg
    upper_center_tr = transform_to_world(upper_center, world_size, spacing)
    lower_center_tr = transform_to_world(lower_center, world_size, spacing)
    centroid_tr = transform_to_world(centroid, world_size, spacing)
    radius_tr = radius / spacing
    length_tr = np.linalg.norm(upper_center_tr - lower_center_tr, axis=0)
    direction_tr = (upper_center_tr - lower_center_tr) / length_tr
    return upper_center_tr, lower_center_tr, centroid_tr, radius_tr, length_tr, direction_tr


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


def get_cylinder_param(cylinder_voxel) :
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
    radius = np.percentile(distances, 90) - 0.25
    
    # get length
    projection = np.dot(cylinder - center, direction)
    distances = np.abs(projection)
    length = np.percentile(distances, 95) * 2 + 1
    
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
    # cs_y = int(np.mean(coordinates))
    cs_y = int(average_y)
    # get roi slice base on corner detection
    x_coords = np.where(mip_img == 255)[1]
    # cs_x = int(np.mean(coordinates_x))
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