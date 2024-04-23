# ImplantDataProcess

基于nnU-Net图像分割的口腔下颌后牙种植方案规划课题 - 数据处理代码

## 代码简述

1. get_dicom 读取dicom文件
2. get_pcd 读取pcd文件
3. get_stl 读取stl文件
4. get_cylinder_param 获取圆柱体参数
5. dataset_split 用于手动划分数据集
6. train_to_nii 将训练集原始图像转化为nii文件
7. val_to_nii 将验证集原始图像转化为nii文件
8. test_to_nii 将测试集原始图像转化为nii文件
9. loss_segment 评价分割性能
10. loss_spacial 评价空间形态
11. fit_cylinder 拟合标准圆柱体
12. nii_to_image 将nii转为二维图像
13. rebuild_nii 将输出重建为nii图像
14. rebuild_dicom 将输出重建为dicom图像
