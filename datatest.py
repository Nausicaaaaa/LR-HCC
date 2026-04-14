import nibabel as nib
import SimpleITK as sitk
import numpy as np
import glob
import re
import os

# 测试数据集完整性

# 测试能否用 nibabel 读取
def test_with_nibabel(file_path):
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        print(f"nibabel 读取成功: {file_path}")
        print(f"数据形状: {data.shape}")
        print(f"数据类型: {data.dtype}")
        return True
    except Exception as e:
        print(f"nibabel 读取失败: {e}")
        return False

def test_with_simpleitk(file_path):
    try:
        img = sitk.ReadImage(file_path)
        data = sitk.GetArrayFromImage(img)
        print(f"SimpleITK 读取成功: {file_path}")
        print(f"数据形状: {data.shape}")
        print(f"数据类型: {data.dtype}")
        return True
    except Exception as e:
        print(f"SimpleITK 读取失败: {e}")
        return False

def check_correspondence():
    images = glob.glob(r"nnUNet\nnUNet_raw\Dataset001_HCC\imagesTr\*.nii.gz")
    labels = glob.glob(r"nnUNet\nnUNet_raw\Dataset001_HCC\labelsTr\*.nii.gz")
    
    # 提取病例ID
    image_ids = set()
    for img in images:
        basename = os.path.basename(img)
        # 移除 _0000 后缀
        case_id = re.sub(r'_\d{4}\.nii\.gz$', '', basename)
        image_ids.add(case_id)
    
    label_ids = set(os.path.basename(l).replace('.nii.gz', '') for l in labels)
    
    print(f"图像病例数: {len(image_ids)}")
    print(f"标签病例数: {len(label_ids)}")
    
    # 检查缺失
    missing_labels = image_ids - label_ids
    if missing_labels:
        print(f"⚠️ 缺少标签的病例: {list(missing_labels)[:5]}")
    
    missing_images = label_ids - image_ids
    if missing_images:
        print(f"⚠️ 缺少图像的病例: {list(missing_images)[:5]}")

# check_correspondence()


def check_case_shape_consistency(image_files, labels_files):
    label_map = {
        os.path.basename(label_path).replace(".nii.gz", ""): label_path
        for label_path in labels_files
    }

    case_to_files = {}
    for image_path in image_files:
        image_name = os.path.basename(image_path)
        case_id = re.sub(r"_\d{4}\.nii\.gz$", "", image_name)
        case_to_files.setdefault(case_id, []).append(image_path)

    missing_labels = []
    read_failed = []
    inconsistent_cases = []
    label_mismatch_cases = []

    for case_id, case_files in case_to_files.items():
        label_path = label_map.get(case_id)
        if label_path is None:
            missing_labels.append(case_id)
            continue

        shape_items = []
        for image_path in sorted(case_files):
            try:
                image_shape = nib.load(image_path).shape
                shape_items.append((os.path.basename(image_path), image_shape))
            except Exception as e:
                read_failed.append((case_id, os.path.basename(image_path), str(e)))

        if not shape_items:
            continue

        unique_shapes = {item[1] for item in shape_items}
        if len(unique_shapes) > 1:
            inconsistent_cases.append((case_id, shape_items))

        try:
            label_shape = nib.load(label_path).shape
        except Exception as e:
            read_failed.append((case_id, os.path.basename(label_path), str(e)))
            continue

        if label_shape not in unique_shapes:
            label_mismatch_cases.append((case_id, shape_items, label_shape))

    print(f"Checked cases: {len(case_to_files)}")
    print(f"Cases missing label: {len(missing_labels)}")
    print(f"Read failed files: {len(read_failed)}")
    print(f"Cases with inconsistent image shapes: {len(inconsistent_cases)}")
    print(f"Cases with label shape mismatch: {len(label_mismatch_cases)}")

    if missing_labels:
        print(f"Missing label samples: {missing_labels[:5]}")

    if read_failed:
        print("Read failed samples:")
        for case_id, file_name, err in read_failed[:5]:
            print(f"  {case_id} - {file_name}: {err}")

    if inconsistent_cases:
        print("Inconsistent image shapes by case:")
        for case_id, shape_items in inconsistent_cases[:5]:
            print(f"  {case_id}")
            for file_name, shape in shape_items:
                print(f"    {file_name}: {shape}")

    if label_mismatch_cases:
        print("Label shape mismatch samples:")
        for case_id, shape_items, label_shape in label_mismatch_cases[:5]:
            print(f"  {case_id}: label={label_shape}")
            for file_name, shape in shape_items:
                print(f"    {file_name}: {shape}")

    return (
        len(missing_labels) == 0
        and len(read_failed) == 0
        and len(inconsistent_cases) == 0
        and len(label_mismatch_cases) == 0
    )

def check_zero_shape(image_files):
    zero_cases = []

    for f in image_files:
        img = nib.load(f)
        shape = img.shape

        if any(s == 0 for s in shape):
            zero_cases.append((os.path.basename(f), shape))

    print("Zero-dimension images:", len(zero_cases))
    for item in zero_cases[:5]:
        print(item)

    return len(zero_cases) == 0

def check_spacing(image_files):

    bad_spacing = []

    for f in image_files:
        img = nib.load(f)
        spacing = img.header.get_zooms()

        if any(s <= 0 for s in spacing):
            bad_spacing.append((os.path.basename(f), spacing))

    print("Bad spacing:", len(bad_spacing))

    for item in bad_spacing[:5]:
        print(item)

    return len(bad_spacing) == 0

def check_nan_inf(image_files):

    bad_cases = []

    for f in image_files:
        data = nib.load(f).get_fdata()

        if np.isnan(data).any() or np.isinf(data).any():
            bad_cases.append(os.path.basename(f))

    print("NaN/Inf images:", len(bad_cases))

    for item in bad_cases[:5]:
        print(item)

    return len(bad_cases) == 0 

def check_empty_label(labels_files):
    empty = []
    for f in labels_files:
        data = nib.load(f).get_fdata()

        if np.max(data) == 0:
            empty.append(os.path.basename(f))

    print("Empty labels:", len(empty))

    for item in empty[:5]:
        print(item)

    return len(empty) == 0

def check_volume_size(image_files):

    huge = []

    for f in image_files:
        shape = nib.load(f).shape

        voxels = shape[0]*shape[1]*shape[2]

        if voxels > 512*512*1000:
            huge.append((os.path.basename(f), shape))

    print("Huge volumes:", len(huge))

    for item in huge[:5]:
        print(item)

    return len(huge) == 0

def check_affine(image_files):

    bad_affine = []

    for f in image_files:
        img = nib.load(f)

        if np.isnan(img.affine).any():
            bad_affine.append(os.path.basename(f))

    print("Bad affine:", len(bad_affine))

    for item in bad_affine[:5]:
        print(item)

    return len(bad_affine) == 0


# 测试所有文件
image_files = glob.glob(r"nnUNet\nnUNet_raw\Dataset001_HCC\imagesTr\*.nii.gz")
labels_files = glob.glob(r"nnUNet\nnUNet_raw\Dataset001_HCC\labelsTr\*.nii.gz")

check_case_shape_consistency(image_files,labels_files) # 检查形状是否一致
check_zero_shape(image_files) # 检查是否存在 shape 中有 0 的图像（无效图像）
check_spacing(image_files) # 检查 spacing 是否异常
check_nan_inf(image_files) # 检查是否存在 NaN 或 Inf
check_empty_label(labels_files) # 检查 label 是否只有 0（空标注）
check_volume_size(image_files) # 检查是否存在超大体积
check_affine(image_files) # 检查 affine 是否异常

