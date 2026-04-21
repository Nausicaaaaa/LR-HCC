import nibabel as nib
import numpy as np
import os

'''
检查目录中的 .nii.gz 文件是否损坏
检查标签是否只有0和1
'''
def check_files(directory, file_type_name):
    """检查目录中的 .nii.gz 文件是否损坏"""
    files = sorted([f for f in os.listdir(directory) if f.endswith('.nii.gz')])
    print(f'总共有 {len(files)} 个 {file_type_name} 文件')
    print()
    
    corrupted = []
    for f in files:
        filepath = os.path.join(directory, f)
        try:
            img = nib.load(filepath)
            _ = img.get_fdata()
        except EOFError as e:
            print(f'警告: 文件损坏 - {f}: {e}')
            corrupted.append(f)
        except Exception as e:
            print(f'警告: 读取失败 - {f}: {e}')
            corrupted.append(f)
    
    return files, corrupted

# 检查 labelsTr
labels_dir = '/mnt/data/KASR/Dengsiyi/LR-HCC/data/nnUNet_raw/Dataset001_HCC/labelsTr'
print('=' * 50)
print('检查 labelsTr 目录')
print('=' * 50)
label_files, corrupted_labels = check_files(labels_dir, 'label')

# 统计 label 唯一值（跳过损坏的文件）
all_unique_values = set()
for f in label_files:
    if f in corrupted_labels:
        continue
    filepath = os.path.join(labels_dir, f)
    try:
        img = nib.load(filepath)
        data = img.get_fdata()
        unique_vals = np.unique(data).astype(int)
        all_unique_values.update(unique_vals)
    except Exception as e:
        print(f'警告: 处理失败 - {f}: {e}')

print()
print(f'所有 label 文件的唯一值: {sorted(all_unique_values)}')
print(f'是否只有 0 和 1: {sorted(all_unique_values) == [0, 1]}')

if corrupted_labels:
    print()
    print(f'损坏的 label 文件数量: {len(corrupted_labels)}')
    print('损坏的 label 文件列表:')
    for cf in corrupted_labels:
        print(f'  - {cf}')

# 检查 imagesTr
print()
print('=' * 50)
print('检查 imagesTr 目录')
print('=' * 50)
images_dir = '/mnt/data/KASR/Dengsiyi/LR-HCC/data/nnUNet_raw/Dataset001_HCC/imagesTr'
image_files, corrupted_images = check_files(images_dir, 'image')

if corrupted_images:
    print()
    print(f'损坏的 image 文件数量: {len(corrupted_images)}')
    print('损坏的 image 文件列表:')
    for cf in corrupted_images:
        print(f'  - {cf}')

# 总结
print()
print('=' * 50)
print('检查完成')
print('=' * 50)
total_corrupted = len(corrupted_labels) + len(corrupted_images)
if total_corrupted == 0:
    print('未发现损坏的文件')
else:
    print(f'总共发现 {total_corrupted} 个损坏的文件')
    print(f'  - labelsTr: {len(corrupted_labels)} 个')
    print(f'  - imagesTr: {len(corrupted_images)} 个')
