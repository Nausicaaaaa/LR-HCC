# 开发可解释AI模型识别高风险背景下肝脏局灶性病变的良恶性并精准诊断HCC

## 数据说明

|           | HCC | 恶性非HCC | 良性 | ALL | 备注                     |
| --------- | --- | --------- | ---- | --- | ------------------------ |
| 附二      | 417 | 42        | 51   | 510 | 存在数据缺失7例          |
| 西南      | 575 | 102       | 34   | 711 | 存在数据缺失2例          |
| all       | 991 | 143       | 85   |     |                          |
| 4模态数据 | 991 | 128       | 66   |     | 去除了模态匹配不佳的数据 |

## 数据处理

1.`preprocess.py`进行预处理
- 文件处理：
  - 遍历地址 把四期CT（A、D、P、S or PS）对应为四种模态，命名为case_xxx_xxxx
  - 处理后的图像、标签分别保存到 `nnUNet_raw\Dataset001_HCC\imagesTr`及`labelsTr`
  - `case_name_mapping.txt`原始ID对应的 case_xxx 和类别
  - `case_skip_data.txt` 为存在缺失数据跳过的ID
- 图像处理：
  - 4 个模态的图像对齐到 A 模态的空间网格，统一shape / spacing / orientation / affine ，保证物理尺度一致
  - <MARK>注意：4个模态因为是不同时期扫的 因此不是天然的完全配准
- 标签处理：
  - 把HCC、恶性非HCC、良性三类的标签映射为不同的值（HCC=1，onHCC=2、laingxing=3）命名为case_xxx
  - <MARK>为确保标签的有效性 取4个标签的并集
  - 去除并集为0或小于500的的case 并返回case列表
  
  > 取并集的问题：有些地方不是ROI而被划分进去 导致干扰
  > 取交集的问题：更谨慎 但有些ROI会被去除掉 只能去除一些交集后ROI体素过小的case（代表模态对齐效果差）

> `Dataset001_HCC`存放4个模态的数据
> `Dataset002_HCC`仅AP模态数据


## 可视化
`visualization_original.py` 原始图像&标签可视化 
`visualisation.py` 处理后的case 图像&标签可视化 **检查是否正确匹配**


## 数据上传服务器
**ssh+rsync**
``` bash
rsync -avP --partial --append-verify --ignore-existing \
-e "ssh -p 21133 -T -c aes128-gcm@openssh.com -o Compression=no" \
/mnt/c/Users/75267/Documents/python项目/LR-HCC/nnUNet/nnUNet_raw/Dataset003_HCC/labelsTr/ \
KASR@106.120.24.118:/mnt/data/KASR/Dengsiyi/LR-HCC/data/nnUNet_raw/Dataset003_HCC/labelsTr/
```

key: Ky#202910cn

## 训练

main.py


watch -n 0.5 nvidia-smi  