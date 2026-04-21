import os
import sys
import argparse
import subprocess
from pathlib import Path

# 添加当前目录到 Python 路径，以便找到自定义的 nnunetv2 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def setup_nnunet_environment(dataset_id, raw_data_base, preprocessed_base, results_base):
    """
    设置nnUNet环境变量
    
    Args:
        dataset_id: 数据集ID (例如: 001, 002)
        raw_data_base: nnUNet_raw的根目录
        preprocessed_base: nnUNet_preprocessed的根目录
        results_base: nnUNet_results的根目录
    """
    os.environ['nnUNet_raw'] = raw_data_base
    os.environ['nnUNet_preprocessed'] = preprocessed_base
    os.environ['nnUNet_results'] = results_base
    
    
    print(f"环境变量设置:")
    print(f"  nnUNet_raw = {raw_data_base}")
    print(f"  nnUNet_preprocessed = {preprocessed_base}")
    print(f"  nnUNet_results = {results_base}")
    
    # 确保目录存在
    Path(raw_data_base).mkdir(parents=True, exist_ok=True)
    Path(preprocessed_base).mkdir(parents=True, exist_ok=True)
    Path(results_base).mkdir(parents=True, exist_ok=True)

def run_command(command, description, env=None):
    """
    运行系统命令并打印输出
    """
    print(f"\n{'='*60}")
    print(f"执行: {description}")
    print(f"命令: {' '.join(command)}")
    print(f"{'='*60}\n")
    
    # 合并环境变量
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    
    try:
        # 使用实时输出
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                  universal_newlines=True, bufsize=1, env=run_env)
        
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode == 0:
            print("\n✅ 执行成功!")
        else:
            print(f"\n❌ 执行失败! 返回码: {process.returncode}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ 执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def plan_and_preprocess(dataset_id, num_processes=8, verify=True):
    print("\n🔍 步骤1: 验证数据集完整性 + 预处理")
    command = [
        'nnUNetv2_plan_and_preprocess',
        '-d', dataset_id,
        '-np', str(num_processes)
    ] 

    if verify:
        command.append('--verify_dataset_integrity')

    run_command(command, "执行数据验证 + 预处理")

def train_model(dataset_id, configuration='2d', fold=0, trainer='nnUNetTrainer', gpu_ids=None, num_gpus=None):
    """
    步骤3: 训练模型
    
    Args:
        dataset_id: 数据集ID
        configuration: 网络配置 ('2d', '3d_fullres', '3d_lowres')
        fold: 指定用于验证的折数 (0-4),其余4折用于训练
        trainer: 训练器名称
    
    参数解释:
        -d: 数据集ID
        -c: 网络配置 (2D, 3D_fullres, 3D_lowres)
        -f: 指定用于验证的折数 (0,1,2,3,4),其余折用于训练
        -tr: 使用的训练器
        -p: 规划器名称 (可选)
        --npz: 保存npz文件用于后期评估

    """
    print(f"\n🏋️ 开始训练模型 (配置={configuration}, 验证集=fold {fold}, 训练集=其他{4}折)")
    # 构建命令
    command = [
        'nnUNetv2_train',
        dataset_id,
        configuration,
        str(fold),
        '-tr', trainer
    ]
    
    # 设置环境变量
    env_vars = {}
    
    if gpu_ids:
        # 设置CUDA可见设备
        env_vars['CUDA_VISIBLE_DEVICES'] = gpu_ids
        print(f"设置使用GPU: {gpu_ids}")
        
        # 计算GPU数量
        gpu_count = len(gpu_ids.split(','))
        
        # 方法1: 使用CUDA_VISIBLE_DEVICES + nnUNet自动检测
        # 这是最简单的方式，让nnUNet自动使用所有可见GPU
        print(f"可见GPU数量: {gpu_count}")
        
        # 可选：如果仍然遇到问题，可以尝试限制使用的GPU数量
        if num_gpus is not None:
            command.extend(['-num_gpus', str(num_gpus)])
    else:
        print("使用所有可用GPU")
    
    run_command(command, f"训练模型 - 验证集为 fold {fold}", env=env_vars)


def train_all_folds(dataset_id, configuration='2d', trainer='nnUNetTrainer', gpu_ids=None):
    """
    训练所有5折交叉验证
    """
    print("\n🔄 训练所有5折交叉验证")
    for fold in range(5):
        train_model(dataset_id, configuration, fold, trainer, gpu_ids)


def find_best_configuration(dataset_id, configuration='2d'):
    """
    步骤3: 找出最佳配置
    
    Args:
        dataset_id: 数据集ID
        configuration: 网络配置
    
    参数解释:
        -d: 数据集ID
        -c: 配置
    """
    print("\n🔬 步骤3: 找出最佳配置")
    command = [
        'nnUNetv2_find_best_configuration',
        dataset_id,
        '-c', configuration
    ]
    run_command(command, "自动选择最佳配置")

def predict_test_set(dataset_id, configuration='2d', trainer='nnUNetTrainer', fold=0, gpu_ids=None):
    """
    步骤4: 预测测试集
    
    Args:
        dataset_id: 数据集ID
        configuration: 使用的配置
        trainer: 训练器名称
        fold: 使用哪一折的模型
    
    参数解释:
        -i: 输入文件夹 (imagesTs)
        -o: 输出文件夹
        -d: 数据集ID
        -c: 配置
        -tr: 训练器
        -f: 使用哪一折的模型
        -chk: 检查点名称 (checkpoint_final.pth 或 checkpoint_best.pth)
        --save_probabilities: 保存概率图
    """
    print(f"\n🔮 步骤4: 预测测试集")
    
    # 设置路径
    input_folder = os.path.join(os.environ['nnUNet_raw'], f'Dataset{dataset_id}', 'imagesTs')
    output_folder = os.path.join(os.environ['nnUNet_results'], f'Dataset{dataset_id}', f'predicted_{configuration}')
    
    # 确保输出目录存在
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    command = [
        'nnUNetv2_predict',
        '-i', input_folder,
        '-o', output_folder,
        '-d', dataset_id,
        '-c', configuration,
        '-tr', trainer,
        '-f', str(fold),
        '-chk', 'checkpoint_final.pth',
        '--save_probabilities'
    ]
    # 设置环境变量
    env_vars = {}
    if gpu_ids:
        env_vars['CUDA_VISIBLE_DEVICES'] = gpu_ids
        print(f"预测使用GPU: {gpu_ids}")
    
    run_command(command, f"预测测试集 - 使用模型 fold {fold}", env=env_vars)

def ensemble_predictions(dataset_id, configurations=['2d', '3d_fullres'], folds=[0,1,2,3,4]):
    """
    步骤5: 集成多个模型的预测结果
    
    Args:
        dataset_id: 数据集ID
        configurations: 要集成的配置列表
        folds: 要集成的折数
    
    参数解释:
        -i: 输入文件夹列表
        -o: 输出文件夹
        --ensemble_method: 集成方法 (softmax 或 majority_vote)
    """
    print(f"\n🤝 步骤5: 集成预测结果")
    
    output_folder = os.path.join(os.environ['nnUNet_results'], f'Dataset{dataset_id}', 'ensemble')
    
    command = [
        'nnUNetv2_ensemble',
        '-i'
    ]
    
    # 添加所有要集成的预测结果文件夹
    for config in configurations:
        pred_folder = os.path.join(os.environ['nnUNet_results'], f'Dataset{dataset_id}', f'predicted_{config}')
        command.append(pred_folder)
    
    command.extend(['-o', output_folder])
    
    run_command(command, "集成多个模型的预测结果")

def evaluate_predictions(dataset_id, predictions_folder):
    """
    步骤6: 评估预测结果
    
    Args:
        dataset_id: 数据集ID
        predictions_folder: 预测结果文件夹
    """
    print(f"\n📊 步骤6: 评估预测结果")
    
    # 获取真实标签文件夹
    labels_folder = os.path.join(os.environ['nnUNet_raw'], f'Dataset{dataset_id}', 'labelsTs')
    
    command = [
        'nnUNetv2_evaluate_folder',
        labels_folder,
        predictions_folder,
        '-djfile', os.path.join(os.environ['nnUNet_results'], f'Dataset{dataset_id}', 'dataset.json'),
        '-plfile', os.path.join(os.environ['nnUNet_results'], f'Dataset{dataset_id}', 'plans.json')
    ]
    
    run_command(command, "计算评估指标 (Dice, Hausdorff等)")


def check_gpu_availability(gpu_ids):
    """
    检查指定的GPU是否可用
    """
    import torch
    
    if gpu_ids is None:
        return True, torch.cuda.device_count()
    
    try:
        # 设置环境变量
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        gpu_list = [int(x.strip()) for x in gpu_ids.split(',')]
        
        # 检查每个GPU
        available_gpus = []
        for gpu_id in gpu_list:
            try:
                torch.cuda.get_device_properties(gpu_id)
                available_gpus.append(gpu_id)
            except RuntimeError:
                print(f"警告: GPU {gpu_id} 不可用")
        
        if len(available_gpus) == 0:
            return False, 0
        
        return True, len(available_gpus)
    except Exception as e:
        print(f"检查GPU时出错: {e}")
        return False, 0
    

def main():

    parser = argparse.ArgumentParser(description='nnUNet训练&测试')
    
    # 必需参数
    parser.add_argument('--dataset_id', type=str, required=True,
                       help='数据集ID,只需三位数字例如: 001, 002')
    parser.add_argument('--raw_data_base', type=str, default='/mnt/data/KASR/Dengsiyi/LR-HCC/data/nnUNet_raw',
                       help='nnUNet_raw 路径')
    
    # 可选参数
    parser.add_argument('--preprocessed_base', type=str, default='/mnt/data/KASR/Dengsiyi/LR-HCC/data/nnUNet_preprocessed',
                       help='nnUNet_preprocessed 路经')
    
    parser.add_argument('--results_base', type=str, default='/mnt/data/KASR/Dengsiyi/LR-HCC/data/nnUNet_results',
                       help='nnUNet_results的根目录')
    
    parser.add_argument('--configuration', type=str, default='2d',
                       choices=['2d', '3d_fullres', '3d_lowres'],
                       help='网络配置: 2d, 3d_fullres, 3d_lowres (默认: 2d)')
    
    parser.add_argument('--trainer', type=str, default='nnUNetTrainer_smallROI',
                       help='训练器名称 (默认: nnUNetTrainer_smallROI)')
    
    parser.add_argument('--num_processes', type=int, default=8,
                       help='预处理并行进程数 (默认: 8)')
    

    parser.add_argument('--gpu_ids', type=str, default=None,
                       help='指定使用的GPU编号,例如: "0,1" (默认使用所有GPU)')
    
    parser.add_argument('--num_gpus', type=int, default=None,
                       help='指定使用的GPU数量 (默认: 自动检测)')

    parser.add_argument('--single_gpu', action='store_true',
                       help='强制使用单GPU (默认: False)')
    
    
    parser.add_argument('--fold', type=int, default=0, choices=[0,1,2,3,4],
                       help='训练/预测哪一折 (默认: 0)')
    
    parser.add_argument('--skip_verification', action='store_true',
                       help='跳过数据集验证步骤')
    
    parser.add_argument('--skip_preprocessing', action='store_true',
                       help='跳过预处理步骤')
    
    parser.add_argument('--skip_training', action='store_true',
                       help='跳过训练步骤')
    
    parser.add_argument('--skip_find_best', action='store_true',
                       help='跳过找最佳配置步骤')
    
    parser.add_argument('--skip_prediction', action='store_true',
                       help='跳过预测步骤')
    
    parser.add_argument('--train_all_folds', action='store_true',
                       help='训练所有5折 (将覆盖--fold参数)')
    
    parser.add_argument('--use_ensemble', action='store_true',
                       help='使用集成预测')
    
    parser.add_argument('--evaluate', action='store_true',
                       help='评估预测结果')
    
    args = parser.parse_args()
    
    # 设置默认路径
    if args.preprocessed_base is None:
        args.preprocessed_base = str(Path(args.raw_data_base).parent / 'nnUNet_preprocessed')
    
    if args.results_base is None:
        args.results_base = str(Path(args.raw_data_base).parent / 'nnUNet_results')
    
    # 检查GPU可用性
    if args.gpu_ids:
        gpu_available, gpu_count = check_gpu_availability(args.gpu_ids)
        if not gpu_available:
            print("错误: 指定的GPU不可用")
            sys.exit(1)
        
        # 如果强制使用单GPU或只有1个GPU可用
        if args.single_gpu or gpu_count == 1:
            print("使用单GPU模式")
            # 只使用第一个GPU
            first_gpu = args.gpu_ids.split(',')[0].strip()
            args.gpu_ids = first_gpu
            args.num_gpus = 1


    print(f"数据集ID: {args.dataset_id}")
    print(f"配置: {args.configuration}")
    print(f"训练器: {args.trainer}")
    print("="*60 + "\n")
    
    # 1. 设置环境
    setup_nnunet_environment(
        args.dataset_id,
        args.raw_data_base,
        args.preprocessed_base,
        args.results_base
    )
    
    # 2. 验证数据集 + 预处理
    if not args.skip_verification:
        verify=True
    else:
        verify=False
        print("\n⚠️  已跳过数据集验证步骤, 请确保数据集格式正确且完整!")

    if not args.skip_preprocessing:
        plan_and_preprocess(args.dataset_id, args.num_processes,verify)
    else:
        print("\n⚠️  已跳过预处理步骤, 请确保数据已经预处理完成!")
    
    # 3. 训练
    if not args.skip_training:
        if args.train_all_folds:
            train_all_folds(args.dataset_id, args.configuration, args.trainer, args.gpu_ids)
        else:
            train_model(args.dataset_id, args.configuration, args.fold, args.trainer, args.gpu_ids, args.num_gpus)

    
    # 4. 找最佳配置 （需要训练完5折）
    if not args.skip_find_best:
        find_best_configuration(args.dataset_id, args.configuration)
    
    # 5. 预测
    if not args.skip_prediction:
        if args.use_ensemble:
            # 集成预测需要先预测各个配置
            print("\n⚠️  集成预测需要先训练多个配置")
            print("请确保已经训练了2d和3d_fullres配置")
            
            # 预测所有配置和所有折
            for config in ['2d', '3d_fullres']:
                for fold in range(5):
                    predict_test_set(args.dataset_id, config, args.trainer, fold, args.gpu_ids)
            
            # 执行集成
            ensemble_predictions(args.dataset_id, ['2d', '3d_fullres'])
        else:
            predict_test_set(args.dataset_id, args.configuration, args.trainer, args.fold, args.gpu_ids)
    
    # 6. 评估
    if args.evaluate:
        if args.use_ensemble:
            pred_folder = os.path.join(args.results_base, f'Dataset{args.dataset_id}', 'ensemble')
        else:
            pred_folder = os.path.join(args.results_base, f'Dataset{args.dataset_id}', f'predicted_{args.configuration}')
        
        evaluate_predictions(args.dataset_id, pred_folder)
    
    print("\n" + "="*60)
    print("✅ nnUNet 训练测试已完成!")
    print("="*60)

if __name__ == "__main__":
    main()

