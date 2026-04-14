

import os
import sys
import argparse
import subprocess
from pathlib import Path

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

def run_command(command, description):
    """
    运行系统命令并打印输出
    
    Args:
        command: 要运行的命令列表
        description: 命令描述
    """
    print(f"\n{'='*60}")
    print(f"执行: {description}")
    print(f"命令: {' '.join(command)}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 执行成功!")
            if result.stdout:
                print("输出:", result.stdout)
        else:
            print("❌ 执行失败!")
            print("错误:", result.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"❌ 执行出错: {e}")
        sys.exit(1)

def verify_dataset(dataset_id):
    """
    步骤1: 验证数据集的完整性
    
    Args:
        dataset_id: 数据集ID (如 '001')
    """
    print("\n🔍 步骤1: 验证数据集完整性")
    command = [
        'nnUNetv2_plan_and_preprocess',
        '-d', dataset_id,
        '--verify_dataset_integrity'
    ]
    run_command(command, "验证数据集结构是否正确")

def plan_and_preprocess(dataset_id, num_processes=8):
    """
    步骤2: 规划和预处理数据  会自动分析 spacing、patch size、决定 2D / 3D、生成预处理数据至:nnUNet_preprocessed/
    
    Args:
        dataset_id: 数据集ID
        num_processes: 并行处理的进程数
    
    参数解释:
        -d: 数据集ID
        -pl: 规划器名称 (默认: ExperimentPlanner)
        -c: 配置 (2D, 3D_fullres, 3D_lowres)
        -np: 并行处理的进程数
    """
    print("\n🛠️ 步骤2: 数据预处理")
    command = [
        'nnUNetv2_plan_and_preprocess',
        '-d', dataset_id,
        '-np', str(num_processes)  # 并行进程数
    ]
    run_command(command, "执行数据预处理和规划")

def train_model(dataset_id, configuration='2d', fold=0, trainer='nnUNetTrainer'):
    """
    步骤3: 训练模型
    
    Args:
        dataset_id: 数据集ID
        configuration: 网络配置 ('2d', '3d_fullres', '3d_lowres')
        fold: 交叉验证的折数 (0-4, 或 'all' 训练所有)
        trainer: 训练器名称
    
    参数解释:
        -d: 数据集ID
        -c: 配置 (2D, 3D_fullres, 3D_lowres)
        -f: 训练哪一折 (0,1,2,3,4 或 all)
        -tr: 使用的训练器
        -p: 规划器名称 (可选)
        --npz: 保存npz文件用于后期评估
    """
    print(f"\n🏋️ 步骤3: 训练模型 (配置={configuration}, fold={fold})")
    command = [
        'nnUNetv2_train',
        dataset_id,
        configuration,
        str(fold),
        '-tr', trainer
    ]
    run_command(command, f"训练模型 - fold {fold}")

def train_all_folds(dataset_id, configuration='2d', trainer='nnUNetTrainer'):
    """
    训练所有5折交叉验证
    
    Args:
        dataset_id: 数据集ID
        configuration: 网络配置
        trainer: 训练器名称
    """
    print("\n🔄 训练所有5折交叉验证")
    for fold in range(5):
        train_model(dataset_id, configuration, fold, trainer)

def find_best_configuration(dataset_id, configuration='2d'):
    """
    步骤4: 找出最佳配置
    
    Args:
        dataset_id: 数据集ID
        configuration: 网络配置
    
    参数解释:
        -d: 数据集ID
        -c: 配置
    """
    print("\n🔬 步骤4: 找出最佳配置")
    command = [
        'nnUNetv2_find_best_configuration',
        dataset_id,
        '-c', configuration
    ]
    run_command(command, "自动选择最佳配置")

def predict_test_set(dataset_id, configuration='2d', trainer='nnUNetTrainer', fold=0):
    """
    步骤5: 预测测试集
    
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
    print(f"\n🔮 步骤5: 预测测试集")
    
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
    run_command(command, f"预测测试集 - 使用模型 fold {fold}")

def ensemble_predictions(dataset_id, configurations=['2d', '3d_fullres'], folds=[0,1,2,3,4]):
    """
    步骤6: 集成多个模型的预测结果
    
    Args:
        dataset_id: 数据集ID
        configurations: 要集成的配置列表
        folds: 要集成的折数
    
    参数解释:
        -i: 输入文件夹列表
        -o: 输出文件夹
        --ensemble_method: 集成方法 (softmax 或 majority_vote)
    """
    print(f"\n🤝 步骤6: 集成预测结果")
    
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
    步骤7: 评估预测结果
    
    Args:
        dataset_id: 数据集ID
        predictions_folder: 预测结果文件夹
    """
    print(f"\n📊 步骤7: 评估预测结果")
    
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

def main(dataset_id=None, raw_data_base=None):

    parser = argparse.ArgumentParser(description='nnUNet训练&测试')
    
    # 必需参数
    parser.add_argument('--dataset_id', type=str, required=True,
                       help='数据集ID,只需三位数字例如: 001, 002')
    parser.add_argument('--raw_data_base', type=str, required=True, default=r'C:\Users\75267\Documents\python项目\LR-HCC\nnUNet\nnUNet_raw',
                       help='nnUNet_raw 路径')
    
    # 可选参数
    parser.add_argument('--preprocessed_base', type=str, default=r'C:\Users\75267\Documents\python项目\LR-HCC\nnUNet\nnUNet_preprocessed',
                       help='nnUNet_preprocessed 路经')
    
    parser.add_argument('--results_base', type=str, default=r'C:\Users\75267\Documents\python项目\LR-HCC\nnUNet\nnUNet_results',
                       help='nnUNet_results的根目录')
    
    parser.add_argument('--configuration', type=str, default='2d',
                       choices=['2d', '3d_fullres', '3d_lowres'],
                       help='网络配置: 2d, 3d_fullres, 3d_lowres (默认: 2d)')
    
    parser.add_argument('--trainer', type=str, default='nnUNetTrainer',
                       help='训练器名称 (默认: nnUNetTrainer)')
    
    parser.add_argument('--num_processes', type=int, default=2,
                       help='预处理并行进程数 (默认: 8)')
    
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
    
    if dataset_id is not None or raw_data_base is not None:
        cli_args = []
        if dataset_id is not None:
            cli_args.extend(['--dataset_id', str(dataset_id)])
        if raw_data_base is not None:
            cli_args.extend(['--raw_data_base', str(raw_data_base)])
        args = parser.parse_args(cli_args)
    else:
        args = parser.parse_args()
    
    # 设置默认路径
    if args.preprocessed_base is None:
        args.preprocessed_base = str(Path(args.raw_data_base).parent / 'nnUNet_preprocessed')
    
    if args.results_base is None:
        args.results_base = str(Path(args.raw_data_base).parent / 'nnUNet_results')
    

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
    
    # 2. 验证数据集
    if not args.skip_verification:
        verify_dataset(args.dataset_id)
    
    # 3. 预处理
    if not args.skip_preprocessing:
        plan_and_preprocess(args.dataset_id, args.num_processes)
    
    # 4. 训练
    if not args.skip_training:
        if args.train_all_folds:
            train_all_folds(args.dataset_id, args.configuration, args.trainer)
        else:
            train_model(args.dataset_id, args.configuration, args.fold, args.trainer)
    
    # 5. 找最佳配置
    if not args.skip_find_best:
        find_best_configuration(args.dataset_id, args.configuration)
    
    # 6. 预测
    if not args.skip_prediction:
        if args.use_ensemble:
            # 集成预测需要先预测各个配置
            print("\n⚠️  集成预测需要先训练多个配置")
            print("请确保已经训练了2d和3d_fullres配置")
            
            # 预测所有配置和所有折
            for config in ['2d', '3d_fullres']:
                for fold in range(5):
                    predict_test_set(args.dataset_id, config, args.trainer, fold)
            
            # 执行集成
            ensemble_predictions(args.dataset_id, ['2d', '3d_fullres'])
        else:
            predict_test_set(args.dataset_id, args.configuration, args.trainer, args.fold)
    
    # 7. 评估
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
    dataset_id = '001'
    raw_data_bas = r'C:\Users\75267\Documents\python项目\LR-HCC\nnUNet\nnUNet_raw'
    main(dataset_id=dataset_id, raw_data_base=raw_data_bas)

# python main.py --dataset_id 001 --raw_data_base C:\Users\75267\Documents\python项目\LR-HCC\nnUNet\nnUNet_raw 
