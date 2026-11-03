#!/usr/bin/env python3
"""
数据集路径验证脚本
用于检查数据集文件是否存在，验证配置是否正确
"""

import os
from config_unetformer import UNetFormerConfig

def check_dataset_paths():
    """检查数据集路径和文件是否存在"""
    print("=" * 60)
    print("🔍 UNetFormer 数据集路径验证")
    print("=" * 60)
    
    # 加载配置
    config = UNetFormerConfig()
    print(f"📊 当前配置:")
    print(f"   数据集: {config.DATASET}")
    print(f"   根目录: {config.FOLDER}")
    print(f"   批次大小: {config.BATCH_SIZE}")
    print()
    
    # 检查根目录
    if not os.path.exists(config.FOLDER):
        print(f"❌ 数据集根目录不存在: {config.FOLDER}")
        print("   请检查FOLDER配置是否正确")
        return False
    else:
        print(f"✅ 数据集根目录存在: {config.FOLDER}")
    
    # 获取数据集特定配置
    dataset_config = config.get_dataset_specific_config(config.DATASET)
    
    if config.DATASET == 'Vaihingen':
        train_ids = dataset_config['train_ids']
        test_ids = dataset_config['test_ids']
        
        main_folder = config.FOLDER + 'Vaihingen/'
        data_pattern = main_folder + dataset_config['folder_structure']['data']
        label_pattern = main_folder + dataset_config['folder_structure']['label']
        eroded_pattern = main_folder + dataset_config['folder_structure']['eroded']
        
        print(f"\n📁 检查Vaihingen数据集文件...")
        print(f"   训练样本ID: {train_ids}")
        print(f"   测试样本ID: {test_ids}")
        print()
        
        # 检查训练文件
        print("🔍 检查训练文件:")
        missing_train_files = []
        
        for file_id in train_ids:
            data_file = data_pattern.format(file_id)
            label_file = label_pattern.format(file_id)
            
            if os.path.exists(data_file):
                print(f"   ✅ 数据文件: area{file_id}.tif")
            else:
                print(f"   ❌ 数据文件缺失: {data_file}")
                missing_train_files.append(data_file)
            
            if os.path.exists(label_file):
                print(f"   ✅ 标签文件: area{file_id}.tif")
            else:
                print(f"   ❌ 标签文件缺失: {label_file}")
                missing_train_files.append(label_file)
        
        print()
        
        # 检查测试文件（腐蚀标签）
        print("🔍 检查测试文件:")
        missing_test_files = []
        
        for file_id in test_ids:
            eroded_file = eroded_pattern.format(file_id)
            
            if os.path.exists(eroded_file):
                print(f"   ✅ 腐蚀标签: area{file_id}_noBoundary.tif")
            else:
                print(f"   ❌ 腐蚀标签缺失: {eroded_file}")
                missing_test_files.append(eroded_file)
        
        print()
        
        # 汇总结果
        total_missing = len(missing_train_files) + len(missing_test_files)
        
        if total_missing == 0:
            print("🎉 所有必需文件都存在！")
            print("✅ 您可以开始训练了")
            
            # 显示推荐的训练命令
            print()
            print("🚀 开始训练:")
            print("   python run_unetformer_training.py")
            
            return True
        else:
            print(f"❌ 发现 {total_missing} 个缺失文件")
            print()
            print("🔧 可能的解决方案:")
            print("1. 检查数据集是否完整下载")
            print("2. 验证文件路径是否正确")
            print("3. 确认文件名格式是否匹配")
            print()
            print("📋 预期的目录结构:")
            print("ISPRS_dataset/")
            print("└── Vaihingen/")
            print("    ├── ISPRS_semantic_labeling_Vaihingen/")
            print("    │   ├── top/")
            print("    │   │   ├── top_mosaic_09cm_area1.tif")
            print("    │   │   ├── top_mosaic_09cm_area3.tif")
            print("    │   │   └── ...")
            print("    │   └── gts_for_participants/")
            print("    │       ├── top_mosaic_09cm_area1.tif")
            print("    │       ├── top_mosaic_09cm_area3.tif")
            print("    │       └── ...")
            print("    └── ISPRS_semantic_labeing_Vaihingen_ground_truth_eroded_for_participants/")
            print("        ├── top_mosaic_09cm_area5_noBoundary.tif")
            print("        ├── top_mosaic_09cm_area15_noBoundary.tif")
            print("        └── ...")
            
            return False
    
    else:
        print(f"❓ 数据集 '{config.DATASET}' 的验证尚未实现")
        return False

def check_dependencies():
    """检查Python依赖包"""
    print("\n" + "=" * 60)
    print("📦 检查Python依赖包")
    print("=" * 60)
    
    required_packages = [
        'torch',
        'torchvision', 
        'timm',
        'einops',
        'numpy',
        'scikit-image',
        'scikit-learn',
        'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (缺失)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ 发现 {len(missing_packages)} 个缺失的依赖包")
        print("🔧 安装命令:")
        for package in missing_packages:
            print(f"   pip install {package}")
        return False
    else:
        print("\n✅ 所有依赖包都已安装")
        return True

def main():
    """主验证函数"""
    print("🔍 开始验证UNetFormer训练环境...")
    
    # 检查依赖包
    deps_ok = check_dependencies()
    
    # 检查数据集
    dataset_ok = check_dataset_paths()
    
    print("\n" + "=" * 60)
    print("📋 验证总结")
    print("=" * 60)
    
    if deps_ok and dataset_ok:
        print("🎉 所有检查都通过了！")
        print("✅ 您的环境已准备好进行UNetFormer训练")
        print()
        print("🚀 下一步:")
        print("   python run_unetformer_training.py")
    else:
        print("❌ 发现一些问题需要解决")
        print("🔧 请根据上述建议修复问题后重新运行此脚本")
        
        if not deps_ok:
            print("   - 安装缺失的Python包")
        if not dataset_ok:
            print("   - 检查数据集路径和文件")

if __name__ == '__main__':
    main() 