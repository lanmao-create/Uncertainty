#!/usr/bin/env python3
"""
UNetFormer 简化启动脚本
使用您的实际数据集路径进行训练
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# 导入我们的模块
from model.UNetFormer import UNetFormer
from config_unetformer import UNetFormerConfig

# ===== 简化的数据集类 =====
class SimpleUNetFormerDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        
        # 基本配置
        if config.DATASET == 'Vaihingen':
            self.train_ids = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37']
            self.labels = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"]
            self.n_classes = len(self.labels)
            
            # 文件路径
            main_folder = config.FOLDER + 'Vaihingen/'
            self.data_pattern = main_folder + 'ISPRS_semantic_labeling_Vaihingen/top/top_mosaic_09cm_area{}.tif'
            self.label_pattern = main_folder + 'ISPRS_semantic_labeling_Vaihingen/gts_for_participants/top_mosaic_09cm_area{}.tif'
            
            # 颜色调色板
            self.palette = {0: (255, 255, 255), 1: (0, 0, 255), 2: (0, 255, 255),
                           3: (0, 255, 0), 4: (255, 255, 0), 5: (255, 0, 0), 6: (0, 0, 0)}
            self.invert_palette = {v: k for k, v in self.palette.items()}
        
        # 文件路径列表
        self.data_files = [self.data_pattern.format(id) for id in self.train_ids]
        self.label_files = [self.label_pattern.format(id) for id in self.train_ids]
        
        # 检查文件是否存在
        missing_files = []
        for f in self.data_files + self.label_files:
            if not os.path.isfile(f):
                missing_files.append(f)
        
        if missing_files:
            print("❌ 以下文件不存在:")
            for f in missing_files:
                print(f"   {f}")
            print(f"\n请检查数据集路径: {config.FOLDER}")
            return
        else:
            print(f"✅ 找到 {len(self.data_files)} 个训练文件")
        
        # 缓存
        self.data_cache = {}
        self.label_cache = {}

    def __len__(self):
        return self.config.BATCH_SIZE * 100  # 较小的epoch大小用于测试

    def convert_from_color(self, arr_3d):
        """RGB颜色编码转换为灰度标签"""
        arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
        for c, i in self.invert_palette.items():
            m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
            arr_2d[m] = i
        return arr_2d

    def get_random_patch(self, img, window_shape):
        """获取随机patch"""
        w, h = window_shape
        W, H = img.shape[-2:]
        x1 = np.random.randint(0, max(1, W - w))
        x2 = x1 + w
        y1 = np.random.randint(0, max(1, H - h))
        y2 = y1 + h
        return x1, x2, y1, y2

    def __getitem__(self, idx):
        from skimage import io
        import random
        
        # 随机选择一个文件
        file_idx = random.randint(0, len(self.data_files) - 1)
        
        # 加载数据
        if file_idx in self.data_cache:
            data = self.data_cache[file_idx]
        else:
            data = io.imread(self.data_files[file_idx])
            data = data.transpose((2, 0, 1)).astype(np.float32) / 255.0
            if self.config.CACHE:
                self.data_cache[file_idx] = data
        
        # 加载标签
        if file_idx in self.label_cache:
            label = self.label_cache[file_idx]
        else:
            label_img = io.imread(self.label_files[file_idx])
            label = self.convert_from_color(label_img).astype(np.int64)
            if self.config.CACHE:
                self.label_cache[file_idx] = label
        
        # 获取随机patch
        x1, x2, y1, y2 = self.get_random_patch(data, self.config.WINDOW_SIZE)
        data_patch = data[:, x1:x2, y1:y2]
        label_patch = label[x1:x2, y1:y2]
        
        # 创建虚拟的边界和对象掩码（全零，因为我们没有这些标注）
        boundary_patch = np.zeros_like(label_patch)
        object_patch = np.zeros_like(label_patch)
        
        return (torch.from_numpy(data_patch),
                torch.from_numpy(boundary_patch),
                torch.from_numpy(object_patch),
                torch.from_numpy(label_patch))

# ===== 简化的损失函数 =====
class SimpleCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)
    
    def forward(self, pred, target):
        return self.ce_loss(pred, target)

# ===== 主训练函数 =====
def main():
    print("🚀 启动UNetFormer训练...")
    
    # 加载配置
    config = UNetFormerConfig()
    print(f"📊 配置信息:")
    print(f"   数据集: {config.DATASET}")
    print(f"   批次大小: {config.BATCH_SIZE}")
    print(f"   训练轮数: {config.EPOCHS}")
    print(f"   输入尺寸: {config.WINDOW_SIZE}")
    print(f"   数据集路径: {config.FOLDER}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用设备: {device}")
    
    # 创建数据集
    print("📁 加载数据集...")
    dataset = SimpleUNetFormerDataset(config)
    if not hasattr(dataset, 'n_classes'):
        print("❌ 数据集加载失败，请检查路径配置")
        return
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=min(4, config.NUM_WORKERS),
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 创建模型
    print("🔧 创建模型...")
    model = UNetFormer(
        decode_channels=config.DECODE_CHANNELS,
        dropout=config.DROPOUT,
        backbone_name=config.BACKBONE,
        pretrained=config.PRETRAINED,
        window_size=config.WINDOW_SIZE_ATTENTION,
        num_classes=dataset.n_classes
    ).to(device)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 模型参数: {total_params:,} (可训练: {trainable_params:,})")
    
    # 创建优化器
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # 创建学习率调度器
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=config.LR_SCHEDULER_MILESTONES, 
        gamma=config.LR_SCHEDULER_GAMMA
    )
    
    # 创建损失函数
    criterion = SimpleCrossEntropyLoss().to(device)
    
    # 创建保存目录
    save_dir = f'./results_{config.DATASET.lower()}'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"💾 模型保存目录: {save_dir}")
    print("🎯 开始训练...")
    print("=" * 60)
    
    # 训练循环
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # 使用tqdm显示进度
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.EPOCHS}")
        
        for batch_idx, (data, boundary, object_mask, target) in enumerate(pbar):
            # 移动到设备
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # 前向传播
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{total_loss/num_batches:.4f}'
            })
            
            # 限制每个epoch的batch数量（用于快速测试）
            if batch_idx >= 10:  # 每个epoch只训练10个batch用于快速测试
                break
        
        # 更新学习率
        scheduler.step()
        avg_loss = total_loss / num_batches
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch:2d} - 损失: {avg_loss:.6f} - 学习率: {current_lr:.6f}")
        
        # 保存模型
        if epoch % config.SAVE_EPOCH == 0:
            save_path = os.path.join(save_dir, f'unetformer_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config.__dict__
            }, save_path)
            print(f"💾 模型已保存: {save_path}")
    
    print("🎉 训练完成!")

if __name__ == '__main__':
    main() 