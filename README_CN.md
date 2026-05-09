# PPF-Net

PPF-Net 是一个面向**太赫兹反射光谱快速重建**的研究型代码仓库。

当前仓库的重点包括：

- THz-only 反射光谱先验学习
- RGB 引导的多模态特征融合
- 像素级与 patch 级光谱重建
- 完整 THz 立方体重建、CSV 导出与成像分析

## 当前主线

当前仓库中最强的实验主线是：

1. 第一阶段训练 THz-only FS 反射光谱像素级重建模型
2. 第二阶段训练 RGB + FS 配对重建模型
3. 把融合单位从单像素提升到局部 patch / 局部区域级
4. 与 FS-only、传统插值方法以及额外深度学习基线做对比

现有结果表明，真正带来提升的关键是 **patch / 局部区域级跨模态融合**，而不是简单 RGB 拼接。

## 仓库结构

- `ppfnet/`
  核心 Python 包。
- `scripts/`
  数据处理、训练、重建、可视化、分析和 benchmark 脚本。
- `outputs/`
  checkpoint、日志、重建 CSV、成像图和分析结果。
- `datasets/`
  RGB 图像和原始 THz CSV 数据。

## 安装

先安装常用依赖：

```powershell
pip install numpy scipy pillow matplotlib torch torchvision
```

如果需要以可编辑模式安装项目：

```powershell
pip install -e .
```

## 推荐实验流程

下面的命令采用你当前仓库里实际在用的实验目录风格，例如 `outputs/stage1_*`、`outputs/stage2_*`。

### 1. 训练第一阶段 THz-only teacher

```powershell
& D:/env/YOLO/python.exe scripts/train_stage1_pixel_spectral_unet.py `
  --modality fs `
  --train-manifest outputs\stage1\splits\train_pairs.csv `
  --val-manifest outputs\stage1\splits\val_pairs.csv `
  --test-manifest outputs\stage1\splits\test_pairs.csv `
  --output-dir outputs\stage1_pixel_fs `
  --epochs 80 `
  --batch-size 256 `
  --amp
```

### 2. 训练第二阶段公平基线 FS-only

```powershell
& D:/env/YOLO/python.exe scripts/train_stage2_fs_only_baseline.py `
  --train-manifest outputs\stage2\splits\train_pairs.csv `
  --val-manifest outputs\stage2\splits\val_pairs.csv `
  --test-manifest outputs\stage2\splits\test_pairs.csv `
  --output-dir outputs\stage2_fs_only_baseline_random `
  --epochs 800 `
  --batch-size 2 `
  --max-pixels-per-sample 512 `
  --amp
```

### 3. 训练最终的 RGB + FS patch 模型

```powershell
& D:/env/YOLO/python.exe scripts/train_stage2_rgb_fs_patch_student.py `
  --train-manifest outputs\stage2\splits\train_pairs.csv `
  --val-manifest outputs\stage2\splits\val_pairs.csv `
  --test-manifest outputs\stage2\splits\test_pairs.csv `
  --teacher-checkpoint outputs\stage1_pixel_fs\checkpoints\stage1_pixel_spectral_unet_best.pt `
  --output-dir outputs\stage2_rgb_fs_patch_student `
  --epochs 200 `
  --batch-size 64 `
  --rgb-patch-size 64 64 `
  --thz-patch-size 7 `
  --max-pixels-per-sample 512 `
  --teacher-weight 0 `
  --amp
```

### 4. 重建测试集并导出 CSV / 成像图

FS-only baseline：

```powershell
& D:/env/YOLO/python.exe scripts/reconstruct_stage2_fs_only_testset.py `
  --manifest outputs\stage2\splits\test_pairs.csv `
  --checkpoint outputs\stage2_fs_only_baseline_random\checkpoints\stage2_fs_only_baseline_best.pt `
  --output-dir outputs\stage2_fs_only_baseline_random\predictions\test_reconstruction `
  --batch-size 512
```

Patch 模型：

```powershell
& D:/env/YOLO/python.exe scripts/reconstruct_stage2_patch_testset.py `
  --manifest outputs\stage2\splits\test_pairs.csv `
  --checkpoint outputs\stage2_rgb_fs_patch_student\checkpoints\stage2_rgb_fs_patch_student_best.pt `
  --output-dir outputs\stage2_rgb_fs_patch_student\predictions\test_reconstruction `
  --batch-size 128
```

### 5. 运行光谱与成像分析

```powershell
& D:/env/YOLO/python.exe scripts/analyze_thz_reconstruction.py `
  --prediction-root outputs\stage2_rgb_fs_patch_student\predictions\test_reconstruction `
  --output-dir outputs\stage2_rgb_fs_patch_student\predictions\test_reconstruction\analysis
```

### 6. 统计模型参数量与推理速度

```powershell
& D:/env/YOLO/python.exe scripts/benchmark_inference_models.py `
  --checkpoints `
  outputs\stage1_pixel_fs\checkpoints\stage1_pixel_spectral_unet_best.pt `
  outputs\stage2_fs_only_baseline_random\checkpoints\stage2_fs_only_baseline_best.pt `
  outputs\stage2_tcn_baseline\checkpoints\stage2_tcn_baseline_best.pt `
  outputs\stage2_rgb_fs_student_local_global\checkpoints\stage2_rgb_fs_student_best.pt `
  outputs\stage2_rgb_fs_patch_student\checkpoints\stage2_rgb_fs_patch_student_best.pt `
  --warmup-iters 20 `
  --benchmark-iters 100 `
  --batch-size 128 `
  --pixels-per-sample 128 `
  --output-csv outputs\model_benchmark.csv `
  --output-json outputs\model_benchmark.json
```

## 外部对比实验

当前仓库已经支持：

- 传统插值方法
  - 线性插值 `linear`
  - `PCHIP`
  - `cubic spline`
- 单模态学习基线
  - 第二阶段 `FS-only baseline`
  - 第二阶段 `TCN baseline`
- 多模态方法
  - 早期像素级 `RGB + FS old`
  - 最终 `RGB + FS patch`

传统插值方法不需要训练，只需直接对测试集做重建：

```powershell
& D:/env/YOLO/python.exe scripts/reconstruct_stage2_interpolation_testset.py `
  --manifest outputs\stage2\splits\test_pairs.csv `
  --method linear `
  --output-dir outputs\stage2_interpolation_linear\predictions\test_reconstruction `
  --batch-size 512
```

## 文档入口

- 英文项目说明：[README.md](./README.md)
- 英文脚本说明：[SCRIPTS_README.md](./SCRIPTS_README.md)
- 中文脚本说明：[SCRIPTS_README_CN.md](./SCRIPTS_README_CN.md)


