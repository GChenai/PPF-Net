# PPF-Net 脚本说明

这份文档列出当前仓库中主要脚本的：

- 作用
- 适用场景
- 典型命令

说明：

- 下方命令采用你当前仓库里实际使用的实验目录风格，例如 `outputs/stage1_*`、`outputs/stage2_*`
- 如果你本地目录名不同，直接把命令中的路径替换成自己的实际路径即可

---

## 一、数据处理脚本

### `scripts/generate_thz_feature_maps.py`

作用：

- 把原始 THz CSV 立方体转换成 2D 特征图
- 主要用于较早期的 2D 特征图 baseline

典型命令：

```powershell
& D:/env/YOLO/python.exe scripts/generate_thz_feature_maps.py `
  --input dataset\thz_seed_only\FS `
  --output outputs\stage1\dataset_fs_features `
  --axis-values 1.0 2.0 3.0 `
  --axis-ranges 0.8:1.2 1.8:2.2
```

### `scripts/generate_feature_manifests.py`

作用：

- 为第一阶段 2D 特征图实验生成 manifest

典型命令：

```powershell
& D:/env/YOLO/python.exe scripts/generate_feature_manifests.py `
  --fs-root outputs\stage1\dataset_fs_features `
  --ts-root outputs\stage1\dataset_ts_features `
  --out-dir outputs\stage1\manifests
```

### `scripts/generate_dataset_splits.py`

作用：

- 根据 manifest 生成 train / val / test 划分

第二阶段示例：

```powershell
& D:/env/YOLO/python.exe scripts/generate_dataset_splits.py `
  --input-manifest outputs\stage2\manifests\rgb_fs_pairs.csv `
  --output-dir outputs\stage2\splits `
  --id-column sample_id `
  --group-column sample_name `
  --class-column class_name
```

### `scripts/generate_stage2_rgb_fs_manifest.py`

作用：

- 为第二阶段 RGB + FS 配对实验生成 manifest

典型命令：

```powershell
& D:/env/YOLO/python.exe scripts/generate_stage2_rgb_fs_manifest.py `
  --rgb-root datasets\images `
  --fs-root datasets\thz_seed_only\FS `
  --out-path outputs\stage2\manifests\rgb_fs_pairs.csv
```

### `scripts/crop_seed_images.py`

作用：

- 对 RGB 西瓜籽图像做裁剪、前景保留或透明背景导出

透明背景示例：

```powershell
& D:/env/YOLO/python.exe scripts/crop_seed_images.py `
  --input-root datasets\images `
  --output-root datasets\images_cropped_transparent `
  --transparent-background
```

---

## 二、第一阶段脚本

### `scripts/inspect_stage1_dataloader.py`

作用：

- 检查第一阶段 2D 特征图 dataloader 是否正常

### `scripts/train_stage1_unet.py`

作用：

- 训练第一阶段较早期的 2D 特征图 baseline

### `scripts/train_stage1_spectral_unet.py`

作用：

- 训练平均光谱 baseline

### `scripts/train_stage1_pixel_spectral_unet.py`

作用：

- 第一阶段主线
- 训练 THz-only FS 反射光谱像素级重建模型

典型命令：

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

### `scripts/reconstruct_stage1_pixel_cube.py`

作用：

- 用第一阶段模型重建单个样本的完整 THz 立方体

### `scripts/reconstruct_stage1_testset.py`

作用：

- 重建整个第一阶段测试集
- 导出：
  - `reconstructed.csv`
  - 成像图
  - 误差图

典型命令：

```powershell
& D:/env/YOLO/python.exe scripts/reconstruct_stage1_testset.py `
  --manifest outputs\stage1\splits\test_pairs.csv `
  --checkpoint outputs\stage1_pixel_fs\checkpoints\stage1_pixel_spectral_unet_best.pt `
  --output-dir outputs\stage1_pixel_fs\predictions\test_reconstruction `
  --batch-size 1024
```

### `scripts/visualize_stage1_pixel_results.py`

作用：

- 为第一阶段生成论文图版

### `scripts/visualize_stage1_spectral_unet.py`

作用：

- 可视化平均光谱重建曲线

### `scripts/visualize_stage1_unet.py`

作用：

- 可视化第一阶段 2D 特征图 baseline 的结果

---

## 三、第二阶段训练脚本

### `scripts/train_stage2_fs_only_baseline.py`

作用：

- 第二阶段公平基线
- 只用反射光谱，不引入 RGB

典型命令：

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

### `scripts/train_stage2_tcn_baseline.py`

作用：

- 第二阶段 TCN baseline
- 作为额外学习型对比模型

典型命令：

```powershell
& D:/env/YOLO/python.exe scripts/train_stage2_tcn_baseline.py `
  --train-manifest outputs\stage2\splits\train_pairs.csv `
  --val-manifest outputs\stage2\splits\val_pairs.csv `
  --test-manifest outputs\stage2\splits\test_pairs.csv `
  --output-dir outputs\stage2_tcn_baseline `
  --epochs 800 `
  --batch-size 256 `
  --base-channels 32 `
  --kernel-size 3 `
  --num-blocks 6 `
  --max-pixels-per-sample 512 `
  --amp
```

### `scripts/train_stage2_rgb_fs_student.py`

作用：

- 第二阶段较早期的 RGB + FS 像素级融合模型
- 主要作为旧版多模态对比模型

### `scripts/train_stage2_rgb_fs_patch_student.py`

作用：

- 第二阶段当前最强主线
- 基于 patch / 局部区域级特征融合的 RGB + FS 重建模型

典型命令：

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

---

## 四、第二阶段重建、对比与分析脚本

### `scripts/reconstruct_stage2_fs_only_testset.py`

作用：

- 用单模态光谱模型重建整个第二阶段测试集
- 当前可用于：
  - `FS-only baseline`
  - `TCN baseline`

典型命令：

```powershell
& D:/env/YOLO/python.exe scripts/reconstruct_stage2_fs_only_testset.py `
  --manifest outputs\stage2\splits\test_pairs.csv `
  --checkpoint outputs\stage2_tcn_baseline\checkpoints\stage2_tcn_baseline_best.pt `
  --output-dir outputs\stage2_tcn_baseline\predictions\test_reconstruction `
  --batch-size 512
```

### `scripts/reconstruct_stage2_patch_testset.py`

作用：

- 用第二阶段 patch 模型重建整个测试集

### `scripts/reconstruct_stage2_interpolation_testset.py`

作用：

- 用传统插值方法重建第二阶段测试集
- 当前支持：
  - `linear`
  - `pchip`
  - `cubic_spline`

线性插值示例：

```powershell
& D:/env/YOLO/python.exe scripts/reconstruct_stage2_interpolation_testset.py `
  --manifest outputs\stage2\splits\test_pairs.csv `
  --method linear `
  --output-dir outputs\stage2_interpolation_linear\predictions\test_reconstruction `
  --batch-size 512
```

### `scripts/analyze_thz_reconstruction.py`

作用：

- 对重建目录做光谱级与成像级指标分析
- 可用于：
  - 深度模型
  - 插值基线

典型命令：

```powershell
& D:/env/YOLO/python.exe scripts/analyze_thz_reconstruction.py `
  --prediction-root outputs\stage2_rgb_fs_patch_student\predictions\test_reconstruction `
  --output-dir outputs\stage2_rgb_fs_patch_student\predictions\test_reconstruction\analysis
```

### `scripts/visualize_stage2_test_comparison.py`

作用：

- 对比：
  - FS-only baseline
  - RGB + FS patch
  的谱线重建曲线

### `scripts/visualize_stage2_patch_results.py`

作用：

- 为最终 patch 模型生成论文图版

### `scripts/benchmark_inference_models.py`

作用：

- 统计模型参数量和前向推理速度
- 当前支持：
  - Stage-1 teacher
  - FS-only baseline
  - TCN baseline
  - RGB + FS old
  - RGB + FS patch

典型命令：

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

---

## 五、推荐顺序

如果你只关注当前最终论文主线，推荐顺序是：

1. `generate_stage2_rgb_fs_manifest.py`
2. `generate_dataset_splits.py`
3. `train_stage1_pixel_spectral_unet.py`
4. `train_stage2_fs_only_baseline.py`
5. `train_stage2_tcn_baseline.py`
6. `train_stage2_rgb_fs_patch_student.py`
7. `reconstruct_stage2_fs_only_testset.py`
8. `reconstruct_stage2_interpolation_testset.py`
9. `reconstruct_stage2_patch_testset.py`
10. `analyze_thz_reconstruction.py`
11. `visualize_stage2_patch_results.py`
12. `benchmark_inference_models.py`

