# PPF-Net Script Reference

This document lists the main scripts in the repository, including:

- what each script does
- when to use it
- a typical command

Notes:

- the examples below follow the current experiment folders in this repository, such as `outputs/stage1_*` and `outputs/stage2_*`
- if your local outputs use different folder names, replace the paths accordingly

---

## 1. Data Preparation

### `scripts/generate_thz_feature_maps.py`

Purpose:

- converts raw THz CSV cubes into 2D feature maps
- mainly used by the older 2D feature-map baseline

Typical command:

```powershell
& D:/env/YOLO/python.exe scripts/generate_thz_feature_maps.py `
  --input dataset\thz_seed_only\FS `
  --output outputs\stage1\dataset_fs_features `
  --axis-values 1.0 2.0 3.0 `
  --axis-ranges 0.8:1.2 1.8:2.2
```

### `scripts/generate_feature_manifests.py`

Purpose:

- builds Stage-1 manifest CSV files from generated feature-map folders

Typical command:

```powershell
& D:/env/YOLO/python.exe scripts/generate_feature_manifests.py `
  --fs-root outputs\stage1\dataset_fs_features `
  --ts-root outputs\stage1\dataset_ts_features `
  --out-dir outputs\stage1\manifests
```

### `scripts/generate_dataset_splits.py`

Purpose:

- creates train / val / test splits from a manifest CSV

Stage-2 example:

```powershell
& D:/env/YOLO/python.exe scripts/generate_dataset_splits.py `
  --input-manifest outputs\stage2\manifests\rgb_fs_pairs.csv `
  --output-dir outputs\stage2\splits `
  --id-column sample_id `
  --group-column sample_name `
  --class-column class_name
```

### `scripts/generate_stage2_rgb_fs_manifest.py`

Purpose:

- builds paired RGB + FS manifests for Stage-2 experiments

Typical command:

```powershell
& D:/env/YOLO/python.exe scripts/generate_stage2_rgb_fs_manifest.py `
  --rgb-root datasets\images `
  --fs-root datasets\thz_seed_only\FS `
  --out-path outputs\stage2\manifests\rgb_fs_pairs.csv
```

### `scripts/crop_seed_images.py`

Purpose:

- segments and crops RGB seed images
- supports foreground-only and transparent-background export

Transparent-background example:

```powershell
& D:/env/YOLO/python.exe scripts/crop_seed_images.py `
  --input-root datasets\images `
  --output-root datasets\images_cropped_transparent `
  --transparent-background
```

---

## 2. Stage-1 Scripts

### `scripts/inspect_stage1_dataloader.py`

Purpose:

- checks whether the Stage-1 2D dataloader is working

### `scripts/train_stage1_unet.py`

Purpose:

- trains the older 2D THz feature-map baseline

### `scripts/train_stage1_spectral_unet.py`

Purpose:

- trains the average-spectrum baseline

### `scripts/train_stage1_pixel_spectral_unet.py`

Purpose:

- Stage-1 mainline
- trains the THz-only FS pixel-wise spectral reconstruction model

Typical command:

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

Purpose:

- reconstructs one full THz cube with the Stage-1 model

### `scripts/reconstruct_stage1_testset.py`

Purpose:

- reconstructs the entire Stage-1 test set and exports:
  - `reconstructed.csv`
  - image maps
  - error maps

Typical command:

```powershell
& D:/env/YOLO/python.exe scripts/reconstruct_stage1_testset.py `
  --manifest outputs\stage1\splits\test_pairs.csv `
  --checkpoint outputs\stage1_pixel_fs\checkpoints\stage1_pixel_spectral_unet_best.pt `
  --output-dir outputs\stage1_pixel_fs\predictions\test_reconstruction `
  --batch-size 1024
```

### `scripts/visualize_stage1_pixel_results.py`

Purpose:

- generates paper-style image boards for Stage-1 reconstruction results

### `scripts/visualize_stage1_spectral_unet.py`

Purpose:

- visualizes average-spectrum reconstruction curves

### `scripts/visualize_stage1_unet.py`

Purpose:

- visualizes the older 2D feature-map reconstruction baseline

---

## 3. Stage-2 Training Scripts

### `scripts/train_stage2_fs_only_baseline.py`

Purpose:

- fair Stage-2 FS-only baseline
- no RGB input

Typical command:

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

Purpose:

- Stage-2 TCN baseline
- an additional learning-based comparison model

Typical command:

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

Purpose:

- older pixel-level RGB + FS student
- mainly used as an earlier multimodal comparison model

### `scripts/train_stage2_rgb_fs_patch_student.py`

Purpose:

- current strongest Stage-2 model
- patch / local-region RGB + FS fusion model

Typical command:

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

## 4. Stage-2 Reconstruction, Comparison, and Analysis

### `scripts/reconstruct_stage2_fs_only_testset.py`

Purpose:

- reconstructs the full Stage-2 test set using a single-modality spectral model
- works for:
  - Stage-2 FS-only baseline
  - Stage-2 TCN baseline

Typical command:

```powershell
& D:/env/YOLO/python.exe scripts/reconstruct_stage2_fs_only_testset.py `
  --manifest outputs\stage2\splits\test_pairs.csv `
  --checkpoint outputs\stage2_tcn_baseline\checkpoints\stage2_tcn_baseline_best.pt `
  --output-dir outputs\stage2_tcn_baseline\predictions\test_reconstruction `
  --batch-size 512
```

### `scripts/reconstruct_stage2_patch_testset.py`

Purpose:

- reconstructs the full Stage-2 test set with the patch model

### `scripts/reconstruct_stage2_interpolation_testset.py`

Purpose:

- reconstructs the Stage-2 test set using traditional interpolation baselines
- supports:
  - `linear`
  - `pchip`
  - `cubic_spline`

Linear interpolation example:

```powershell
& D:/env/YOLO/python.exe scripts/reconstruct_stage2_interpolation_testset.py `
  --manifest outputs\stage2\splits\test_pairs.csv `
  --method linear `
  --output-dir outputs\stage2_interpolation_linear\predictions\test_reconstruction `
  --batch-size 512
```

### `scripts/analyze_thz_reconstruction.py`

Purpose:

- computes spectral and image-map metrics from reconstructed sample folders
- works for:
  - deep learning models
  - interpolation baselines

Typical command:

```powershell
& D:/env/YOLO/python.exe scripts/analyze_thz_reconstruction.py `
  --prediction-root outputs\stage2_rgb_fs_patch_student\predictions\test_reconstruction `
  --output-dir outputs\stage2_rgb_fs_patch_student\predictions\test_reconstruction\analysis
```

### `scripts/visualize_stage2_test_comparison.py`

Purpose:

- compares spectral curves between:
  - FS-only baseline
  - RGB + FS patch model

### `scripts/visualize_stage2_patch_results.py`

Purpose:

- generates paper-style image boards for the final Stage-2 patch model

### `scripts/benchmark_inference_models.py`

Purpose:

- benchmarks parameter count and forward inference speed
- supports:
  - Stage-1 teacher
  - FS-only baseline
  - TCN baseline
  - old RGB + FS student
  - patch RGB + FS student

Typical command:

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

## 5. Recommended Order

If you only care about the current final paper line, the recommended order is:

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

