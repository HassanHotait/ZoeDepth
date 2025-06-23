# Environment Setup VENV

```bash
python -m venv env/venv_zoe
```

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117
```



# Evaluate on KITTI

```bash
python online_evaluate.py -m zoedepth_nk -d kitti_test --save_preds
```

| a1   | a2   | a3   | abs_rel | rmse  | log_10 | rmse_log | silog | sq_rel |
|------|------|------|---------|-------|--------|-----------|--------|--------|
| 0.966 | 0.993 | 0.996 | 0.057   | 2.362 | 0.026  | 0.087     | 7.363  | 0.204  |

# Evaluate on Prescan

```bash
python online_evaluate.py -m zoedepth_nk -d prescan --save_preds
```

| a1    | a2    | a3    | abs_rel | rmse  | log_10 | rmse_log | silog  | sq_rel |
|-------|-------|-------|---------|-------|--------|-----------|--------|--------|
| 0.053 | 0.116 | 0.252 | 2.072   | 3.081 | 0.444  | 1.098     | 40.093 | 7.417  |


# Evaluate on Prescan with Custom Weights

```bash
python online_evaluate.py -m zoedepth_nk --pretrained_resource="local::C:\Users\Hasan\shortcuts\monodepth3_checkpoints\ZoeDepthNKv1_05-Jun_20-28-f7aa0db17ecb_best.pt" -d prescan --save_preds
python online_evaluate.py -m zoedepth_nk --pretrained_resource="local::C:\Users\Hasan\shortcuts\monodepth3_checkpoints\ZoeDepthNKv1_05-Jun_20-28-f7aa0db17ecb_best.pt" -d my_kitti_set --save_preds

```

| a1    | a2    | a3    | abs_rel | rmse | log_10 | rmse_log | silog  | sq_rel |
|-------|-------|-------|---------|------|--------|-----------|--------|--------|
| 0.264 | 0.562 | 0.764 | 0.353   | 0.9  | 0.207  | 0.59      | 38.794 | 0.342  |


# Training on vKITTI2

When we train we dont start from random weights, we start from  'https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_NK.pt"

```bash
python train_mono.py -m zoedepth_nk -d prescan --pretrained_resource="url::https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_NK.pt"
```

python train_mono.py -m zoedepth_nk -d my_kitti_set --pretrained_resource="url::https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_NK.pt"



# Offline Eval


python offline_evaluate.py -m zoedepth_nk -d my_kitti_set --pred_dir C:\Users\Hasan\OneDrive\Documents\Datasets\KITTI\preds_20250611_201649
python offline_evaluate.py -m zoedepth_nk -d prescan --pred_dir C:\Users\Hasan\OneDrive\Desktop\Projects\ZoeDepth\postProcessedData\preds_20250623_233107

python online_evaluate.py -m zoedepth_nk -d my_kitti_set --save_preds




# Most Recent (Our Checkpoint)
python online_evaluate.py -m zoedepth_nk --pretrained_resource="local::C:\Users\Hasan\shortcuts\monodepth3_checkpoints\ZoeDepthNKv1_05-Jun_20-28-f7aa0db17ecb_best.pt" -d prescan --save_preds
python offline_evaluate.py -m zoedepth_nk -d prescan --pred_dir C:\Users\Hasan\OneDrive\Desktop\Projects\TestKitti\postProcessedData\preds_20250612_153126

# Official Checkpoint on NYU-KITTI Mix
python online_evaluate.py -m zoedepth_nk -d prescan --save_preds
python offline_evaluate.py -m zoedepth_nk -d prescan --pred_dir C:\Users\Hasan\OneDrive\Desktop\Projects\TestKitti\postProcessedData\preds_20250612_161556

# Original Crops

