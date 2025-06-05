# Evaluate on KITTI

```bash
python evaluate.py -m zoedepth_nk -d kitti_test
```

| a1   | a2   | a3   | abs_rel | rmse  | log_10 | rmse_log | silog | sq_rel |
|------|------|------|---------|-------|--------|-----------|--------|--------|
| 0.966 | 0.993 | 0.996 | 0.057   | 2.362 | 0.026  | 0.087     | 7.363  | 0.204  |

# Evaluate on Prescan

```bash
python evaluate.py -m zoedepth_nk -d prescan
```

| a1    | a2    | a3    | abs_rel | rmse  | log_10 | rmse_log | silog  | sq_rel |
|-------|-------|-------|---------|-------|--------|-----------|--------|--------|
| 0.053 | 0.116 | 0.252 | 2.072   | 3.081 | 0.444  | 1.098     | 40.093 | 7.417  |


# Evaluate on Prescan with Custom Weights

```bash
python evaluate.py -m zoedepth_nk --pretrained_resource="local::C:\Users\Hasan\shortcuts\monodepth3_checkpoints\ZoeDepthNKv1_05-Jun_20-28-f7aa0db17ecb_best.pt" -d prescan
```

| a1    | a2    | a3    | abs_rel | rmse | log_10 | rmse_log | silog  | sq_rel |
|-------|-------|-------|---------|------|--------|-----------|--------|--------|
| 0.264 | 0.562 | 0.764 | 0.353   | 0.9  | 0.207  | 0.59      | 38.794 | 0.342  |


# Training on vKITTI2

When we train we dont start from random weights, we start from  'https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_NK.pt"

```bash
python train_mono.py -m zoedepth_nk -d prescan --pretrained_resource="url::https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_NK.pt"
```

