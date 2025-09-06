# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

"""Miscellaneous utility functions."""

from scipy import ndimage

import base64
import math
import re
from io import BytesIO

import matplotlib
import matplotlib.cm
import numpy as np
import requests
import torch
import torch.distributed as dist
import torch.nn
import torch.nn as nn
import torch.utils.data.distributed
from PIL import Image
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt


class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg


def denormalize(x):
    """Reverses the imagenet normalization applied to the input.

    Args:
        x (torch.Tensor - shape(N,3,H,W)): input tensor

    Returns:
        torch.Tensor - shape(N,3,H,W): Denormalized input
    """
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
    return x * std + mean


class RunningAverageDict:
    """A dictionary of running averages."""
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if new_dict is None:
            return

        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        if self._dict is None:
            return None
        return {key: value.get_value() for key, value in self._dict.items()}
    
    def __repr__(self):
        return repr(self.get_value())
    
class Metrics:
    def __init__(self):
        self._metrics = {}

    def __getitem__(self, key):
        if key not in self._metrics:
            self._metrics[key] = RunningAverageDict()
        return self._metrics[key]

    def __setitem__(self, key, value):
        if not isinstance(value, RunningAverageDict):
            raise TypeError(f"Expected RunningAverageDict, got {type(value)}")
        self._metrics[key] = value

    def get_all_values(self):
        return {key: rad.get_value() for key, rad in self._metrics.items()}

    def to_dict(self, round_vals=True, round_precision=3):
        r = (lambda m: round(m, round_precision)) if round_vals else (lambda m: m)
        return {
            key: {k: r(v) for k, v in rad.get_value().items()}
            for key, rad in self._metrics.items()
        }

    def __repr__(self):
        return repr(self.to_dict())
    
# class ObjectMetrics:
#     def __init__(self):
#         self.obj_count = 0
#         self.gt = []
#         self.pred = []
#         self.error = []

#     def update(self,gt, pred, labels):
#         if gt.shape[-2:] != pred.shape[-2:]:
#             pred = nn.functional.interpolate(
#                 pred, gt.shape[-2:], mode='bilinear', align_corners=True)
#         gt = gt.squeeze().cpu().numpy()
#         pred = pred.squeeze().cpu().numpy()
#         assert gt.shape == pred.shape, f"GT shape {gt.shape} does not match prediction shape {pred.shape}"
#         for l in labels:
#             obj_center3d = l['center_3d'].squeeze().cpu().numpy()
#             obj_center3d = np.array([int(obj_center3d[0]), int(obj_center3d[1])])
#             try:
#                 obj_depth = gt[obj_center3d[1], obj_center3d[0]]
#             except:
#                 print(f"Warning: Object center {obj_center3d} is out of bounds for the ground truth depth map of shape {gt.shape}. Skipping this object.")
#                 continue
#             if  obj_depth!= 0 and l['occlusion'] == 0:
#                 self.gt.append(obj_depth)
#                 self.pred.append(pred[obj_center3d[1], obj_center3d[0]])
#                 self.error.append(obj_depth - pred[obj_center3d[1], obj_center3d[0]])
#                 self.obj_count += 1

#     def get_value(self):
#         if len(self.gt) == 0 or len(self.pred) == 0:
#             return {"absrel": None, "count": 0}
#         gt = np.array(self.gt)
#         pred = np.array(self.pred)
#         absrel = np.mean(np.abs(gt - pred) / gt)
#         return {"abs_rel": absrel, "count": self.obj_count}
    
#     def __repr__(self):
#         value = self.get_value()
#         return f"ObjectMetrics(abs_rel={value['abs_rel']:.4f}, count={value['count']})"

class ObjectMetrics:
    def __init__(self, label="MIDAS [relative] calibrated with Pointcloud gt"):
        self.obj_count = 0
        self.gt = []
        self.pred = []
        self.error = []
        self.label = label  # plot legend label

    def update(self, gt, pred, labels):
        if gt.shape[-2:] != pred.shape[-2:]:
            pred = nn.functional.interpolate(
                pred, gt.shape[-2:], mode='bilinear', align_corners=True)
        gt = gt.squeeze().cpu().numpy()
        pred = pred.squeeze().cpu().numpy()
        assert gt.shape == pred.shape, f"GT shape {gt.shape} does not match prediction shape {pred.shape}"
        for l in labels:
            obj_center3d = l['center_3d'].squeeze().cpu().numpy()
            obj_center3d = np.array([int(obj_center3d[0]), int(obj_center3d[1])])
            try:
                obj_depth = gt[obj_center3d[1], obj_center3d[0]]
            except:
                print(f"Warning: Object center {tuple(obj_center3d)} is out of bounds for the ground truth depth map of shape {gt.shape}. Skipping this object.")
                continue
            if obj_depth != 0 and l['occlusion'] == 0:
                self.gt.append(obj_depth)
                self.pred.append(pred[obj_center3d[1], obj_center3d[0]])
                self.error.append(obj_depth - pred[obj_center3d[1], obj_center3d[0]])
                self.obj_count += 1

    def get_value(self):
        if len(self.gt) == 0 or len(self.pred) == 0:
            return {"abs_rel": None, "count": 0}
        gt = np.array(self.gt)
        pred = np.array(self.pred)
        absrel = np.mean(np.abs(gt - pred) / gt)
        return {"abs_rel": absrel, "count": self.obj_count}

    def plot_scatter(self, color='r'):
        if not self.gt or not self.error:
            print("No data to plot.")
            return
        gt = np.array(self.gt)
        error = np.array(self.error)
        plt.figure()
        plt.scatter(gt, error, c=color, label=self.label)
        plt.xlabel('Ground Truth Depth [m]')
        plt.ylabel('Depth Error [m]')
        plt.title(f'Distribution of Depth Error | {len(gt)} Objects')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_error_bins(self, bin_size=10, color='r', save_dir=None):
        if not self.gt or not self.error:
            print("No data to plot.")
            return
        gt = np.array(self.gt)
        error = np.array(np.abs(self.error))

        bins = np.arange(0, gt.max() + bin_size, bin_size)
        mean_errors = []
        interval_counts = []

        for i in range(len(bins) - 1):
            start, end = bins[i], bins[i + 1]
            indices = np.where((gt >= start) & (gt < end))[0]
            if len(indices) > 0:
                mean_errors.append(np.mean(error[indices]))
                interval_counts.append(len(indices))
            else:
                mean_errors.append(np.nan)
                interval_counts.append(0)

        mean_errors = np.array(mean_errors)
        interval_counts = np.array(interval_counts)

        plt.figure()
        plt.plot(bins[:-1], mean_errors, color + '-', marker='o', label=self.label)

        bin_range = 10
        for i in range(len(bins)-1):
            if not np.isnan(mean_errors[i]):
                plt.text(bins[i], mean_errors[i] + 0.25, f'{mean_errors[i]/bin_range:.2f}', color=color, ha='left', va='bottom')
            bin_range += 10

        xtick_labels = [f'{int(bins[i])}-{int(bins[i + 1])}m\n{interval_counts[i]} obj' for i in range(len(bins)-1)]
        plt.xticks(bins[:-1], xtick_labels)
        plt.xlabel('Ground Truth Depth [m]')
        plt.ylabel('Mean Depth Error [m]')
        plt.title(f'Mean Depth Error vs. Ground Truth Depth Range | {len(gt)} Objects\nKITTI')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

        # Optional saving
        if save_dir:
            np.save(f'{save_dir}/DepthBins.npy', bins)
            np.save(f'{save_dir}/MeanErrors.npy', mean_errors)
            np.save(f'{save_dir}/IntervalCounts.npy', interval_counts)
            np.save(f'{save_dir}/Errors.npy', error)

    def __repr__(self):
        value = self.get_value()
        return f"ObjectMetrics(abs_rel={value['abs_rel']:.4f}, count={value['count']})" if value['abs_rel'] is not None else "ObjectMetrics(abs_rel=None, count=0)"



def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img


def count_parameters(model, include_all=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or include_all)


def compute_scale_and_shift(prediction, target, mask):
    """Compute scale and shift to align prediction with target using least squares.
    
    Args:
        prediction (numpy.ndarray): Predicted values
        target (numpy.ndarray): Target values  
        mask (numpy.ndarray): Valid mask
        
    Returns:
        tuple: (scale, shift) values
    """
    # Convert to flattened arrays for easier computation
    pred_masked = prediction[mask]
    target_masked = target[mask]
    
    if len(pred_masked) == 0:
        return 1.0, 0.0
    
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = np.sum(pred_masked * pred_masked)
    a_01 = np.sum(pred_masked)
    a_11 = np.sum(mask)
    
    # right hand side: b = [b_0, b_1]
    b_0 = np.sum(pred_masked * target_masked)
    b_1 = np.sum(target_masked)
    
    # solution: x = A^-1 . b
    det = a_00 * a_11 - a_01 * a_01
    
    if det <= 0:
        return 1.0, 0.0
    
    scale = (a_11 * b_0 - a_01 * b_1) / det
    shift = (-a_01 * b_0 + a_00 * b_1) / det
    
    return scale, shift


def compute_errors(gt, pred, max_depth_eval=10.0):
    """Compute metrics for 'pred' compared to 'gt'

    Here we assume the inputs are both metric, even if we are evaluating MIDAS relative depth.
    In that case, the caller should have already aligned, capped, and inverted the predictions.

    Args:
        gt (numpy.ndarray): Ground truth values
        pred (numpy.ndarray): Predicted values

        gt.shape should be equal to pred.shape

    Returns:
        dict: Dictionary containing the following metrics:
            'a1': Delta1 accuracy: Fraction of pixels that are within a scale factor of 1.25
            'a2': Delta2 accuracy: Fraction of pixels that are within a scale factor of 1.25^2
            'a3': Delta3 accuracy: Fraction of pixels that are within a scale factor of 1.25^3
            'abs_rel': Absolute relative error
            'rmse': Root mean squared error
            'log_10': Absolute log10 error
            'sq_rel': Squared relative error
            'rmse_log': Root mean squared error on the log scale
            'silog': Scale invariant log error
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)


def compute_metrics(gt, pred, interpolate=True, dataset='nyu', metric_eval=True, **kwargs):
    """Compute metrics of predicted depth maps. Applies cropping and masking as necessary or specified via arguments. Refer to compute_errors for more details on metrics.
    """
    if 'config' in kwargs:
        config = kwargs['config']
        garg_crop = config.garg_crop
        eigen_crop = config.eigen_crop
        min_depth_eval = config.min_depth_eval
        max_depth_eval = config.max_depth_eval
        # print(f"Using config settings for evaluation: garg_crop={garg_crop}, eigen_crop={eigen_crop}, min_depth_eval={min_depth_eval}, max_depth_eval={max_depth_eval}")
        # # if hasattr(config, 'metric_eval'):
        #     metric_eval = config.metric_eval

    if gt.shape[-2:] != pred.shape[-2:] and interpolate:
        pred = nn.functional.interpolate(
            pred, gt.shape[-2:], mode='bilinear', align_corners=True)

    pred = pred.squeeze().cpu().numpy()
    gt_depth = gt.squeeze().cpu().numpy()
    # If relative depth evaluation, treat pred as disparity, align, cap, and invert
    if not metric_eval:
        valid_mask = (gt_depth > 0) & (pred > 0) & np.isfinite(gt_depth) & np.isfinite(pred)
        target_disparity = np.zeros_like(gt_depth)
        target_disparity[valid_mask] = 1.0 / gt_depth[valid_mask]
        scale, shift = compute_scale_and_shift(pred, target_disparity, valid_mask)
        pred_aligned = scale * pred + shift
        depth_cap = max_depth_eval
        disparity_cap = 1.0 / depth_cap
        pred_aligned[pred_aligned < disparity_cap] = disparity_cap
        pred = 1.0 / pred_aligned

    pred[pred < min_depth_eval] = min_depth_eval
    pred[pred > max_depth_eval] = max_depth_eval
    pred[np.isinf(pred)] = max_depth_eval
    pred[np.isnan(pred)] = min_depth_eval

    valid_mask = np.logical_and(gt_depth > min_depth_eval, gt_depth < max_depth_eval)

    if garg_crop or eigen_crop:
        gt_height, gt_width = gt_depth.shape
        eval_mask = np.zeros(valid_mask.shape)

        if garg_crop:
            eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                      int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

        elif eigen_crop:
            # print("-"*10, " EIGEN CROP ", "-"*10)
            if dataset == 'kitti':
                eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                          int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
            else:
                # assert gt_depth.shape == (480, 640), "Error: Eigen crop is currently only valid for (480, 640) images"
                eval_mask[45:471, 41:601] = 1
        else:
            eval_mask = np.ones(valid_mask.shape)
    valid_mask = np.logical_and(valid_mask, eval_mask)
    return compute_errors(gt_depth[valid_mask], pred[valid_mask], max_depth_eval=max_depth_eval), pred


def compute_metrics_object(gt, pred, sample,interpolate=True, garg_crop=False, eigen_crop=True, dataset='nyu', min_depth_eval=0.1, max_depth_eval=10, **kwargs):
    """Compute metrics of predicted depth maps. Applies cropping and masking as necessary or specified via arguments. Refer to compute_errors for more details on metrics.
    """
    if 'config' in kwargs:
        config = kwargs['config']
        garg_crop = config.garg_crop
        eigen_crop = config.eigen_crop
        min_depth_eval = config.min_depth_eval
        max_depth_eval = config.max_depth_eval

    if gt.shape[-2:] != pred.shape[-2:] and interpolate:
        pred = nn.functional.interpolate(
            pred, gt.shape[-2:], mode='bilinear', align_corners=True)

    gt = gt.squeeze().cpu().numpy()
    pred = pred.squeeze().cpu().numpy()

    assert gt.shape == pred.shape, f"GT shape {gt.shape} does not match prediction shape {pred.shape}"
    pred_obj = np.zeros_like(pred)
    gt_obj = np.zeros_like(gt)
    obj_count = 0
    for l in sample['label']:
        obj_center3d = l['center_3d'].squeeze().cpu().numpy()
        obj_center3d = np.array([int(obj_center3d[0]), int(obj_center3d[1])])
        try:
            obj_depth = gt[obj_center3d[1], obj_center3d[0]]
        except:
            print(f"Warning: Object center {obj_center3d} is out of bounds for the ground truth depth map of shape {gt.shape}. Skipping this object.")
            continue
        if obj_depth != 0 and l['occlusion'] == 0:
            pred_obj[obj_center3d[1], obj_center3d[0]] = pred[obj_center3d[1], obj_center3d[0]]
            gt_obj[obj_center3d[1], obj_center3d[0]] = gt[obj_center3d[1], obj_center3d[0]]
            obj_count += 1

    gt = gt_obj
    pred = pred_obj

    # If relative depth evaluation, treat pred as disparity, align, cap, and invert
    metric_eval = kwargs.get('metric_eval', True)
    if metric_eval is False:
        valid_mask = (gt > 0) & (pred > 0) & np.isfinite(gt) & np.isfinite(pred)
        target_disparity = np.zeros_like(gt)
        target_disparity[valid_mask] = 1.0 / gt[valid_mask]
        scale, shift = compute_scale_and_shift(pred, target_disparity, valid_mask)
        pred_aligned = scale * pred + shift
        disparity_cap = 1.0 / max_depth_eval
        pred_aligned[pred_aligned < disparity_cap] = disparity_cap
        pred = 1.0 / pred_aligned

    pred[pred < min_depth_eval] = min_depth_eval
    pred[pred > max_depth_eval] = max_depth_eval
    pred[np.isinf(pred)] = max_depth_eval
    pred[np.isnan(pred)] = min_depth_eval

    gt_depth = gt
    valid_mask = np.logical_and(gt_depth > min_depth_eval, gt_depth < max_depth_eval)

    if garg_crop or eigen_crop:
        gt_height, gt_width = gt_depth.shape
        eval_mask = np.zeros(valid_mask.shape)

        if garg_crop:
            eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                      int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

        elif eigen_crop:
            # print("-"*10, " EIGEN CROP ", "-"*10)
            if dataset == 'kitti':
                eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                          int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
            else:
                # assert gt_depth.shape == (480, 640), "Error: Eigen crop is currently only valid for (480, 640) images"
                eval_mask[45:471, 41:601] = 1
        else:
            eval_mask = np.ones(valid_mask.shape)
    valid_mask = np.logical_and(valid_mask, eval_mask)
    return compute_errors(gt_depth[valid_mask], pred[valid_mask], metric_eval=metric_eval, max_depth_eval=max_depth_eval), obj_count


#################################### Model uilts ################################################


def parallelize(config, model, find_unused_parameters=True):

    if config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        model = model.cuda(config.gpu)

    config.multigpu = False
    if config.distributed:
        # Use DDP
        config.multigpu = True
        config.rank = config.rank * config.ngpus_per_node + config.gpu
        dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                                world_size=config.world_size, rank=config.rank)
        config.batch_size = int(config.batch_size / config.ngpus_per_node)
        # config.batch_size = 8
        config.workers = int(
            (config.num_workers + config.ngpus_per_node - 1) / config.ngpus_per_node)
        print("Device", config.gpu, "Rank",  config.rank, "batch size",
              config.batch_size, "Workers", config.workers)
        torch.cuda.set_device(config.gpu)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(config.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu], output_device=config.gpu,
                                                          find_unused_parameters=find_unused_parameters)

    elif config.gpu is None:
        # Use DP
        config.multigpu = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    return model


#################################################################################################


#####################################################################################################


class colors:
    '''Colors class:
    Reset all colors with colors.reset
    Two subclasses fg for foreground and bg for background.
    Use as colors.subclass.colorname.
    i.e. colors.fg.red or colors.bg.green
    Also, the generic bold, disable, underline, reverse, strikethrough,
    and invisible work with the main class
    i.e. colors.bold
    '''
    reset = '\033[0m'
    bold = '\033[01m'
    disable = '\033[02m'
    underline = '\033[04m'
    reverse = '\033[07m'
    strikethrough = '\033[09m'
    invisible = '\033[08m'

    class fg:
        black = '\033[30m'
        red = '\033[31m'
        green = '\033[32m'
        orange = '\033[33m'
        blue = '\033[34m'
        purple = '\033[35m'
        cyan = '\033[36m'
        lightgrey = '\033[37m'
        darkgrey = '\033[90m'
        lightred = '\033[91m'
        lightgreen = '\033[92m'
        yellow = '\033[93m'
        lightblue = '\033[94m'
        pink = '\033[95m'
        lightcyan = '\033[96m'

    class bg:
        black = '\033[40m'
        red = '\033[41m'
        green = '\033[42m'
        orange = '\033[43m'
        blue = '\033[44m'
        purple = '\033[45m'
        cyan = '\033[46m'
        lightgrey = '\033[47m'


def printc(text, color):
    print(f"{color}{text}{colors.reset}")

############################################

def get_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return img

def url_to_torch(url, size=(384, 384)):
    img = get_image_from_url(url)
    img = img.resize(size, Image.ANTIALIAS)
    img = torch.from_numpy(np.asarray(img)).float()
    img = img.permute(2, 0, 1)
    img.div_(255)
    return img

def pil_to_batched_tensor(img):
    return ToTensor()(img).unsqueeze(0)

def save_raw_16bit(depth, fpath="raw.png"):
    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze().cpu().numpy()
    
    assert isinstance(depth, np.ndarray), "Depth must be a torch tensor or numpy array"
    assert depth.ndim == 2, "Depth must be 2D"
    depth = depth * 256  # scale for 16-bit png
    depth = depth.astype(np.uint16)
    depth = Image.fromarray(depth)
    depth.save(fpath)
    print("Saved raw depth to", fpath)