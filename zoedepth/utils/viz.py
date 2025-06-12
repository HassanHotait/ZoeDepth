from turtle import color
import rerun as rr
import numpy as np

from typing import Any, Iterable
import pyarrow as pa
import numpy.typing as npt
from rerun.datatypes import Angle, Quaternion, Rotation3D, RotationAxisAngle


import torchvision.transforms as transforms
import torch.nn as nn

import torch

class ClassBatch(rr.ComponentBatchLike):
    """A batch of class data."""

    def __init__(self: Any, cls: npt.ArrayLike) -> None:
        self.cls = cls

    def component_name(self) -> str:
        """The name of the custom component."""
        return "Class"

    def as_arrow_array(self) -> pa.Array:
        """The arrow batch representing the custom component."""
        return pa.array(self.cls, type=pa.string())
    
class OcclusionBatch(rr.ComponentBatchLike):
    """A batch of occlusion data."""

    def __init__(self: Any, occlusions: npt.ArrayLike) -> None:
        self.occlusions = occlusions

    def component_name(self) -> str:
        """The name of the custom component."""
        return "Occlusion"

    def as_arrow_array(self) -> pa.Array:
        """The arrow batch representing the custom component."""
        return pa.array(self.occlusions, type=pa.float32())
    
class TruncationBatch(rr.ComponentBatchLike):
    """A batch of truncation data."""

    def __init__(self: Any, truncations: npt.ArrayLike) -> None:
        self.truncations = truncations

    def component_name(self) -> str:
        """The name of the custom component."""
        return "Truncation"

    def as_arrow_array(self) -> pa.Array:
        """The arrow batch representing the custom component."""
        return pa.array(self.truncations, type=pa.float32())
    
class PoseBatch(rr.ComponentBatchLike):
    """A batch of pose data."""

    def __init__(self: Any, poses: npt.ArrayLike) -> None:
        self.poses: np.ndarray = np.asarray(poses, dtype=np.float32)
    def component_name(self) -> str:
        """The name of the custom component."""
        return "Pose"

    def as_arrow_array(self) -> pa.Array:
        """The arrow batch representing the custom component."""
        num_elements = self.poses.shape[-1]  # Get the number of elements per pose
        return pa.FixedSizeListArray.from_arrays(pa.array(self.poses.ravel(), type=pa.float32()), num_elements)

class CustomBoxes3D(rr.AsComponents):
    """A custom archetype that extends Rerun's builtin `Points3D` archetype with a custom component."""

    def __init__(self: Any, objs: npt.ArrayLike, color) -> None:
        self.objs = objs
        self.color = color
        # self.occlusions = occlusions
        # self.truncations = truncations

    def as_component_batches(self) -> Iterable[rr.ComponentBatchLike]:
        return (
            list(rr.Boxes3D(
                centers=[obj['pos_rr'] for obj in self.objs], # [x, y, z]
                sizes=[obj['dim'] for obj in self.objs], # [x, y, z]
                rotations=[
                    RotationAxisAngle(axis=[0, 0, 1], angle=Angle(rad=-obj['rot_y'])) # [x, y, z]
                for obj in self.objs],
                radii=0.025,
                colors=[self.color for obj in self.objs],
                # labels = [f'{pos[1]:.1f}']
            ).as_component_batches())  # The components from Points3D
            + [rr.IndicatorComponentBatch("KITTI Objects")]  # Our custom indicator
            + [ClassBatch([obj['class'] for obj in self.objs])]  # Custom confidence data
            + [OcclusionBatch([obj['occlusion'] for obj in self.objs])]  # Custom confidence data
            + [TruncationBatch([obj['truncation'] for obj in self.objs])]  # Custom confidence data
            + [PoseBatch([obj['pos'] for obj in self.objs])]  # Custom confidence data
        )
    
class CustomBoxes2D(rr.AsComponents):
    """A custom archetype that extends Rerun's builtin `Points3D` archetype with a custom component."""

    def __init__(self: Any, objs: npt.ArrayLike, img,color) -> None:
        self.objs = objs
        self.color = color
        self.img = img
        # self.occlusions = occlusions
        # self.truncations = truncations

    def as_component_batches(self) -> Iterable[rr.ComponentBatchLike]:
        # return (
        #     list(rr.Boxes2D(
        #     mins=[[xmin, ymin] for xmin, ymin, xmax, ymax in [gt_obj['bbox2d'] for gt_obj in self.objs] 
        #       if 0 <= xmin < self.img.shape[1] and 0 <= xmax < self.img.shape[1] and 0 <= ymin < self.img.shape[0] and 0 <= ymax < self.img.shape[0]],
        #     sizes=[[np.abs(xmax - xmin), np.abs(ymax - ymin)] for xmin, ymin, xmax, ymax in [gt_obj['bbox2d'] for gt_obj in self.objs] 
        #        if 0 <= xmin < self.img.shape[1] and 0 <= xmax < self.img.shape[1] and 0 <= ymin < self.img.shape[0] and 0 <= ymax < self.img.shape[0]]
        #     ).as_component_batches())  # The components from Points3D
        #     + [rr.IndicatorComponentBatch("KITTI Objects")]  # Our custom indicator
        #     + [ClassBatch([obj['class'] for obj in self.objs])]  # Custom confidence data
        #     + [OcclusionBatch([obj['occlusion'] for obj in self.objs])]  # Custom confidence data
        #     + [TruncationBatch([obj['truncation'] for obj in self.objs])]  # Custom confidence data
        #     + [PoseBatch([obj['pos'] for obj in self.objs])]  # Custom confidence data
        # )
        return (
            list(rr.Boxes2D(
            mins=[[xmin, ymin] for xmin, ymin, xmax, ymax in [gt_obj['bbox2d'] for gt_obj in self.objs]],
            sizes=[[np.abs(xmax - xmin), np.abs(ymax - ymin)] for xmin, ymin, xmax, ymax in [gt_obj['bbox2d'] for gt_obj in self.objs]]
            ).as_component_batches())  # The components from Points3D
            + [rr.IndicatorComponentBatch("KITTI Objects")]  # Our custom indicator
            + [ClassBatch([obj['class'] for obj in self.objs])]  # Custom confidence data
            + [OcclusionBatch([obj['occlusion'] for obj in self.objs])]  # Custom confidence data
            + [TruncationBatch([obj['truncation'] for obj in self.objs])]  # Custom confidence data
            + [PoseBatch([obj['pos'] for obj in self.objs])]  # Custom confidence data
        )
    


def rerun_log(frame_id, depth, pred, sample):
    def convert_labels_to_numpy(labels):
        def convert(item):
            if isinstance(item, torch.Tensor):
                arr = item.cpu().numpy()
                # Unwrap single-element arrays/lists
                if arr.shape == (1,):
                    return arr.item()
                elif arr.shape == (1, arr.shape[-1]):
                    return arr[0]
                return arr
            elif isinstance(item, list):
                # Unwrap single-element lists
                if len(item) == 1 and not isinstance(item[0], dict) and not isinstance(item[0], list):
                    return convert(item[0])
                return [convert(subitem) for subitem in item]
            elif isinstance(item, dict):
                return {k: convert(v) for k, v in item.items()}
            else:
                return item  # e.g., strings

        return [convert(label) for label in labels]

    if depth.shape[-2:] != pred.shape[-2:]:
        pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)
    rr.set_time_sequence("Frame ID: ", frame_id)
    rr.set_time_seconds('Frame ID: ', frame_id)

    img = np.array(transforms.ToPILImage()(sample['image'].squeeze().cpu()))

    # Convert depth and pred to (H, W) NumPy arrays
    depth = depth.squeeze().cpu().numpy()  # shape: (352, 1216)
    pred = pred.squeeze().cpu().numpy()    # shape: (352, 1216)

    label = convert_labels_to_numpy(sample['label'])




    rr.log("/world/gt_depth", rr.DepthImage(depth,colormap=1))
    # rr.log("/world/pred_depth", rr.Image(pred))
    rr.log("/world/pred_depth", rr.DepthImage(pred,colormap=1))
    rr.log("/world/image", rr.Image(img))

    rr.log("/world/image/box2D", CustomBoxes2D(label,img,color=[255,0,0]))
    rr.log("/world/image/center_3d",rr.Points2D(
        positions=np.array([l['center_3d'] for l in label]),
        colors=[[255, 0, 0] for _ in label],
        radii=[[5] for _ in label]))