import os
import numpy as np
import matplotlib.pyplot as plt

def parse_prescan_label_file(label_list, idx,K=None):
    """parse label text file into a list of numpy arrays, one for each frame"""
    print(idx)
    f = open(f"C:\\Users\\Hasan\\OneDrive\\Desktop\\Projects\\TestKitti\\postProcessedData\\data_5\\labels\\{str(idx).zfill(6)}.txt")

    line_list = []
    for line in f:
        line = line.split()
        line_list.append(line)

    # each line corresponds to one detection
    det_dict_list = []
    for line in line_list:
        # det_dict holds info on one detection
        det_dict = {}
        det_dict["track_id"] = int(line[0])
        det_dict["prescan_class"] = str(line[1])
        det_dict["class"] = str(line[2])
        if det_dict["class"] == "DontCare":
            continue
        det_dict["truncation"] = float(line[3])
        det_dict["occlusion"] = int(line[4])
        det_dict["alpha"] = float(
            line[5]
        )  # obs angle relative to straight in front of camera
        x_min = int(round(float(line[6])))
        y_min = int(round(float(line[7])))
        x_max = int(round(float(line[8])))
        y_max = int(round(float(line[9])))
        det_dict["bbox2d"] = np.array([x_min, y_min, x_max, y_max])
        length = float(line[12])
        width = float(line[11])
        height = float(line[10])
        det_dict["dim"] = np.array([length, width, height])
        x_pos = float(line[13])
        y_pos = float(line[14])
        z_pos = float(line[15])
        det_dict["pos"] = np.array([x_pos, y_pos, z_pos])
        if K is not None:
            det_dict['center_3d'] = K @ np.array([x_pos, y_pos, z_pos]).reshape(3,1)
            det_dict['center_3d'] = (det_dict['center_3d'][:2] / det_dict['center_3d'][2]).reshape(2,)  # (x, y) in pixels
        det_dict["pos_rr"] = np.array([x_pos, z_pos, -y_pos])
        det_dict["rot_y"] = float(line[16])
        det_dict_list.append(det_dict)

    return det_dict_list

def plot_depth_distribution_from_test_file(test_file_path, labels_path):
    depths = []

    with open(test_file_path, 'r') as f:
        lines = f.readlines()

    # Extract frame indices from each line
    frame_indices = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 1:
            # Extract the filename, then extract the frame index from it
            img_path = parts[0]
            frame_idx = int(os.path.basename(img_path).split('.')[0])  # e.g., '000091' -> 91
            frame_indices.append(frame_idx)

    for idx in frame_indices:
        dets = parse_prescan_label_file(label_list=None, idx=idx)

        for det in dets:
            if det["occlusion"] == 0:
                depth = det["pos"][2]  # Z-position = depth
                depths.append(depth)

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(depths, bins=50, edgecolor='black')
    plt.title("Depth Distribution (occlusion = 0) for Test File Frames")
    plt.xlabel("Depth (meters)")
    plt.ylabel("Number of Objects")
    plt.grid(True)
    plt.show()

# Example usage
test_file_path = r"C:\Users\Hasan\OneDrive\Desktop\Projects\TestKitti\postProcessedData\data_5\train_test_inputs\test.txt"
labels_path = r"C:\Users\Hasan\OneDrive\Desktop\Projects\TestKitti\postProcessedData\data_5\labels"
plot_depth_distribution_from_test_file(test_file_path, labels_path)
