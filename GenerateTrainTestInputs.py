import os
import random

# Define directories
img_directory = r"C:\Users\Hasan\OneDrive\Desktop\Projects\ZoeDepth\postProcessedData\data_5\images"
depth_directory = r"C:\Users\Hasan\OneDrive\Desktop\Projects\ZoeDepth\postProcessedData\data_5\depthMap"
output_dir = r"C:\Users\Hasan\OneDrive\Desktop\Projects\ZoeDepth\postProcessedData\data_5\train_test_inputs"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Set train split ratio
train_ratio = 0.8  # 80% training, 20% testing

# Get and sort file lists to ensure matching order
img_files = sorted(os.listdir(img_directory))
depth_files = sorted(os.listdir(depth_directory))

# Combine image and depth file paths
pairs = list(zip(img_files, depth_files))

# Shuffle the pairs
random.shuffle(pairs)

# Split into train and test
split_index = int(len(pairs) * train_ratio)
train_pairs = pairs[:split_index]
test_pairs = pairs[split_index:]

# Function to write pairs to file
def write_pairs(pairs, filename):
    with open(filename, "w") as f:
        for img_file, depth_file in pairs:
            full_path_img = os.path.join(img_directory, img_file)
            full_path_depth = os.path.join(depth_directory, depth_file)
            f.write(f"{full_path_img} {full_path_depth} 733\n")

# Write to train.txt and test.txt
train_file = os.path.join(output_dir, "train.txt")
test_file = os.path.join(output_dir, "test.txt")

write_pairs(train_pairs, train_file)
write_pairs(test_pairs, test_file)

print(f"Train and test file paths have been written to:\n{train_file}\n{test_file}")
