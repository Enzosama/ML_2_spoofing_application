import os
import shutil
import random
import zipfile

# Định nghĩa đường dẫn
base_dir = "/teamspace/studios/this_studio/face_anti_spoofing/dataset"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
zip_path = os.path.join(base_dir, "dataset_split.zip")

# Tạo thư mục train và test nếu chưa có
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Lấy danh sách các thư mục con (giả định mỗi thư mục con là một nhãn)
classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d not in ["train", "test"]]

# Chia dữ liệu
for cls in classes:
    class_path = os.path.join(base_dir, cls)
    images = os.listdir(class_path)
    random.shuffle(images)
    
    split_idx = int(0.8 * len(images))
    train_images, test_images = images[:split_idx], images[split_idx:]
    
    # Tạo thư mục đích
    train_class_path = os.path.join(train_dir, cls)
    test_class_path = os.path.join(test_dir, cls)
    os.makedirs(train_class_path, exist_ok=True)
    os.makedirs(test_class_path, exist_ok=True)
    
    # Di chuyển file
    for img in train_images:
        shutil.move(os.path.join(class_path, img), os.path.join(train_class_path, img))
    
    for img in test_images:
        shutil.move(os.path.join(class_path, img), os.path.join(test_class_path, img))

# Nén thành file ZIP
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(base_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, base_dir)
            zipf.write(file_path, arcname)

print(f"Dataset đã được chia và nén thành {zip_path}")