import matplotlib
matplotlib.use("Agg")

# Import các thư viện cần thiết
from model.livenessnet import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os

# Cấu hình tham số với giá trị mặc định
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default='dataset', help="path to input dataset")
ap.add_argument("-m", "--model", type=str, default='liveness.model', help="path to trained model")
ap.add_argument("-l", "--le", type=str, default='le.pickle', help="path to label encoder")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
ap.add_argument("-hist", "--history", type=str, default="history.pickle", help="path to training history")
ap.add_argument("-r", "--repeat", type=int, default=1, help="number of times to repeat the dataset")
args = vars(ap.parse_args())

# Thiết lập hyperparameters
INIT_LR = 1e-4
BS = 4
EPOCHS = 10  # Tăng epochs và dùng early stopping

# Load và kiểm tra dataset
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
if not imagePaths:
    raise ValueError("[ERROR] Dataset path không tồn tại hoặc rỗng!")

data = []
labels = []

# Xử lý và kiểm tra ảnh
for _ in range(args["repeat"]):  # Repeat dataset loading
    for imagePath in imagePaths:
        try:
            if not os.path.exists(imagePath):
                print(f"[WARNING] Không tìm thấy ảnh: {imagePath}")
                continue
                
            label = imagePath.split(os.path.sep)[-2]
            image = cv2.imread(imagePath)
            
            if image is None:
                print(f"[WARNING] Không thể đọc ảnh: {imagePath}")
                continue
                
            image = cv2.resize(image, (32, 32))
            data.append(image)
            labels.append(label)
        except Exception as e:
            print(f"[WARNING] Lỗi xử lý ảnh {imagePath}: {str(e)}")
            continue

# Kiểm tra dữ liệu sau khi load
if len(data) == 0:
    raise ValueError("[ERROR] Không có dữ liệu hợp lệ trong dataset!")

# Chuẩn hóa dữ liệu
data = np.array(data, dtype="float32") / 255.0
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, num_classes=2)

# Kiểm tra phân bố classes
class_counts = np.sum(labels, axis=0)
print(f"[INFO] Phân bố classes: {dict(zip(le.classes_, class_counts))}")

# Chia train/test với stratify
(trainX, testX, trainY, testY) = train_test_split(data, labels, 
                                                 test_size=0.15, 
                                                 random_state=42,
                                                 stratify=labels)

# Data augmentation với các tham số phù hợp
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.1  # Thêm validation split
)

# Tính class weights
class_totals = labels.sum(axis=0)
class_weight = {0: 1.0, 1: class_totals[0]/class_totals[1]}

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=args["model"] + "_best.h5",
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# Khởi tạo và compile model
print("[INFO] compiling model...")
opt = Adam(learning_rate=INIT_LR)
model = LivenessNet.build(width=32, height=32, depth=3, classes=len(le.classes_))
model.compile(
    loss="binary_crossentropy", 
    optimizer=opt,
    metrics=["accuracy"]
)

# Train với generator
print("[INFO] training network...")
history = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=aug.flow(testX, testY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_steps=len(testX) // BS,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

# Đánh giá model
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1), 
                          predictions.argmax(axis=1),
                          target_names=le.classes_,
                          zero_division=1))

# Lưu model và các artifacts
print("[INFO] serializing network...")
model.save(args["model"] + ".h5")
model.save("liveness.keras")

with open(args["le"], "wb") as f:
    f.write(pickle.dumps(le))

with open(args["history"], "wb") as f:
    pickle.dump(history.history, f)