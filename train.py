import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils
from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import os
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from keras.models import Sequential


INPUT_SIZE = 64

# Hàm load dữ liệu từ thư mục
def load_data(dir_path, img_size=(100,100)):
    X = []
    y = []
    i = 0
    labels = dict()
    for path in tqdm(sorted(os.listdir(dir_path))):
        if not path.startswith('.'):
            labels[i] = path
            for file in os.listdir(dir_path + path):
                if not file.startswith('.'):
                    img = cv2.imread(dir_path + path + '/' + file)
                    X.append(img)
                    y.append(i)
            i += 1
    X = np.array(X)
    y = np.array(y)
    return X, y, labels

# Hàm crop ảnh, tìm biên
def crop_imgs(set_name, add_pixels_value=0):
    set_new = []
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # Giảm nhiễu từ ảnh đầu vào
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        # Tìm biên dựa vào ảnh nhị phân
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # Tìm các điểm biên
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        set_new.append(new_img)
    return np.array(set_new)

# Hàm xử lý hình ảnh đầu vào
def preprocess_imgs(set_name, img_size):
    set_new = []
    for img in set_name:
        img = cv2.resize(img, dsize=img_size)
        set_new.append(preprocess_input(img))
    return np.array(set_new)

# Hàm in ra hình ảnh từ tập dữ liệu sau khi thực hiện crop
def plot_samples(X, y, labels_dict, n=200):
    for index in range(len(labels_dict)):
        imgs = X[np.argwhere(y == index)][:n]
        j = 10
        i = int(n/j)
        plt.figure(figsize=(15,6))
        c = 1
        for img in imgs:
            plt.subplot(i,j,c)
            plt.imshow(img[0])
            plt.xticks([])
            plt.yticks([])
            c += 1
        plt.suptitle('Tumor: {}'.format(labels_dict[index]))
        plt.show()

# Hàm lưu hình ảnh vào thư mục
def SaveCropTrainData(listImages, listLabels):
    i = 0
    for (img, label) in zip(listImages, listLabels):
        if label == 0:
            cv2.imwrite(r'D:\NhanDangUNao\Training\TRAIN_CROP\gt\glioma' + str(i) + '.png', img)
        elif label == 1:
            cv2.imwrite(r'D:\NhanDangUNao\Training\TRAIN_CROP\mt\meningioma' + str(i) + '.png', img)
        elif label == 2:
            cv2.imwrite(r'D:\NhanDangUNao\Training\TRAIN_CROP\no\notumor' + str(i) + '.png', img)
        else:
            cv2.imwrite(r'D:\NhanDangUNao\Training\TRAIN_CROP\pt\pituitary' + str(i) + '.png', img)
        i += 1

def SaveCropValData(listImages, listLabels):
    i = 0
    for (img, label) in zip(listImages, listLabels):
        if label == 0:
            cv2.imwrite(r'D:\NhanDangUNao\Training\VAL_CROP\gt\glioma' + str(i) + '.png', img)
        elif label == 1:
            cv2.imwrite(r'D:\NhanDangUNao\Training\VAL_CROP\mt\meningioma' + str(i) + '.png', img)
        elif label == 2:
            cv2.imwrite(r'D:\NhanDangUNao\Training\VAL_CROP\no\notumor' + str(i) + '.png', img)
        else:
            cv2.imwrite(r'D:\NhanDangUNao\Training\VAL_CROP\pt\pituitary' + str(i) + '.png', img)
        i += 1


# Đặt đường dẫn đến tập dữ liệu
DIRECTORY = r'D:\NhanDangUNao\Dataset' + "\\"
CATEGORIES = ["glioma", "meningioma", "notumor", "pituitary"]
data, labels, labelname = load_data(DIRECTORY, (INPUT_SIZE,INPUT_SIZE))

# Xử lý dữ liệu
#labels = to_categorical(labels, num_classes=4)
(trainX, valX, trainY, valY) = train_test_split(data, labels,
	test_size=0.25, stratify=labels, random_state=42)

# Crop size ảnh từ bộ dữ liệu
trainX_crop = crop_imgs(set_name = trainX)
valX_crop = crop_imgs(set_name = valX)

# Lưu ảnh sau khi crop
SaveCropTrainData(trainX_crop, trainY)
SaveCropValData(valX_crop, valY)


# Sinh ảnh
from keras_preprocessing.image import ImageDataGenerator
TRAIN_DIR = r'D:\NhanDangUNao\Training\TRAIN_CROP'
VAL_DIR = r'D:\NhanDangUNao\Training\VAL_CROP'
RANDOM_SEED = 123

train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=[0.5, 1.5],
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocess_input
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    color_mode='rgb',
    target_size=(INPUT_SIZE, INPUT_SIZE),
    batch_size=32,
    class_mode='categorical',
    seed=RANDOM_SEED
)

validation_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    color_mode='rgb',
    target_size=(INPUT_SIZE, INPUT_SIZE),
    batch_size=32,
    class_mode='categorical',
    shuffle=False,
    seed=RANDOM_SEED
)


# Khởi tạo model
base_model = VGG16(input_shape=(INPUT_SIZE,INPUT_SIZE,3), include_top=False, weights='imagenet')
for layer in base_model.layers:
    layer.trainable = True
model = Sequential()
model.add(Input(shape=(INPUT_SIZE,INPUT_SIZE,3)))
model.add(base_model)
model.add(Dropout(0.75))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

# In model
model.summary()

# Dừng sớm, cắt giảm tốc độ học
stopearly = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              verbose=1, 
                              patience=20
                             )

reduce_learningrate = ReduceLROnPlateau(monitor='val_loss',
                              mode='min',
                              verbose=1,
                              patience=1,
                              min_delta=0.00001,
                              factor=0.5
                             )

# Biên dịch model
EPOCHS = 30
BS = 32

model.compile(optimizer=Adam(learning_rate=0.0001),
             loss='categorical_crossentropy',
             metrics=['categorical_accuracy'])

# Lưu lại lịch sử training
H = model.fit_generator(
	train_generator,
	steps_per_epoch=len(trainX_crop) // BS,
	validation_data=validation_generator,
	validation_steps=len(valX_crop) // BS,
	epochs=EPOCHS,
    callbacks = [reduce_learningrate]
    )
model.save("tumor.model", save_format="h5")

# In lịch sử training ra bằng đồ thị
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["categorical_accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_categorical_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch 40")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("brain_tumor.png")

# Ma trận nhầm lẫn
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
predictions = model.predict_generator(validation_generator)
y_pred = np.argmax(predictions, axis=1)
print(confusion_matrix(validation_generator.classes, y_pred))