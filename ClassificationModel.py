import os
import csv
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageTk, ImageEnhance
from tkinter import Tk, Canvas, Button, PhotoImage, filedialog
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, SimpleRNN, LSTM, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle

# === Fungsi Simpan Metrik Evaluasi ===
def save_metrics(model_name, y_true, y_pred, output_file="model_evaluation_report.csv"):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    row = [model_name, accuracy, precision, recall]
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

# === Header File Evaluasi ===
with open("model_evaluation_report.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Model", "Accuracy", "Precision", "Recall"])

# === 1. Load Gambar ===
IMG_SIZE = 150
data = []
labels = []
base_dir = 'assets/image_collection'

for label in ['fresh', 'rotten']:
    label_dir = os.path.join(base_dir, label)
    for subfolder in os.listdir(label_dir):
        subfolder_path = os.path.join(label_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        for img_name in os.listdir(subfolder_path):
            img_path = os.path.join(subfolder_path, img_name)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                data.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

data = np.array(data)
labels = np.array(labels)
print(f"Loaded {len(data)} images.")

# === 2. Encode Label ===
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# === 3. Split Data ===
X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

# === 4. ===================== Model Klasik ===================== ===
X_train_norm = X_train.astype("float32") / 255.0
X_test_norm = X_test.astype("float32") / 255.0
X_train_flat = X_train_norm.reshape(len(X_train_norm), -1)
X_test_flat = X_test_norm.reshape(len(X_test_norm), -1)

models_classic = {
    "RandomForest": RandomForestClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "SVM": SVC(probability=True),
    "NaiveBayes": GaussianNB(),
    "KNN": KNeighborsClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=1000)
}

for name, model in models_classic.items():
    model.fit(X_train_flat, y_train)
    preds = model.predict(X_test_flat)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.2f}")
    save_metrics(name, y_test, preds)
    with open(f"{name}_model.pkl", "wb") as f:
        pickle.dump(model, f)

# === 5. ===================== CNN ===================== ===
X_train_dl = X_train.astype("float32") / 255.0
X_test_dl = X_test.astype("float32") / 255.0
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train_dl, y_train_cat, epochs=10, validation_data=(X_test_dl, y_test_cat))
cnn_model.save("CNN_model.h5")
cnn_preds = np.argmax(cnn_model.predict(X_test_dl), axis=1)
save_metrics("CNN", y_test, cnn_preds)

# === 6. ===================== RNN ===================== ===
X_train_rnn = X_train_dl.reshape(-1, IMG_SIZE, IMG_SIZE * 3)
X_test_rnn = X_test_dl.reshape(-1, IMG_SIZE, IMG_SIZE * 3)

rnn_model = Sequential([
    SimpleRNN(64, input_shape=(IMG_SIZE, IMG_SIZE * 3), return_sequences=False),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])
rnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
rnn_model.fit(X_train_rnn, y_train_cat, epochs=10, validation_data=(X_test_rnn, y_test_cat))
rnn_model.save("RNN_model.h5")
rnn_preds = np.argmax(rnn_model.predict(X_test_rnn), axis=1)
save_metrics("RNN", y_test, rnn_preds)

# === 7. ===================== LSTM ===================== ===
lstm_model = Sequential([
    LSTM(64, input_shape=(IMG_SIZE, IMG_SIZE * 3), return_sequences=False),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])
lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train_rnn, y_train_cat, epochs=10, validation_data=(X_test_rnn, y_test_cat))
lstm_model.save("LSTM_model.h5")
lstm_preds = np.argmax(lstm_model.predict(X_test_rnn), axis=1)
save_metrics("LSTM", y_test, lstm_preds)

# === 8. ===================== MobileNetV2 ===================== ===
X_mobilenet = np.array([cv2.resize(img, (224, 224)) for img in data])
X_mobilenet = preprocess_input(X_mobilenet)
y_mobilenet = to_categorical(labels_encoded)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_mobilenet, y_mobilenet, test_size=0.2, random_state=42)

base_model = MobileNetV2(include_top=False, input_shape=(224, 224, 3), weights="imagenet")
base_model.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)
mobilenet_model = Model(inputs=base_model.input, outputs=predictions)

mobilenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
mobilenet_model.fit(X_train_m, y_train_m, epochs=10, validation_data=(X_test_m, y_test_m))
mobilenet_model.save("MobileNetV2_model.h5")
mobilenet_preds = np.argmax(mobilenet_model.predict(X_test_m), axis=1)
y_test_m_labels = np.argmax(y_test_m, axis=1)
save_metrics("MobileNetV2", y_test_m_labels, mobilenet_preds)
