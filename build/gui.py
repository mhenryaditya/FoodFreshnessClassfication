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
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import joblib
from tensorflow.keras.models import load_model

# Asset and dataset path handling
def get_asset_path():
    base = Path(sys._MEIPASS) if hasattr(sys, '_MEIPASS') else Path(__file__).parent
    return base / 'assets' / 'frame0'

def get_data_path():
    base = Path(sys._MEIPASS) if hasattr(sys, '_MEIPASS') else Path(__file__).parent
    return base / 'assets' / 'image_collection'

ASSETS_PATH = get_asset_path()
DATASET_PATH = get_data_path()

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

# Global variables
image = None
brightness_level = 1.0
canvas = None
image_6 = None  # Canvas item ID for the image display

# Image handling functions
def open_file():
    try:
        filepath = filedialog.askopenfilename(filetypes=(("Image files", "*.jpg *.png *.jpeg"), ("All files", "*.*")))
        if filepath:
            img = Image.open(filepath).resize((275, 242))
            save_image(img)
            change_image_box(img)
            canvas.itemconfig(nama_image, text=Path(filepath).name)
            canvas.itemconfig(ukuran_image, text="275 x 242")
    except Exception as e:      
        print(f"Error opening file: {e}")

def save_image(image_now):
    global image
    image = image_now

def change_image_box(my_image):
    global image_6
    current_image = ImageTk.PhotoImage(my_image)
    canvas.itemconfig(image_6, image=current_image)
    canvas.image_6 = current_image  # Prevent garbage collection

def reset_prediction():
    global image, image_image_6, image_6
    image_image_6 = PhotoImage(file=relative_to_assets("image_6.png"))
    canvas.itemconfig(image_6, image=image_image_6)
    image = None
    # Reset UI texts
    texts = {
        nama_image: "Nama foto", akurasi: "Akurasi", ukuran_image: "Ukuran",
        kesimpulan: "Kesimpulan", nama_model: "Model", akurasi_model: "Akurasi Model",
        recall: "Recall", precision: "Precision", penjelasan_model: "Penjelasan"
    }
    for text_id, value in texts.items():
        canvas.itemconfig(text_id, text=value)

def brightness_up():
    global image, brightness_level
    if image:
        brightness_level += 0.1
        update_brightness()

def brightness_down():
    global image, brightness_level
    if image:
        brightness_level = max(0.1, brightness_level - 0.1)
        update_brightness()

def update_brightness():
    enhancer = ImageEnhance.Brightness(image)
    enhanced_image = enhancer.enhance(brightness_level)
    change_image_box(enhanced_image)
    print(f"Brightness set to {brightness_level:.1f}")

# Model handling
def get_data_generators(model_type='default'):
    input_size = (224, 224) if model_type == 'mobilenet' else (150, 150)
    preprocess = preprocess_input if model_type == 'mobilenet' else lambda x: x / 255.0
    datagen = ImageDataGenerator(preprocessing_function=preprocess, validation_split=0.2)
    
    train_generator = datagen.flow_from_directory(
        DATASET_PATH, target_size=input_size, batch_size=32, class_mode='binary',
        subset='training', shuffle=True
    )
    val_generator = datagen.flow_from_directory(
        DATASET_PATH, target_size=input_size, batch_size=32, class_mode='binary',
        subset='validation', shuffle=False
    )
    return train_generator, val_generator, input_size

def preprocess_image_for_model(img, input_size, model_type):
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img = img.resize(input_size)
    img_array = img_to_array(img)
    if model_type == 'mobilenet':
        img_array = preprocess_input(img_array)
    else:
        img_array /= 255.0
    return img_array

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision_val = precision_score(y_true, y_pred)
    recall_val = recall_score(y_true, y_pred)
    return accuracy, precision_val, recall_val

def run_model(model_config):
    global image
    if image is None:
        print("Tidak ada gambar untuk diprediksi.")
        return

    try:
        model_type = model_config['type']
        train_generator, val_generator, input_size = get_data_generators(model_type)
        
        # Prepare model
        if model_type in ['rf', 'knn', 'dt', 'svm', 'nb', 'lr']:
            x_train, y_train = next(train_generator)
            x_val, y_val = next(val_generator)
            x_train_flat = x_train.reshape(x_train.shape[0], -1)
            x_val_flat = x_val.reshape(x_val.shape[0], -1)
            
            model = model_config['model']()
            model.fit(x_train_flat, y_train)
            y_pred = model.predict(x_val_flat)
            accuracy, precision_val, recall_val = evaluate_model(y_val, y_pred)
            
            img_array = preprocess_image_for_model(image, input_size, model_type)
            img_array = img_array.reshape(1, -1)
            pred = model.predict(img_array)[0]
            prob = model.predict_proba(img_array)[0][1]
            label = 'Fresh' if pred == 1 else 'Rotten'
        
        elif model_type in ['cnn', 'mobilenet']:
            model = model_config['model']()
            model.fit(train_generator, epochs=10, validation_data=val_generator, verbose=2)
            val_generator.reset()
            y_pred_prob = model.predict(val_generator)
            y_pred = (y_pred_prob > 0.5).astype(int)
            y_true = val_generator.classes
            report = classification_report(y_true, y_pred, output_dict=True)
            accuracy, precision_val, recall_val = report['accuracy'], report['1']['precision'], report['1']['recall']
            
            img_array = preprocess_image_for_model(image, input_size, model_type)
            img_array = np.expand_dims(img_array, axis=0)
            prob = model.predict(img_array)[0][0]
            pred = int(prob > 0.5)
            label = 'Fresh' if pred == 1 else 'Spoiled'
        
        elif model_type in ['rnn', 'lstm']:
            x_train, y_train = next(train_generator)
            x_val, y_val = next(val_generator)
            x_train_rnn = x_train.reshape((x_train.shape[0], 150, 150*3))
            x_val_rnn = x_val.reshape((x_val.shape[0], 150, 150*3))
            
            model = model_config['model']()
            model.fit(x_train_rnn, y_train, epochs=10, validation_data=(x_val_rnn, y_val), verbose=2)
            y_prob = model.predict(x_val_rnn).flatten()
            y_pred = (y_prob > 0.5).astype(int)
            accuracy, precision_val, recall_val = evaluate_model(y_val, y_pred)
            
            img_array = preprocess_image_for_model(image, input_size, model_type)
            img_array = img_array.reshape(1, 150, 150 * 3)
            prob = model.predict(img_array)[0][0]
            pred = int(prob > 0.5)
            label = 'Fresh' if pred == 1 else 'Spoiled'
        
        # Update UI
        canvas.itemconfig(nama_model, text=model_config['name'])
        canvas.itemconfig(akurasi_model, text=f"Akurasi Model: {accuracy:.2f}")
        canvas.itemconfig(recall, text=f"Recall: {recall_val:.2f}")
        canvas.itemconfig(precision, text=f"Precision: {precision_val:.2f}")
        canvas.itemconfig(penjelasan_model, text=model_config['description'])
        canvas.itemconfig(akurasi, text=f"Akurasi: {prob:.2f}")
        canvas.itemconfig(kesimpulan, text=f"{label} Food")
    
    except Exception as e:
        print(f"Error running model {model_config['name']}: {e}")

# Model configurations
MODEL_CONFIGS = {
    'rf': {
        'name': 'Model Random Forest',
        'type': 'rf',
        'model': lambda: RandomForestClassifier(n_estimators=100, random_state=42),
        'description': 'Model memanfaatkan fitur citra secara langsung.'
    },
    'knn': {
        'name': 'Model KNN',
        'type': 'knn',
        'model': lambda: KNeighborsClassifier(n_neighbors=5),
        'description': 'Model knn secara langsung.'
    },
    'dt': {
        'name': 'Model Decision Tree',
        'type': 'dt',
        'model': lambda: DecisionTreeClassifier(random_state=42),
        'description': 'Model DT secara langsung.'
    },
    'svm': {
        'name': 'Model SVM',
        'type': 'svm',
        'model': lambda: SVC(kernel='rbf', probability=True, random_state=42),
        'description': 'Model SVM secara langsung.'
    },
    'nb': {
        'name': 'Model NaiveBayes',
        'type': 'nb',
        'model': lambda: GaussianNB(),
        'description': 'Model NaiveBayes secara langsung.'
    },
    'lr': {
        'name': 'Model Logistic Regression',
        'type': 'lr',
        'model': lambda: LogisticRegression(max_iter=1000, random_state=42),
        'description': 'Model LR secara langsung.'
    },
    'cnn': {
    'name': 'Model CNN',
    'type': 'cnn',
    'model': lambda: (
        lambda model=Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
            MaxPooling2D(2,2),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Flatten(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dense(1, activation='sigmoid')
        ]): (
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']),
            model
        )[1]
    )(),
    'description': 'Model CNN diproses deep learning.'
},
    'rnn': {
    'name': 'Model RNN',
    'type': 'rnn',
    'model': lambda: (
        lambda model=Sequential([
            SimpleRNN(64, input_shape=(150, 150*3), activation='tanh'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ]): (
            model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy']),
            model
        )[1]
    )(),
    'description': 'Model RNN secara langsung.'
},
    'lstm': {
    'name': 'Model LSTM',
    'type': 'lstm',
    'model': lambda: (
        lambda model=Sequential([
            LSTM(64, input_shape=(150, 150*3), activation='tanh'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ]): (
            model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy']),
            model
        )[1]
    )(),
    'description': 'Model LSTM secara langsung.'
},
    'mobilenet': {
    'name': 'Model MobileNetV2',
    'type': 'mobilenet',
    'model': lambda: (
        lambda: (
            lambda base=MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3)): (
                setattr(base, 'trainable', False),
                base
            )[1]
        )()
    )(),
    'model': lambda: (
        lambda base=MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3)): (
            setattr(base, 'trainable', False),
            base
        )[1],
        lambda base: (
            model := Model(inputs=base.input, outputs=Dense(1, activation='sigmoid')(
                Dense(32, activation='relu')(
                    GlobalAveragePooling2D()(base.output)
                )
            )),
            model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy']),
            model
        )[2]
    )[1]((lambda base=MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3)): (
        setattr(base, 'trainable', False),
        base
    )[1])()),
    'description': 'Model MobileNetV2 untuk klasifikasi gambar fresh atau busuk.'
}
}

def run_prediction():
    print("Running performance analysis (placeholder)")
    # Implement performance analysis logic here if needed

# UI setup
def setup_ui():
    global canvas, nama_image, kesimpulan, akurasi, ukuran_image, precision, recall, akurasi_model, nama_model, penjelasan_model, image_6
    window = Tk()
    window.geometry("960x544")
    window.configure(bg="#FFFFFF")
    window.resizable(False, False)

    canvas = Canvas(window, bg="#FFFFFF", height=544, width=960, bd=0, highlightthickness=0, relief="ridge")
    canvas.place(x=0, y=0)

    # Load images
    images = [
        ("image_1.png", 480.0, 271.0), ("image_2.png", 480.0, 272.0), ("image_3.png", 645.0, 192.0),
        ("image_4.png", 480.0, 22.0), ("image_5.png", 177.0, 192.0), ("image_6.png", 177.335693359375, 191.9558868408203),
        ("image_7.png", 17.0, 18.0), ("image_8.png", 942.0, 18.0), ("image_9.png", 17.0, 526.0),
        ("image_10.png", 942.0, 525.9999938804057), ("image_11.png", 807.0, 348.0), ("image_12.png", 316.0, 348.0)
    ]
    for file, x, y in images:
        img = PhotoImage(file=relative_to_assets(file))
        if file == "image_6.png":
            image_6 = canvas.create_image(x, y, image=img)
        else:
            canvas.create_image(x, y, image=img)
        canvas.__setattr__(f"image_{file.split('.')[0]}", img)

    # Create text elements
    nama_model = canvas.create_text(674.0, 65.0, anchor="nw", text="Model", fill="#E2DED3", font=("Roboto BoldItalic", 28 * -1, "italic"))
    penjelasan_model = canvas.create_text(674.0, 202.0, anchor="nw", text="Penjelasan", fill="#E2DED3", font=("Roboto MediumItalic", 22 * -1, "italic"))
    nama_image = canvas.create_text(368.0, 124.0, anchor="nw", text="Nama foto", fill="#E2DED3", font=("Roboto MediumItalic", 22 * -1, "italic"))
    kesimpulan = canvas.create_text(368.0, 176.0, anchor="nw", text="Kesimpulan", fill="#E2DED3", font=("Roboto MediumItalic", 22 * -1, "italic"))
    akurasi = canvas.create_text(368.0, 98.0, anchor="nw", text="Akurasi", fill="#E2DED3", font=("Roboto MediumItalic", 22 * -1, "italic"))
    canvas.create_text(368.0, 65.0, anchor="nw", text="Hasil Prediksi", fill="#E2DED3", font=("Roboto BoldItalic", 28 * -1, "italic"))
    ukuran_image = canvas.create_text(368.0, 150.0, anchor="nw", text="Ukuran", fill="#E2DED3", font=("Roboto MediumItalic", 22 * -1, "italic"))
    precision = canvas.create_text(674.0, 150.0, anchor="nw", text="Precision", fill="#E2DED3", font=("Roboto MediumItalic", 22 * -1, "italic"))
    recall = canvas.create_text(674.0, 124.0, anchor="nw", text="Recall", fill="#E2DED3", font=("Roboto MediumItalic", 22 * -1, "italic"))
    canvas.create_text(674.0, 176.0, anchor="nw", text="Dataset", fill="#E2DED3", font=("Roboto", 22 * -1, "italic"))
    akurasi_model = canvas.create_text(674.0, 98.0, anchor="nw", text="Akurasi Model", fill="#E2DED3", font=("Roboto MediumItalic", 22 * -1, "italic"))

    # Create buttons
    buttons = [
        (MODEL_CONFIGS['lstm'], 492.0, 428.0, "button_1.png", 116.0, 40.0),
        (MODEL_CONFIGS['mobilenet'], 492.0, 374.0, "button_2.png", 116.0, 40.0),
        (MODEL_CONFIGS['svm'], 28.0, 428.0, "button_3.png", 116.0, 40.0),
        (MODEL_CONFIGS['cnn'], 376.0, 374.0, "button_4.png", 116.0, 40.0),
        (MODEL_CONFIGS['lr'], 260.0, 428.0, "button_5.png", 116.0, 40.0),
        (MODEL_CONFIGS['nb'], 144.0, 428.0, "button_6.png", 116.0, 40.0),
        (brightness_down, 750.0, 428.0, "button_7.png", 116.0, 40.0),
        (MODEL_CONFIGS['dt'], 260.0, 374.0, "button_8.png", 116.0, 40.0),
        (MODEL_CONFIGS['knn'], 144.0, 374.0, "button_9.png", 116.0, 40.0),
        (MODEL_CONFIGS['rnn'], 376.0, 428.0, "button_10.png", 116.0, 40.0),
        (run_prediction, 570.0, 486.0, "button_11.png", 360.0, 40.0),
        (open_file, 28.0, 486.0, "button_12.png", 360.0, 40.0),
        (reset_prediction, 442.0, 486.0, "button_13.png", 85.0, 40.0),
        (MODEL_CONFIGS['rf'], 28.0, 374.0, "button_14.png", 116.0, 40.0),
        (brightness_up, 750.0, 374.0, "button_15.png", 116.0, 40.0),
    ]
    for command, x, y, file, w, h in buttons:
        if isinstance(command, dict):
            cmd = lambda c=command: run_model(c)
        else:
            cmd = command
        btn_img = PhotoImage(file=relative_to_assets(file))
        btn = Button(image=btn_img, borderwidth=0, highlightthickness=0, command=cmd, relief="flat")
        btn.place(x=x, y=y, width=w, height=h)
        btn.image = btn_img  # Prevent garbage collection

    # Draw rectangles
    canvas.create_rectangle(-2, 33, 2, 514, fill="#FFFFFF", outline="")
    canvas.create_rectangle(35, 542, 925, 544, fill="#FFFFFF", outline="")
    canvas.create_rectangle(35, 1, 925, 3, fill="#FFFFFF", outline="")

    window.mainloop()

if __name__ == "__main__":
    setup_ui()