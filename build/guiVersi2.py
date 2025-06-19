from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageTk, ImageEnhance
from tkinter import Tk, Canvas, Button, PhotoImage, filedialog, messagebox 
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd
import pickle
from tensorflow.keras.models import load_model as keras_load_model

# Asset and dataset path handling
def get_asset_path():
    base = Path(sys._MEIPASS) if hasattr(sys, '_MEIPASS') else Path(__file__).parent
    return base / 'assets' / 'frame0'

def get_data_path():
    base = Path(sys._MEIPASS) if hasattr(sys, '_MEIPASS') else Path(__file__).parent
    return base / 'assets' / 'image_collection'

def get_model_path():
    base = Path(sys._MEIPASS) if hasattr(sys, '_MEIPASS') else Path(__file__).parent
    return base / 'assets' / 'model'

ASSETS_PATH = get_asset_path()
DATASET_PATH = get_data_path()
MODEL_PATH = get_model_path()

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def relative_to_model(path: str) -> Path:
    return MODEL_PATH / Path(path)

# Global variables
image = None
brightness_level = 1.0
canvas = None
image_6 = None  # Canvas item ID for the image display
selected_model_config = None

# Image handling functions
def open_file():
    try:
        filepath = filedialog.askopenfilename(filetypes=(("Image files", "*.jpg *.png *.jpeg"), ("All files", "*.*")))
        if filepath:
            img = Image.open(filepath).resize((275, 242))
            save_image(img)
            change_image_box(img)
            jenisfile = Path(filepath).suffix.replace(".", "").upper()  # Ambil ekstensi dan ubah jadi huruf besar
            canvas.itemconfig(nama_image, text=f"Tipe Gambar: {jenisfile}")
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
    global image, image_image_6, image_6, selected_model_config
    image_image_6 = PhotoImage(file=relative_to_assets("image_6.png"))
    canvas.itemconfig(image_6, image=image_image_6)
    image = None
    selected_model_config = None
    # Reset UI texts
    texts = {
        nama_image: "Nama foto", akurasi_f: "Probabilitas(fresh)", akurasi_r: "Probabilitas(rotten)",
        kesimpulan: "Kesimpulan", nama_model: "Model", akurasi_model: "Akurasi Model",
        recall: "Recall", precision: "Precision", datasetui:"Dataset", penjelasan_model: "Penjelasan"
    }
    print(f"Reset")
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

def preprocess_image_for_model(img, input_size, model_type):
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img = img.resize(input_size)
    # img_array = img_to_array(img)
    img_array = np.array(img)
    if model_type == 'mobilenet':
        img_array = preprocess_input(img_array)
    else:
        # img_array /= 255.0
        img_array = img_array.astype("float32") / 255.0
    return img_array

def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
    ext = model_path.suffix.lower()
    if ext == ".pkl":
        with open(model_path, "rb") as file:
            model = pickle.load(file)
    elif ext == ".h5":
        model = keras_load_model(model_path)
    else:
        raise ValueError(f"Format model tidak didukung: {ext}")

    return model

def run_model(model_config):
    try:
        df = pd.read_csv(relative_to_model("model_evaluation_report.csv"))
        # Ambil baris yang cocok berdasarkan nama model
        model_name = model_config['name']
        row = df[df['Model'] == model_name]
        if row.empty:
            print(f"Model {model_name} tidak ditemukan dalam laporan evaluasi.")
            return
        accuracy = float(row['Accuracy'].values[0])
        precision_val = float(row['Precision'].values[0])
        recall_val = float(row['Recall'].values[0])

        # Update UI
        canvas.itemconfig(nama_model, text=model_config['name'])
        canvas.itemconfig(akurasi_model, text=f"Akurasi Model: {accuracy:.2f}")
        canvas.itemconfig(recall, text=f"Recall: {recall_val:.2f}")
        canvas.itemconfig(precision, text=f"Precision: {precision_val:.2f}")
        canvas.itemconfig(penjelasan_model, text=f"Penjelasan: \n{model_config['description']}")
        canvas.itemconfig(datasetui, text="Dataset: 1680 Gambar")

    except Exception as e:
        print(f"Error running model {model_config['name']}: {e}")

# Model configurations
MODEL_CONFIGS = {
    'rf': {
        'name': 'RandomForest',
        'type': 'rf',
        'description': 'Model memanfaatkan fitur \ncitra secara langsung.'
    },
    'knn': {
        'name': 'KNN',
        'type': 'knn',
        'description': 'Model KNN secara \nlangsung.'
    },
    'dt': {
        'name': 'DecisionTree',
        'type': 'dt',
        'description': 'Model Decision Tree \nsecara langsung.'
    },
    'svm': {
        'name': 'SVM',
        'type': 'svm',
        'description': 'Model SVM secara \nlangsung.'
    },
    'nb': {
        'name': 'NaiveBayes',
        'type': 'nb',
        'description': 'Model Naive Bayes \nsecara langsung.'
    },
    'lr': {
        'name': 'LogisticRegression',
        'type': 'lr',
        'description': 'Model Logistic Regression \nsecara langsung.'
    },
    'cnn': {
        'name': 'CNN',
        'type': 'cnn',
        'description': 'Model CNN berbasis \nConv2D.'
    },
    'mobilenet': {
        'name': 'MobileNetV2',
        'type': 'mobilenet',
        'description': 'Model MobileNetV2 \ndengan transfer learning.'
    },
    'rnn': {
        'name': 'RNN',
        'type': 'rnn',
        'description': 'Model RNN berbasis \nSimpleRNN.'
    },
    'lstm': {
        'name': 'LSTM',
        'type': 'lstm',
        'description': 'Model LSTM untuk data \nberurutan.'
    }
}

def run_prediction(model_config):
    try:
        model_type = model_config['type']
        model_filename = {
            "rf": "RandomForest_model.pkl",
            "knn": "KNN_model.pkl",
            "dt": "DecisionTree_model.pkl",
            "svm": "SVM_model.pkl",
            "nb": "NaiveBayes_model.pkl",
            "lr": "LogisticRegression_model.pkl",
            "cnn": "CNN_model.h5",
            "mobilenet": "MobileNetV2_model.h5",
            "rnn": "RNN_model.h5",
            "lstm": "LSTM_model.h5"
        }.get(model_type)

        if model_filename is None:
            print(f"Model type '{model_type}' tidak dikenali.")
            return
        
        model_path = relative_to_model(model_filename)

        # Prepare model
        if model_type in ['rf', 'knn', 'dt', 'svm', 'nb', 'lr']:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            img_array = preprocess_image_for_model(image, (150, 150), model_type)
            img_array = img_array.reshape(1, -1)

            prob_f = model.predict_proba(img_array)[0][0]
            prob_r = model.predict_proba(img_array)[0][1]
            label = 'Fresh' if prob_f > prob_r else 'Rotten'
        
        elif model_type in ['cnn']:
            model = load_model(model_path)
            model.compile()
            img_array = preprocess_image_for_model(image, (150, 150), model_type)
            img_array = np.expand_dims(img_array, axis=0)
            prob_f = model.predict(img_array)[0][0]
            prob_r = model.predict(img_array)[0][1]
            
            label = 'Fresh' if prob_f > prob_r else 'Rotten'

        elif model_type in ['mobilenet']:
            model = load_model(model_path)
            model.compile()
            img_array = preprocess_image_for_model(image, (224, 224), model_type)
            img_array = np.expand_dims(img_array, axis=0)
            prob_f = model.predict(img_array)[0][0]
            prob_r = model.predict(img_array)[0][1]
            
            label = 'Fresh' if prob_f > prob_r else 'Rotten'
        
        elif model_type in ['rnn', 'lstm']:
            model = load_model(model_path)
            model.compile()
            img_array = preprocess_image_for_model(image, (150, 150), model_type)
            img_array = img_array.reshape(1, 150, 150 * 3)
            prob_f = model.predict(img_array)[0][0]
            prob_r = model.predict(img_array)[0][1]
            
            label = 'Fresh' if prob_f > prob_r else 'Rotten'
        
        # Update UI
        print(f"Running model")
        canvas.itemconfig(akurasi_f, text=f"Probabilitas(fresh): {prob_f:.2f}")
        canvas.itemconfig(akurasi_r, text=f"Probabilitas(rotten): {prob_r:.2f}")
        canvas.itemconfig(kesimpulan, text=f"Kesimpulan: {label} Food")
    
    except Exception as e:
        print(f"Error running model {model_config['name']}: {e}")

def run_prediction_button_clicked():
    if image is None:
        messagebox.showwarning("Peringatan", "Silakan pilih gambar terlebih dahulu.")
        return
    if selected_model_config is None:
        messagebox.showwarning("Peringatan", "Silakan pilih model terlebih dahulu.")
        return
    run_prediction(selected_model_config)

def select_model(config):
    global selected_model_config
    selected_model_config = config
    run_model(config)

def setup_ui():
    global canvas, nama_image, kesimpulan, akurasi_f, akurasi_r, datasetui
    global precision, recall, akurasi_model, nama_model, penjelasan_model, image_6

    window = Tk()
    window.geometry("960x544")
    window.configure(bg="#FFFFFF")
    window.resizable(False, False)

    canvas = Canvas(window, bg="#FFFFFF", height=544, width=960, bd=0, highlightthickness=0, relief="ridge")
    canvas.place(x=0, y=0)

    # Simpan semua gambar di list agar tidak terhapus dari memori
    canvas.images = []

    # Load image layer
    image_positions = [
        ("image_1.png", 480.0, 271.0), ("image_2.png", 480.0, 272.0), ("image_3.png", 645.0, 192.0),
        ("image_4.png", 480.0, 22.0), ("image_5.png", 177.0, 192.0), ("image_6.png", 177.34, 191.96),
        ("image_7.png", 17.0, 18.0), ("image_8.png", 942.0, 18.0), ("image_9.png", 17.0, 526.0),
        ("image_10.png", 942.0, 526.0), ("image_11.png", 807.0, 348.0), ("image_12.png", 316.0, 348.0)
    ]

    for file, x, y in image_positions:
        img = PhotoImage(file=relative_to_assets(file))
        if file == "image_6.png":
            image_6 = canvas.create_image(x, y, image=img)
        else:
            canvas.create_image(x, y, image=img)
        canvas.images.append(img)  # Simpan referensi gambar

    # Text elements
    nama_model = canvas.create_text(674.0, 65.0, anchor="nw", text="Model", fill="#E2DED3",
                                    font=("Roboto BoldItalic", -28, "italic"))
    penjelasan_model = canvas.create_text(674.0, 202.0, anchor="nw", text="Penjelasan", fill="#E2DED3",
                                          font=("Roboto MediumItalic", -22, "italic"))
    nama_image = canvas.create_text(368.0, 150.0, anchor="nw", text="Nama foto", fill="#E2DED3",
                                    font=("Roboto MediumItalic", -22, "italic"))
    kesimpulan = canvas.create_text(368.0, 176.0, anchor="nw", text="Kesimpulan", fill="#E2DED3",
                                    font=("Roboto MediumItalic", -22, "italic"))
    akurasi_f = canvas.create_text(368.0, 98.0, anchor="nw", text="Probabilitas(fresh)", fill="#E2DED3",
                                 font=("Roboto MediumItalic", -22, "italic"))
    canvas.create_text(368.0, 65.0, anchor="nw", text="Hasil Prediksi", fill="#E2DED3",
                       font=("Roboto BoldItalic", -28, "italic"))
    akurasi_r = canvas.create_text(368.0, 124.0, anchor="nw", text="Probabilitas(rotten)", fill="#E2DED3",
                                      font=("Roboto MediumItalic", -22, "italic"))
    precision = canvas.create_text(674.0, 150.0, anchor="nw", text="Precision", fill="#E2DED3",
                                   font=("Roboto MediumItalic", -22, "italic"))
    recall = canvas.create_text(674.0, 124.0, anchor="nw", text="Recall", fill="#E2DED3",
                                font=("Roboto MediumItalic", -22, "italic"))
    datasetui = canvas.create_text(674.0, 176.0, anchor="nw", text="Dataset", fill="#E2DED3",
                       font=("Roboto", -22, "italic"))
    akurasi_model = canvas.create_text(674.0, 98.0, anchor="nw", text="Akurasi Model", fill="#E2DED3",
                                       font=("Roboto MediumItalic", -22, "italic"))

    # Button config
    button_definitions = [
        ("lstm", 492.0, 428.0, "button_1.png"), ("mobilenet", 492.0, 374.0, "button_2.png"),
        ("svm", 28.0, 428.0, "button_3.png"), ("cnn", 376.0, 374.0, "button_4.png"),
        ("lr", 260.0, 428.0, "button_5.png"), ("nb", 144.0, 428.0, "button_6.png"),
        ("brightness_down", 750.0, 428.0, "button_7.png"), ("dt", 260.0, 374.0, "button_8.png"),
        ("knn", 144.0, 374.0, "button_9.png"), ("rnn", 376.0, 428.0, "button_10.png"),
        ("run_prediction", 570.0, 486.0, "button_11.png", 360.0, 40.0), ("open_file", 28.0, 486.0, "button_12.png", 360.0, 40.0),
        ("reset_prediction", 442.0, 486.0, "button_13.png", 85.0, 40.0), ("rf", 28.0, 374.0, "button_14.png"),
        ("brightness_up", 750.0, 374.0, "button_15.png"),
    ]

    for btn_data in button_definitions:
        key = btn_data[0]
        x = btn_data[1]
        y = btn_data[2]
        img_file = btn_data[3]
        width = btn_data[4] if len(btn_data) > 4 else 116  # default width
        height = btn_data[5] if len(btn_data) > 5 else 40  # default height

        # Tentukan fungsi berdasarkan key
        if key in MODEL_CONFIGS:
            cmd = lambda c=MODEL_CONFIGS[key]: select_model(c)
        elif key == "brightness_up":
            cmd = brightness_up
        elif key == "brightness_down":
            cmd = brightness_down
        elif key == "run_prediction":
            cmd = run_prediction_button_clicked
        elif key == "open_file":
            cmd = open_file
        elif key == "reset_prediction":
            cmd = reset_prediction
        else:
            continue

        btn_img = PhotoImage(file=relative_to_assets(img_file))
        btn = Button(image=btn_img, borderwidth=0, highlightthickness=0, command=cmd, relief="flat")
        btn.place(x=x, y=y, width=width, height=height)
        btn.image = btn_img
        canvas.images.append(btn_img)

    # Draw borders
    canvas.create_rectangle(-2, 33, 2, 514, fill="#FFFFFF", outline="")
    canvas.create_rectangle(35, 542, 925, 544, fill="#FFFFFF", outline="")
    canvas.create_rectangle(35, 1, 925, 3, fill="#FFFFFF", outline="")

    window.mainloop()


if __name__ == "__main__":
    setup_ui()