import os
import io
from rembg import remove
from PIL import Image

def process_folder(input_root, output_root):
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            input_path = os.path.join(root, file)

            # Buat path relatif terhadap input_root
            rel_path = os.path.relpath(input_path, input_root)
            rel_dir = os.path.dirname(rel_path)

            # Buat path output
            output_dir = os.path.join(output_root, rel_dir)
            os.makedirs(output_dir, exist_ok=True)

            filename_wo_ext = os.path.splitext(file)[0]
            output_filename = f"{filename_wo_ext}_transparent.png"  # simpan sebagai PNG
            output_path = os.path.join(output_dir, output_filename)

            try:
                # 1. Hapus background
                with open(input_path, 'rb') as f:
                    input_image = f.read()
                removed = remove(input_image)

                # 2. Buka gambar hasil dengan transparansi
                img = Image.open(io.BytesIO(removed)).convert("RGBA")

                # 3. Simpan gambar dengan transparansi (PNG)
                img.save(output_path)

                print(f"Sukses: {output_path}")
            except Exception as e:
                print(f"Gagal: {input_path} => {e}")

# Folder input dan output
input_base = 'dataset_augmentasi'
output_base = 'dataset_augmentasi_cleaned_transparrent'

# Proses semua gambar di fresh dan rotten
for category in ['fresh', 'rotten']:
    process_folder(
        input_root=os.path.join(input_base, category),
        output_root=os.path.join(output_base, category)
    )
