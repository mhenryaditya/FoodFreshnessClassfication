import os

# Folder tempat gambar berada
folder_path = 'apple'

# Huruf yang ingin ditambahkan
huruf_tambahan = 'a'  # bisa diganti sesuai kebutuhan

# Daftar ekstensi file gambar yang akan diproses
ekstensi_gambar = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

# Proses rename
for filename in os.listdir(folder_path):
    # Dapatkan ekstensi file
    _, ext = os.path.splitext(filename)

    # Periksa apakah file adalah gambar
    if ext.lower() in ekstensi_gambar:
        old_path = os.path.join(folder_path, filename)
        new_filename = huruf_tambahan + filename
        new_path = os.path.join(folder_path, new_filename)

        # Ubah nama file
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {new_filename}")
