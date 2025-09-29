import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def preprocess(input_file, output_file):
    print("="*50)
    print("🚀 Mulai Preprocessing Dataset")
    print("="*50)

    # Load dataset
    data = pd.read_csv(input_file)
    print(f"Jumlah data awal: {len(data)}")

    # 1. Cek & hapus missing values
    print("\n🔎 Tahap 1: Cek Missing Values")
    print("Jumlah missing values per kolom:\n", data.isnull().sum())
    data = data.dropna()
    print("Sisa data setelah hapus missing values:", len(data))

    # 2. Hapus duplikat
    print("\n🔎 Tahap 2: Hapus Data Duplikat")
    print("Jumlah duplikat sebelum:", data.duplicated().sum())
    data = data.drop_duplicates()
    print("Jumlah duplikat setelah:", data.duplicated().sum())
    print("Total data:", len(data))

    # 3. Encoding variabel kategorikal
    print("\n🔎 Tahap 3: Encoding Variabel Kategorikal")
    le_gender = LabelEncoder()
    data["Jenis Kelamin"] = le_gender.fit_transform(data["Jenis Kelamin"])
    print("Mapping Jenis Kelamin:", dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_))))

    le_status = LabelEncoder()
    data["Status Gizi"] = le_status.fit_transform(data["Status Gizi"])
    print("Mapping Status Gizi:", dict(zip(le_status.classes_, le_status.transform(le_status.classes_))))

    # 4. Standarisasi fitur numerik
    print("\n🔎 Tahap 4: Standarisasi Fitur Numerik")
    scaler = StandardScaler()
    data[["Umur (bulan)", "Tinggi Badan (cm)"]] = scaler.fit_transform(
        data[["Umur (bulan)", "Tinggi Badan (cm)"]]
    )
    print("Contoh hasil scaling:")
    print(data[["Umur (bulan)", "Tinggi Badan (cm)"]].head())

    # Buat folder output jika belum ada
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Simpan hasil preprocessing
    data.to_csv(output_file, index=False)
    print("\n✅ Data siap untuk pemodelan, disimpan di:", output_file)


if __name__ == "__main__":
    input_file = "namadataset_raw/data_balita.csv"
    output_file = "namadataset_preprocessing/data_balita_preprocessing.csv"
    preprocess(input_file, output_file)
