import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def preprocess(input_file, output_file):
    print("="*50)
    print("🚀 Mulai Preprocessing Dataset Balita")
    print("="*50)

    # Load dataset
    data = pd.read_csv(input_file)

    # Cek missing values
    data = data.dropna()

    # Hapus duplikat
    data = data.drop_duplicates()

    # Encoding variabel kategorikal
    le_gender = LabelEncoder()
    data["Jenis Kelamin"] = le_gender.fit_transform(data["Jenis Kelamin"])

    le_status = LabelEncoder()
    data["Status Gizi"] = le_status.fit_transform(data["Status Gizi"])

    # Standarisasi fitur numerik
    scaler = StandardScaler()
    data[["Umur (bulan)", "Tinggi Badan (cm)"]] = scaler.fit_transform(
        data[["Umur (bulan)", "Tinggi Badan (cm)"]]
    )

    # Pastikan folder tujuan ada
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Simpan hasil
    data.to_csv(output_file, index=False)
    print(f"✅ Data berhasil diproses dan disimpan ke {output_file}")

if __name__ == "__main__":
    input_file = "namadataset_raw/data_balita.csv"
    output_file = "namadataset_preprocessing/data_balita_ready.csv"
    preprocess(input_file, output_file)
