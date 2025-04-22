# Sistem Absensi dengan Pendeteksi Wajah (DETR)

Sistem absensi otomatis menggunakan teknologi pengenalan wajah berbasis web dengan Flask dan DETR (Detection Transformer).

## Fitur Utama

- Pengenalan wajah otomatis untuk absensi (check-in/check-out) menggunakan DETR
- Pendaftaran karyawan dengan pengenalan wajah
- Laporan absensi dengan filter dan ekspor ke Excel
- Dukungan untuk berbagai sumber kamera (webcam, IP camera, file video)
- Pengaturan sistem yang mudah dikonfigurasi
- Dukungan untuk komputasi GPU (CUDA) dan CPU

## Teknologi

- Python 3.8+
- Flask
- PyTorch
- DETR (Detection Transformer)
- OpenCV
- SQLite
- Bootstrap 5

## Cara Instalasi

### Instalasi dependency

```bash
# Buat virtual environment
python -m venv venv

# Aktifkan virtual environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependency
pip install -r requirements.txt
```

### Penggunaan GPU (Opsional)

Untuk performa yang lebih baik, sistem ini mendukung akselerasi GPU dengan CUDA. Pastikan Anda telah menginstal driver NVIDIA dan CUDA sebelum menggunakan mode GPU.

1. Instal CUDA Toolkit dari [situs resmi NVIDIA](https://developer.nvidia.com/cuda-downloads)
2. Verifikasi instalasi CUDA dengan perintah:
   ```bash
   nvidia-smi
   ```
3. Pilih 'CUDA' sebagai device di pengaturan aplikasi

## Cara Menjalankan

```bash
# Setelah mengaktifkan virtual environment
python app.py
```

Atau dengan menjalankan script helper:
```bash
python run.py
```

Kemudian buka browser dan akses `http://localhost:5000`

## Struktur Proyek

```
├── app.py                  # Aplikasi utama Flask
├── camera.py               # Modul pengelolaan kamera
├── database.py             # Modul database
├── detr_utils.py           # Utilitas DETR untuk pengenalan wajah
├── run.py                  # Script bantuan untuk menjalankan aplikasi
├── requirements.txt        # Daftar dependency
├── static/                 # Aset statis (CSS, JS, dll)
├── templates/              # Template HTML
│   ├── base.html           # Template dasar
│   ├── index.html          # Halaman absensi
│   ├── enrollment.html     # Halaman pendaftaran
│   ├── reports.html        # Halaman laporan
│   └── settings.html       # Halaman pengaturan
├── data/                   # Direktori penyimpanan data
│   ├── database.db         # Database SQLite
│   ├── faces/              # Penyimpanan data wajah
│   └── logs/               # Log aplikasi
└── utils/                  # Utilitas tambahan
    ├── config.py           # Pengaturan konfigurasi
    └── logger.py           # Pengaturan logging
```

## Penjelasan DETR

DETR (DEtection TRansformer) adalah model deteksi objek berbasis transformer yang dikembangkan oleh Facebook AI Research. Beberapa keunggulan DETR dibandingkan dengan metode tradisional:

1. **End-to-end training:** DETR tidak memerlukan aturan post-processing yang rumit seperti Non-Maximum Suppression (NMS).
2. **Arsitektur berbasis transformer:** Menggunakan self-attention yang efektif untuk memahami konteks global dalam gambar.
3. **Kemampuan generalisasi yang baik:** Dapat mendeteksi objek dalam berbagai posisi dan pencahayaan.
4. **Dukungan GPU:** Dapat memanfaatkan akselerasi GPU untuk inferensi yang lebih cepat.

Pada sistem ini, DETR digunakan untuk mendeteksi orang dalam frame, kemudian bagian atas tubuh (diasumsikan sebagai wajah) diekstrak dan digunakan untuk pengenalan.

## Konfigurasi

Aplikasi akan secara otomatis membuat direktori yang diperlukan saat dijalankan pertama kali. Pengaturan tambahan dapat dikonfigurasi melalui antarmuka pengaturan.

## Lisensi

Hak Cipta (c) 2023. Semua hak dilindungi undang-undang. 