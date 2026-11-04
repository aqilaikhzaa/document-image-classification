# Document Image Classification

### Klasifikasi Gambar Dokumen Menggunakan Deep Learning dengan EfficientNet-B0 dan ConvNeXt-Tiny

Repository ini menyajikan implementasi klasifikasi gambar dokumen menggunakan pendekatan *Deep Learning* dengan arsitektur EfficientNet-B0 dan ConvNeXt-Tiny. Tahapan yang dilakukan mencakup eksplorasi data, preprocessing, pelatihan model, optimasi, evaluasi, serta prediksi untuk mengklasifikasikan gambar dokumen ke dalam 16 kategori.

Studi ini berfokus pada analisis dan perbandingan performa kedua arsitektur, serta mengevaluasi pengaruh peningkatan resolusi input terhadap kemampuan model dalam mengenali karakteristik visual dokumen, seperti struktur, tata letak, teks, dan elemen grafis.

---

## 1. Deskripsi Proyek

Klasifikasi gambar dokumen merupakan proses pengenalan dan pengelompokan dokumen berdasarkan karakteristik visualnya. Tantangan utama dalam permasalahan ini adalah adanya kemiripan struktur, tata letak, teks, tabel, serta elemen visual antarjenis dokumen.

Pada proyek ini, tiga eksperimen utama dilakukan:

- **EfficientNet-B0 (224 × 224)** sebagai model baseline.
- **ConvNeXt-Tiny (224 × 224)** sebagai model pengembangan.
- **ConvNeXt-Tiny (320 × 320)** sebagai model optimasi resolusi tinggi.

Eksperimen dilakukan melalui tahapan eksplorasi data, preprocessing, pelatihan model, fine-tuning, evaluasi, dan prediksi terhadap data testing.

## 2. Dataset

Dataset yang digunakan berasal dari kompetisi Kaggle **Document Image Classification**.

Informasi dataset:

| Keterangan | Jumlah |
|---|---:|
| Data training awal | 32.003 gambar |
| Data training valid setelah pemeriksaan | 32.002 gambar |
| Data training setelah pembagian | 25.601 gambar |
| Data validation | 6.401 gambar |
| Data testing | 7.994 gambar |
| Jumlah kelas | 16 |
| Format gambar | TIFF |
| Ukuran tinggi gambar | 1.000 piksel |

### Kategori Dokumen

Model dikembangkan untuk mengenali 16 kategori dokumen berikut:

1. Advertisement
2. Budget
3. Email
4. File Folder
5. Form
6. Handwritten
7. Invoice
8. Letter
9. Memo
10. News Article
11. Presentation
12. Questionnaire
13. Resume
14. Scientific Publication
15. Scientific Report
16. Specification

### Pemeriksaan Integritas Dataset

Pemeriksaan integritas dilakukan terhadap seluruh 32.003 gambar training. Hasil pemeriksaan menemukan satu gambar yang mengalami kerusakan, sehingga gambar tersebut dikeluarkan dari proses pelatihan dan evaluasi.

Dataset yang valid kemudian dibagi menggunakan metode *Stratified Train-Test Split* dengan rasio 80:20 dan random state 42.

## 3. Metodologi

Tahapan pengerjaan proyek meliputi:

1. **Exploratory Data Analysis (EDA)**  
   Memahami karakteristik dataset, distribusi kelas, ukuran gambar, dan karakteristik visual dokumen.

2. **Data Preparation**  
   Melakukan pemeriksaan integritas gambar, pembersihan data, dan pembagian data training serta validation secara terstratifikasi.

3. **Image Preprocessing**  
   Mengubah ukuran gambar menggunakan metode resize dengan padding putih untuk mempertahankan rasio aspek gambar. Selanjutnya dilakukan normalisasi menggunakan nilai mean dan standard deviation ImageNet.

4. **Baseline Modeling**  
   Mengimplementasikan EfficientNet-B0 dengan resolusi input 224 × 224 sebagai model baseline.

5. **Model Development**  
   Mengimplementasikan ConvNeXt-Tiny dengan resolusi input 224 × 224 sebagai model pengembangan.

6. **Model Optimization**  
   Melakukan fine-tuning ConvNeXt-Tiny menggunakan resolusi input 320 × 320 untuk meningkatkan kemampuan model dalam mengenali detail visual dokumen.

7. **Model Evaluation**  
   Mengevaluasi performa model menggunakan Accuracy, Precision, Recall, Macro F1-Score, dan Confusion Matrix.

8. **Prediction and Submission**  
   Menggunakan checkpoint ConvNeXt-Tiny 320 × 320 untuk menghasilkan prediksi pada data testing dan menyusun file submission sesuai format kompetisi.

## 4. Arsitektur Model

### 4.1 EfficientNet-B0

EfficientNet-B0 digunakan sebagai model baseline untuk memperoleh gambaran awal performa klasifikasi dokumen.

Model menggunakan bobot pretrained ImageNet yang kemudian disesuaikan pada bagian classifier agar menghasilkan 16 kelas keluaran.

Resolusi input yang digunakan adalah 224 × 224 piksel.

### 4.2 ConvNeXt-Tiny (224 × 224)

ConvNeXt-Tiny digunakan sebagai model pengembangan untuk membandingkan performanya dengan EfficientNet-B0.

Eksperimen dilakukan dengan resolusi input 224 × 224 piksel. Model dilatih menggunakan optimizer AdamW, mixed precision, dan mekanisme early stopping.

### 4.3 ConvNeXt-Tiny (320 × 320)

Pada tahap optimasi, resolusi input ditingkatkan menjadi 320 × 320 piksel.

Fine-tuning dilakukan menggunakan checkpoint ConvNeXt-Tiny resolusi 224 × 224 sebagai model awal. Eksperimen menggunakan batch size 8, optimizer AdamW, learning rate 1e-5, dan maksimal 5 epoch.

Model terbaik diperoleh pada epoch ke-5 dengan validation accuracy sebesar 86,2678%.

## 5. Hasil Eksperimen

Berdasarkan hasil evaluasi yang tersimpan dalam `results/model_comparison.csv`, diperoleh perbandingan performa berikut.

| Model | Resolusi | Validation Accuracy |
|---|---:|---:|
| EfficientNet-B0 | 224 × 224 | 83,25% |
| ConvNeXt-Tiny | 224 × 224 | 85,8616% |
| ConvNeXt-Tiny | 320 × 320 | **86,2678%** |

### 5.1 Evaluasi Model Terbaik

Berdasarkan hasil evaluasi akhir pada data validation, ConvNeXt-Tiny dengan resolusi 320 × 320 menghasilkan performa sebagai berikut:

| Metrik | Hasil |
|---|---:|
| Validation Accuracy | 86,2678% |
| Macro Precision | 0,8637 |
| Macro Recall | 0,8622 |
| Macro F1-Score | 0,8624 |
| Weighted F1-Score | 0,8628 |

Model ConvNeXt-Tiny 320 × 320 menunjukkan peningkatan validation accuracy sebesar 0,4062 percentage points dibandingkan ConvNeXt-Tiny 224 × 224 berdasarkan nilai pada file perbandingan model.

Hasil evaluasi per kelas disimpan dalam file `classification_report_320.csv`, sedangkan visualisasi kesalahan klasifikasi disajikan melalui `confusion_matrix_320.png`.

### 5.2 Riwayat Pelatihan ConvNeXt-Tiny 320

Berikut merupakan ringkasan hasil pelatihan model pada setiap epoch.

| Epoch | Training Accuracy | Validation Accuracy |
|---:|---:|---:|
| 1 | 90,12% | 85,05% |
| 2 | 93,14% | 85,53% |
| 3 | 94,72% | 85,42% |
| 4 | 96,02% | 85,96% |
| 5 | 96,57% | 86,27% |

Hasil tersebut memperlihatkan peningkatan performa validation accuracy hingga epoch ke-5. Checkpoint terbaik kemudian disimpan untuk digunakan pada proses inferensi data testing.

## 6. Prediction and Submission

Tahap prediksi dilakukan menggunakan model ConvNeXt-Tiny dengan resolusi 320 × 320 dan checkpoint terbaik hasil eksperimen Notebook 02B.

Proses inferensi dilakukan terhadap seluruh 7.994 gambar testing.

File submission disusun dengan dua kolom:

- `id`: identitas setiap gambar testing.
- `label`: hasil prediksi kelas dokumen.

Validasi file submission memastikan bahwa jumlah baris sesuai dengan data testing, urutan ID tetap sama dengan sample submission, tidak terdapat nilai kosong, dan label berada pada rentang 0–15.

File hasil prediksi disimpan sebagai:

- `submission.csv`
- `submission_convnext_320.csv`

## 7. Struktur Repository

```text
document-image-classification/
│
├── notebooks/
│   ├── 01_EDA_Document_Classification.ipynb
│   ├── 02A_Modelling_Document_Classification.ipynb
│   ├── 02B_Optimasi_Document_Classification.ipynb
│   └── 03_Submission_Document_Classification.ipynb
│
├── results/
│   ├── classification_report_320.csv
│   ├── confusion_matrix_320.png
│   ├── model_comparison.csv
│   ├── submission.csv
│   └── training_history_320.csv
│
├── .gitignore
├── README.md
└── requirements.txt
```

### Deskripsi Notebook

| Notebook | Fungsi |
|---|---|
| 01 EDA | Eksplorasi dan pemahaman dataset |
| 02A Modelling | Implementasi EfficientNet-B0 |
| 02B Optimasi | Pengembangan ConvNeXt-Tiny dan optimasi resolusi |
| 03 Submission | Inferensi dan pembuatan file submission |

## 8. Teknologi dan Library

Teknologi yang digunakan dalam proyek ini meliputi:

- Python
- PyTorch
- Torchvision
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Pillow (PIL)
- tqdm
- Kaggle Notebook
- NVIDIA Tesla T4 GPU

## 9. Lingkungan Eksperimen

Seluruh eksperimen dilakukan menggunakan Kaggle Notebook dengan dukungan GPU NVIDIA Tesla T4.

Dataset dan checkpoint model tidak disertakan secara langsung dalam repository ini. Untuk menjalankan notebook, pengguna perlu menyiapkan dataset kompetisi melalui Kaggle serta menyesuaikan lokasi input dan checkpoint sesuai konfigurasi masing-masing notebook.

## 10. Kesimpulan

Berdasarkan eksperimen yang telah dilakukan, ConvNeXt-Tiny dengan resolusi input 320 × 320 memberikan validation accuracy tertinggi sebesar 86,2678% dengan Macro F1-Score sebesar 0,8624.

Hasil tersebut menunjukkan bahwa peningkatan resolusi input melalui fine-tuning dapat memberikan peningkatan performa klasifikasi dibandingkan eksperimen ConvNeXt-Tiny pada resolusi 224 × 224.

Model ConvNeXt-Tiny 320 × 320 selanjutnya digunakan sebagai model final untuk menghasilkan prediksi pada data testing sebanyak 7.994 gambar.

---

*Repository ini disusun sebagai dokumentasi implementasi dan eksperimen proyek Document Image Classification menggunakan pendekatan Deep Learning.*
