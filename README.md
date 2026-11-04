# Document Image Classification

### Klasifikasi Gambar Dokumen Menggunakan Deep Learning dengan EfficientNet-B0 dan ConvNeXt-Tiny

Repository ini berisi dokumentasi dan implementasi proyek klasifikasi gambar dokumen menggunakan pendekatan *Deep Learning*. Proyek ini dikembangkan untuk memenuhi tugas mata kuliah melalui kompetisi **Document Image Classification** yang diselenggarakan oleh asisten dosen.

Penelitian ini berfokus pada penerapan dan optimasi arsitektur Convolutional Neural Network (CNN) untuk mengklasifikasikan gambar dokumen ke dalam 16 kategori berbeda.

---

## 1. Deskripsi Proyek

Klasifikasi gambar dokumen merupakan proses pengenalan dan pengelompokan dokumen berdasarkan karakteristik visualnya. Tantangan utama dalam permasalahan ini adalah adanya kemiripan struktur, tata letak, teks, tabel, serta elemen visual antarjenis dokumen.

Pada proyek ini, dua arsitektur *Deep Learning* digunakan dan dibandingkan, yaitu:

- **EfficientNet-B0** sebagai model baseline.
- **ConvNeXt-Tiny** sebagai model optimasi.

Eksperimen dilakukan dengan beberapa resolusi gambar, strategi pelatihan, evaluasi, dan *Test-Time Augmentation* (TTA) untuk memperoleh performa klasifikasi yang lebih baik.

## 2. Dataset

Dataset yang digunakan berasal dari kompetisi Kaggle **Document Image Classification**.

Informasi dataset:

| Keterangan | Jumlah |
|---|---:|
| Data training | 32.003 gambar |
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

## 3. Metodologi

Tahapan pengerjaan proyek meliputi:

1. **Exploratory Data Analysis (EDA)**  
   Memahami karakteristik dataset, distribusi kelas, ukuran gambar, dan karakteristik visual dokumen.

2. **Data Preparation**  
   Melakukan persiapan data, pembagian data training dan validasi menggunakan pendekatan stratified splitting, serta preprocessing gambar.

3. **Baseline Modeling**  
   Mengimplementasikan EfficientNet-B0 sebagai model pembanding awal.

4. **Model Optimization**  
   Mengimplementasikan ConvNeXt-Tiny dengan resolusi gambar 224 × 224 dan 320 × 320.

5. **Model Evaluation**  
   Mengevaluasi model menggunakan Accuracy, Precision, Recall, dan Macro F1-Score serta confusion matrix.

6. **Test-Time Augmentation (TTA)**  
   Menguji augmentasi pada tahap inferensi untuk melihat potensi peningkatan performa prediksi.

7. **Prediction and Submission**  
   Menghasilkan prediksi terhadap data testing dan menyusun file submission sesuai format kompetisi.

## 4. Arsitektur Model

### EfficientNet-B0

EfficientNet-B0 digunakan sebagai baseline untuk memperoleh gambaran awal performa klasifikasi dokumen.

### ConvNeXt-Tiny

ConvNeXt-Tiny digunakan sebagai arsitektur optimasi. Eksperimen dilakukan menggunakan dua resolusi input:

- 224 × 224 piksel
- 320 × 320 piksel

Model dengan resolusi 320 × 320 menghasilkan performa validasi yang lebih baik dibandingkan model baseline maupun eksperimen ConvNeXt 224 × 224.

## 5. Hasil Eksperimen

Berikut merupakan hasil evaluasi eksperimen yang telah dilakukan:

| Model | Validation Accuracy | Macro F1-Score |
|---|---:|---:|
| EfficientNet-B0 (224) | 83,25% | - |
| ConvNeXt-Tiny (224) | 85,8616% | 0,8581 |
| ConvNeXt-Tiny (320) | 86,4865% | 0,8640 |
| ConvNeXt-Tiny (320) + TTA | 86,5021% | 0,8642 |

### Hasil Kompetisi

| Keterangan | Skor |
|---|---:|
| Public Leaderboard Score | 0,86616 |

*Catatan: Skor leaderboard merupakan hasil yang tercatat pada saat eksperimen berlangsung dan dapat berubah mengikuti pembaruan kompetisi.*

## 6. Teknologi dan Library

Beberapa teknologi yang digunakan dalam proyek ini meliputi:

- Python
- PyTorch
- Torchvision
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Pillow (PIL)
- Kaggle Notebook
- NVIDIA Tesla T4 GPU

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

## 8. Lingkungan Eksperimen

Seluruh eksperimen dilakukan menggunakan Kaggle Notebook dengan dukungan GPU NVIDIA Tesla T4.

Dataset dan checkpoint model tidak disertakan secara langsung dalam repository ini. Untuk menjalankan notebook, pengguna perlu menyiapkan dataset kompetisi serta menyesuaikan lokasi input dan checkpoint model.

*Dokumentasi ini disusun sebagai bagian dari pengerjaan dan pengembangan proyek kompetisi Document Image Classification.*
