# SMARTBIN: Waste Detection Using YOLOv8 and Streamlit

SMARTBIN adalah aplikasi deteksi sampah berbasis YOLOv8 yang dapat mengklasifikasikan sampah menjadi tiga kelas:

* Organik
* Anorganik
* B3

Aplikasi ini menggunakan model `best.pt` hasil training YOLOv8 dan dijalankan melalui website Streamlit.

---

## 1. Struktur Project

```bash
.
├── .streamlit/
├── kode training/
├── app.py
├── best.pt
├── requirements.txt
├── runtime.txt
└── README.md
```

Keterangan:

* `kode training/` berisi kode training YOLOv8.
* `app.py` adalah file utama aplikasi Streamlit.
* `best.pt` adalah model hasil training.
* `requirements.txt` berisi library yang dibutuhkan.
* `runtime.txt` berisi versi Python untuk deployment.

---

## 2. Persiapan Dataset

Dataset menggunakan format YOLO dengan struktur:

```bash
dataset_split/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

Isi `data.yaml`:

```yaml
train: /kaggle/working/dataset_split/train/images
val: /kaggle/working/dataset_split/valid/images
test: /kaggle/working/dataset_split/test/images

nc: 3
names:
  - Organik
  - Anorganik
  - B3
```

Pastikan label hanya berisi class:

```text
0 = Organik
1 = Anorganik
2 = B3
```

---

## 3. Cek Label Dataset

Sebelum training, cek apakah ada class yang salah.

```python
import glob
from collections import Counter

label_files = glob.glob("/kaggle/working/dataset_split/**/*.txt", recursive=True)

counter = Counter()
bad_files = []

for file in label_files:
    with open(file) as f:
        for line in f:
            if line.strip():
                cls = int(line.split()[0])
                counter[cls] += 1

                if cls not in [0, 1, 2]:
                    bad_files.append(file)

print("Distribusi class:", counter)
print("File bermasalah:", len(set(bad_files)))
```

Jika masih ada class selain `0`, `1`, dan `2`, label tersebut harus dihapus atau diperbaiki terlebih dahulu.

---

## 4. Training Model YOLOv8

Install Ultralytics:

```bash
pip install ultralytics
```

Training model dari YOLOv8 Nano:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="/kaggle/working/dataset_split/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    patience=10,
    name="trash_yolov8n"
)
```

Setelah training selesai, model terbaik akan tersimpan di:

```bash
/kaggle/working/runs/detect/trash_yolov8n/weights/best.pt
```

---

## 5. Fine-Tuning Model Lama

Jika sudah memiliki model sebelumnya, gunakan `best.pt` lama untuk fine-tuning dataset baru.

```python
from ultralytics import YOLO

model = YOLO("/kaggle/input/trash-yolo-best-model/best.pt")

model.train(
    data="/kaggle/working/dataset_split/data.yaml",
    epochs=40,
    imgsz=640,
    batch=16,
    patience=10,
    name="trash_finetune"
)
```

Hasil model baru akan tersimpan di:

```bash
/kaggle/working/runs/detect/trash_finetune/weights/best.pt
```

Download model dari Kaggle:

```python
from IPython.display import FileLink

FileLink("/kaggle/working/runs/detect/trash_finetune/weights/best.pt")
```

Setelah di-download, pindahkan file `best.pt` ke folder project yang sama dengan `app.py`.

---

## 6. Menyiapkan Aplikasi Streamlit

Pastikan file project sudah seperti ini:

```bash
.
├── app.py
├── best.pt
├── requirements.txt
└── runtime.txt
```

Isi `requirements.txt`:

```txt
streamlit
ultralytics
opencv-python-headless
pillow
numpy
```

Isi `runtime.txt`:

```txt
python-3.11
```

---

## 7. Menjalankan Aplikasi Secara Lokal

Install library:

```bash
pip install -r requirements.txt
```

Jalankan Streamlit:

```bash
streamlit run app.py
```

Buka browser pada alamat:

```bash
http://localhost:8501
```

---

## 8. Upload ke GitHub

Tambahkan semua file ke repository:

```bash
git add .
git commit -m "Upload SMARTBIN project"
git push origin main
```

Pastikan file berikut sudah ada di GitHub:

```bash
app.py
best.pt
requirements.txt
runtime.txt
.streamlit/
```

---

## 9. Deploy ke Streamlit Cloud

Langkah deployment:

1. Buka Streamlit Cloud.
2. Klik **New App**.
3. Pilih repository GitHub.
4. Pilih branch `main`.
5. Isi main file path:

```text
app.py
```

6. Klik **Deploy**.
7. Tunggu proses instalasi selesai.
8. Aplikasi siap digunakan melalui link Streamlit Cloud.

---

## 10. Cara Kerja Aplikasi

Alur sistem:

```text
User upload gambar / menggunakan webcam
        ↓
Gambar diproses oleh model YOLOv8
        ↓
Model mendeteksi objek sampah
        ↓
Aplikasi menampilkan bounding box
        ↓
Aplikasi menampilkan jenis sampah dan akurasi
        ↓
Aplikasi memberikan rekomendasi pembuangan
```

---

## Troubleshooting

### best.pt tidak ditemukan

Pastikan `best.pt` berada satu folder dengan `app.py`.

```bash
.
├── app.py
└── best.pt
```

### Streamlit tidak dikenali

Install Streamlit:

```bash
pip install streamlit
```

### Error OpenCV saat deploy

Gunakan:

```txt
opencv-python-headless
```

Jangan gunakan:

```txt
opencv-python
```

### Dataset Kaggle tidak bisa diedit

Folder `/kaggle/input` bersifat read-only. Copy dataset ke `/kaggle/working` terlebih dahulu sebelum cleaning atau training.
