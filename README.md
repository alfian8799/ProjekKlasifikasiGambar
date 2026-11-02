# 🎮 Popular Game Gameplay Classifier (Computer Vision)

Sistem klasifikasi gambar (deep learning) mutakhir untuk mendeteksi secara otomatis tangkapan layar (gameplay) dari 5 video game populer secara real-time. Proyek ini dirancang menggunakan pendekatan Transfer Learning berkinerja tinggi, divalidasi dengan resolusi input non-uniform, dan dikompresi ke berbagai format distribusi modern untuk efisiensi komputasi maksimal.

## 📝 Ringkasan Proyek

Masalah: Mengidentifikasi asal judul game secara otomatis dari sebuah gambar screenshot acak atau cuplikan video streaming dengan efisiensi tinggi.

Dataset: 5.000 Gambar gameplay populer (1.000 gambar per kelas):
- God of War
- Fortnite
- Genshin Impact
- Apex Legends
- Forza Horizon

Model Inti: MobileNetV2 (frozen weights pada ImageNet) + Custom Dense Classification Head dengan tingkat regularisasi tinggi (Dropout 40%).

## 🏆 Hasil Metrik & Evaluasi

- Akurasi Evaluasi (Test Set): 95.60%
- Loss Evaluasi (Test Set): 0.1121

Kelebihan Model:
- Non-Uniform Resolution Proof: Lulus uji verifikasi keberagaman resolusi dan rotasi sudut acak (15° hingga 270°) tanpa kegagalan pra-pemrosesan.
- Early Stopping & Learning Rate Decay: Pelatihan berhenti optimal dan stabil sebelum terjadi overfitting berkat integrasi callback dinamis.

## ⚙️ Petunjuk Instalasi

### 1. Prasyarat

Pastikan Anda telah memasang Python versi 3.8 - 3.11 di sistem Anda.

### 2. Kloning Repositori

```bash
git clone https://github.com/alfian8799/ProjekKlasifikasiGambar.git
cd ProjekKlasifikasiGambar
```

### 3. Instalasi Dependensi

Pasang semua pustaka yang diperlukan menggunakan berkas `requirements.txt` bawaan:

```bash
pip install -r requirements.txt
```

## 🚀 Cara Menjalankan Inferensi

### A. Menggunakan Model TF-Lite (Python Script)

Anda dapat langsung menguji model `.tflite` terkompresi menggunakan skrip inferensi berikut:

```python
import numpy as np
import tensorflow as tf
from PIL import Image

# 1. Muat Model TF-Lite dan Alokasikan Tensor
interpreter = tf.lite.Interpreter(model_path="tflite/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 2. Muat dan Persiapkan Gambar
img = Image.open("path_ke_gambar_gameplay.jpg").resize((150, 150))
img_array = np.array(img, dtype=np.float32) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# 3. Jalankan Inferensi
interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()

# 4. Baca Hasil Prediksi
predictions = interpreter.get_tensor(output_details[0]['index'])
class_names = ['Apex Legends', 'Fortnite', 'Forza Horizon', 'Genshin Impact', 'God of War']
predicted_class = class_names[np.argmax(predictions[0])]
confidence = float(np.max(predictions[0]) * 100)

print(f"Hasil Prediksi: {predicted_class} ({confidence:.2f}%)")
```

### B. Integrasi Sisi Klien Web (TensorFlow.js)

Untuk menjalankan inferensi langsung di browser, gunakan pustaka `@tensorflow/tfjs` pada berkas HTML/JS Anda:

```javascript
import * as tf from '@tensorflow/tfjs';

// Muat Model Web
const model = await tf.loadLayersModel('tfjs_model/model.json');

// Ubah elemen gambar HTML menjadi tensor
const imgElement = document.getElementById('my-image');
const tensor = tf.browser.fromPixels(imgElement)
  .resizeNearestNeighbor([150, 150])
  .toFloat()
  .div(tf.scalar(255.0))
  .expandDims();

// Jalankan Prediksi
const predictions = model.predict(tensor);
predictions.print();
```

## 📁 Struktur Berkas ZIP Submission

```text
submission/
├── README.md               # Ringkasan & Petunjuk Proyek
├── requirements.txt        # Dependensi Python
├── saved_model/            # Format TensorFlow SavedModel asli
│   ├── saved_model.pb
│   └── variables/
├── tfjs_model/             # Format Terkompresi TensorFlow.js untuk Web
│   ├── model.json
│   └── group1-shard*.bin
└── tflite/                 # Format Mobile/Edge Teroptimasi
    ├── model.tflite
    └── label.txt
```
