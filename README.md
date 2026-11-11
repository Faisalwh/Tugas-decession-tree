📂 Struktur File
Pastikan file dataset berada di dalam satu folder yang sama dengan script Python yang akan dijalankan:
- Updated Quality of Life Data.csv : File dataset utama yang berisi data kualitas hidup.
- Program_Random_Forest_Visualisasi.py : Script untuk Skenario 1 (Menggunakan 11 data awal dan visualisasi 5 pohon).
- Program_RF_Seluruh_Data_Teman.py : Script untuk Skenario 2 (Menggunakan seluruh data).

🚀 Skenario Pengujian

Skenario 1: Pengujian Data Terbatas (Sesuai Hitungan Manual)
- Dataset: Menggunakan 10 baris pertama sebagai data latih (Train) dan baris ke-11 sebagai data uji (Test).
- Parameter Model: 5 Pohon Keputusan (n_estimators=5), maksimal 1 fitur (max_features=1), kedalaman maksimal 1 (max_depth=1).
- Visualisasi: Menampilkan plot 5 pohon sederhana secara berdampingan.
- Tujuan: Memahami cara kerja dasar Random Forest secara manual/terbatas untuk mengklasifikasikan kemungkinan 'Panjang Umur'. Model pada tahap ini dibatasi pembelajarannya menyerupai model dasar (stump) untuk membuktikan konsep algoritma CART, sehingga wajar jika akurasi atau tebakan awalnya belum maksimal.

Skenario 2: Pengujian Seluruh Data (Sesuai Permintaan Tugas)
- Dataset: Menggunakan seluruh baris data dalam CSV dengan pembagian 80% untuk data latih dan 20% untuk data uji (menggunakan metode stratify agar komposisi label tetap seimbang).
- Parameter Model: 100 Pohon Keputusan (n_estimators=100) dengan parameter standar bawaan model, sehingga Random Forest bekerja dengan kapasitas optimal.
- Visualisasi: Hanya menampilkan visualisasi 10 pohon pertama untuk kebutuhan presentasi (agar tampilan jendela grafik rapi, tidak memberatkan perangkat, dan label teks tetap terbaca).
- Tujuan: Melihat performa asli dan maksimal dari model Random Forest. Dengan data latih yang jauh lebih masif (ribuan baris) dan 100 pohon pendukung, model mampu mencapai akurasi dan metrik performa yang jauh lebih tinggi dan andal.
