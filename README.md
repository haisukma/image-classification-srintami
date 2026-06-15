# Image Classification - Srintami

Proyek ini merupakan aplikasi klasifikasi gambar menggunakan TensorFlow.

## Persiapan

Pastikan Python sudah terinstall di komputer.

Cek versi Python:

```bash
python --version
```

atau

```bash
python3 --version
```

## Clone Repository

```bash
git clone https://github.com/haisukma/image-classification-srintami.git
cd image-classification-srintami
```

## Membuat Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

Jika berhasil, akan muncul `(venv)` di depan terminal.

## Install Dependencies

Install seluruh library yang dibutuhkan dari file `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Menjalankan Program

Untuk melakukan prediksi/klasifikasi gambar:

```bash
python test.py
```

## Menjalankan API

Untuk melakukan prediksi/klasifikasi gambar:

```bash
uvicorn main:app
```

## Menonaktifkan Virtual Environment

```bash
deactivate
```
