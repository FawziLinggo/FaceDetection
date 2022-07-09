# FaceDetection
ini adalah Program untuk Mendeteksi wajah yang berjarak dari 1, 60, 100 hingga 150 Meter 


Setiap log akan dapat di dilhat pada file log `app.log` berisi tentang
setiap pengerjaan yang telah dibuat.

Program terdiri dari :
- `def run_program` : Program untuk menjalankan program secara keseluruhan
- `def homomorphic` : Untuk run filtering homomorphic
- `def face_detector` : Algoritma Pendeteksi wajah
- `morfologi` : Algoritma morfologi Dilasi, erosi, opening, dan closing

Berikut Catatan :
- Class HomomorphicFilter adalah Class untuk melakukan Filtering HomomorphicFilter. 

yang perlu di install :
- Sckit-images
  ```shell
  pip python -m pip install -U pip
  python -m pip install -U scikit-image
  ```
- PIL
    ```shell
    pip install pillow
    ```