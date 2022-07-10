""" untuk Kebutuhan logging """
import logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

"""Memanggil Fungsi Fungsi pada file Condition.py"""
import condition

while(True):
    print("++++++ Pilih Program Program ++++++\n"
          "1. Filtering Pada semua foto di directory \n"
          "2. Filtering, Morfologi dan Face Detection (hanya 1 Foto) \n"
          "3. Exit Program \n"
          "Pilih sesuai angka (Harus Angka)"
          )
    Hello = int(input("Masukkan Angka : "))
    if(Hello==2):
        condition.run_program("")
        logging.info("Menjalankan Program Filtering, Morfologi dan Face Detection (hanya 1 Foto)")
    elif(Hello==1):
        print("program sedang dibuat")
    elif(Hello==3):
        print("Terima Kasih")
        break
    else:
        print("Oh tidak bisa, harus angka 1 dan 2 ya")