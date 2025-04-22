import os
import tkinter as tk
from tkinter import messagebox
import traceback

# Import modul-modul aplikasi
from utils.config import Config
from utils.logger import setup_logger
from database import Database
from face_utils import FaceUtils
from camera import CameraManager
from gui.main_window import MainWindow

def main():
    """Fungsi utama aplikasi"""
    try:
        # Setup logging
        logger = setup_logger(name="main")
        logger.info("Starting Face Attendance System")
        
        # Cek dan buat direktori yang dibutuhkan
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/logs", exist_ok=True)
        os.makedirs("data/faces", exist_ok=True)
        
        # Buat root window
        root = tk.Tk()
        
        # Tampilkan splash screen atau pesan loading
        splash_label = tk.Label(root, text="Memuat Sistem Absensi...", font=("Helvetica", 16))
        splash_label.pack(padx=20, pady=20)
        root.update()
        
        # Inisialisasi aplikasi
        main_window = MainWindow(root)
        
        # Hapus splash screen
        splash_label.destroy()
        
        # Jalankan main loop
        root.mainloop()
        
    except Exception as e:
        # Log error
        if 'logger' in locals():
            logger.error(f"Error in main: {str(e)}")
            logger.error(traceback.format_exc())
        
        # Tampilkan pesan error
        messagebox.showerror("Error", f"Terjadi kesalahan saat menjalankan aplikasi:\n{str(e)}")

if __name__ == "__main__":
    main() 