import os
import logging
from datetime import datetime

def setup_logger(name='face_attendance', log_level=logging.INFO, log_to_file=True, log_file='data/logs/app.log'):
    """
    Setup logger untuk aplikasi
    
    Args:
        name (str): Nama logger
        log_level: Level log (logging.DEBUG, logging.INFO, dll)
        log_to_file (bool): Apakah log akan ditulis ke file
        log_file (str): Path file log
        
    Returns:
        logging.Logger: Logger yang dikonfigurasi
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Buat formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Buat console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Tambahkan handler ke logger
    logger.addHandler(console_handler)
    
    # Tambahkan file handler jika log_to_file=True
    if log_to_file:
        try:
            # Buat direktori log jika belum ada
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                
            # Buat file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            
            # Tambahkan ke logger
            logger.addHandler(file_handler)
        except Exception as e:
            logger.error(f"Gagal mengatur file logging: {str(e)}")
    
    return logger

def log_exception(logger, e, message="Exception terjadi"):
    """
    Log exception dengan detail
    
    Args:
        logger (logging.Logger): Logger untuk digunakan
        e (Exception): Exception yang terjadi
        message (str): Pesan tambahan
    """
    logger.error(f"{message}: {str(e)}")
    logger.exception(e)

def log_attendance(employee_name, employee_id, action, success):
    """
    Log aktivitas absensi ke file terpisah
    
    Args:
        employee_name (str): Nama karyawan
        employee_id (str): ID karyawan
        action (str): Tindakan (check-in atau check-out)
        success (bool): Apakah tindakan berhasil
    """
    try:
        # Buat direktori log jika belum ada
        log_dir = 'data/logs'
        os.makedirs(log_dir, exist_ok=True)
        
        # Nama file log berdasarkan tanggal
        today = datetime.now().strftime('%Y-%m-%d')
        log_file = f"{log_dir}/attendance_{today}.log"
        
        # Format waktu
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Format log entry
        status = "SUCCESS" if success else "FAILED"
        log_entry = f"{timestamp} | {status} | {employee_id} | {employee_name} | {action}\n"
        
        # Tulis ke file
        with open(log_file, 'a') as f:
            f.write(log_entry)
    except Exception as e:
        # Fallback ke logging standar jika gagal
        logging.error(f"Failed to log attendance: {str(e)}")
        
# Contoh penggunaan
# logger = setup_logger()
# try:
#     # Kode yang mungkin menimbulkan exception
#     result = 10 / 0
# except Exception as e:
#     log_exception(logger, e, "Error saat melakukan operasi pembagian") 