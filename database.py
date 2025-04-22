import os
import sqlite3
import pickle
import numpy as np
from datetime import datetime

class Database:
    def __init__(self, db_path="data/database.db"):
        # Memastikan direktori data ada
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.create_tables()
    
    def get_connection(self):
        """Get a new connection to the database"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    
    def close(self):
        """Method kept for compatibility"""
        pass
    
    def create_tables(self):
        """Membuat tabel-tabel yang diperlukan jika belum ada"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Tabel Employees
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            department TEXT,
            position TEXT,
            face_encoding BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Tabel Attendance
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            check_in TIMESTAMP,
            check_out TIMESTAMP,
            date TEXT,
            FOREIGN KEY (employee_id) REFERENCES employees (id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_employee(self, employee_id, name, department="", position="", face_encoding=None):
        """Menambahkan karyawan baru ke database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Konversi face_encoding numpy array ke bytes untuk penyimpanan
            face_encoding_bytes = pickle.dumps(face_encoding) if face_encoding is not None else None
            
            cursor.execute('''
            INSERT INTO employees (employee_id, name, department, position, face_encoding)
            VALUES (?, ?, ?, ?, ?)
            ''', (employee_id, name, department, position, face_encoding_bytes))
            
            conn.commit()
            last_id = cursor.lastrowid
            conn.close()
            return True, last_id
        except sqlite3.IntegrityError:
            return False, "Employee ID already exists"
        except Exception as e:
            return False, str(e)
    
    def update_employee(self, id, employee_id=None, name=None, department=None, position=None, face_encoding=None):
        """Mengupdate data karyawan"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Dapatkan data karyawan saat ini
            cursor.execute("SELECT * FROM employees WHERE id = ?", (id,))
            employee = cursor.fetchone()
            
            if not employee:
                conn.close()
                return False, "Employee not found"
            
            # Update nilai-nilai yang diberikan
            employee_id = employee_id if employee_id is not None else employee['employee_id']
            name = name if name is not None else employee['name']
            department = department if department is not None else employee['department']
            position = position if position is not None else employee['position']
            
            # Untuk face_encoding, kita perlu special handling karena disimpan sebagai BLOB
            if face_encoding is not None:
                face_encoding_bytes = pickle.dumps(face_encoding)
            else:
                face_encoding_bytes = employee['face_encoding']
            
            cursor.execute('''
            UPDATE employees
            SET employee_id = ?, name = ?, department = ?, position = ?, face_encoding = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            ''', (employee_id, name, department, position, face_encoding_bytes, id))
            
            conn.commit()
            conn.close()
            return True, id
        except sqlite3.IntegrityError:
            return False, "Employee ID already exists"
        except Exception as e:
            return False, str(e)
    
    def delete_employee(self, id):
        """Menghapus karyawan dari database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM employees WHERE id = ?", (id,))
            conn.commit()
            
            rowcount = cursor.rowcount
            conn.close()
            
            if rowcount > 0:
                return True, id
            else:
                return False, "Employee not found"
        except Exception as e:
            return False, str(e)
    
    def get_all_employees(self):
        """Mendapatkan semua data karyawan"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM employees ORDER BY name")
        employees = cursor.fetchall()
        
        result = []
        for employee in employees:
            emp_dict = dict(employee)
            
            # Konversi BLOB face_encoding kembali ke numpy array
            if emp_dict['face_encoding']:
                emp_dict['face_encoding'] = pickle.loads(emp_dict['face_encoding'])
            
            result.append(emp_dict)
        
        conn.close()
        return result
    
    def get_employee_by_id(self, id):
        """Mendapatkan data karyawan berdasarkan ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM employees WHERE id = ?", (id,))
        employee = cursor.fetchone()
        
        if employee:
            emp_dict = dict(employee)
            
            # Konversi BLOB face_encoding kembali ke numpy array
            if emp_dict['face_encoding']:
                emp_dict['face_encoding'] = pickle.loads(emp_dict['face_encoding'])
            
            conn.close()
            return emp_dict
        
        conn.close()
        return None
    
    def get_employee_by_employee_id(self, employee_id):
        """Mendapatkan data karyawan berdasarkan employee_id"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM employees WHERE employee_id = ?", (employee_id,))
        employee = cursor.fetchone()
        
        if employee:
            emp_dict = dict(employee)
            
            # Konversi BLOB face_encoding kembali ke numpy array
            if emp_dict['face_encoding']:
                emp_dict['face_encoding'] = pickle.loads(emp_dict['face_encoding'])
            
            conn.close()
            return emp_dict
        
        conn.close()
        return None
    
    def get_all_face_encodings(self):
        """Mendapatkan semua face encodings dengan ID karyawan untuk pengenalan wajah"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, name, face_encoding FROM employees WHERE face_encoding IS NOT NULL")
        employees = cursor.fetchall()
        
        result = []
        for employee in employees:
            emp_dict = {
                'id': employee['id'],
                'name': employee['name'],
                'face_encoding': pickle.loads(employee['face_encoding'])
            }
            result.append(emp_dict)
            
        conn.close()
        return result
    
    def record_attendance(self, employee_id, check_in=True):
        """Mencatat absensi karyawan (check-in atau check-out)"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            today = datetime.now().strftime('%Y-%m-%d')
            
            if check_in:
                # Periksa apakah sudah ada check-in hari ini
                cursor.execute("""
                SELECT * FROM attendance 
                WHERE employee_id = ? AND date = ? AND check_in IS NOT NULL
                """, (employee_id, today))
                
                existing = cursor.fetchone()
                
                if existing:
                    # Sudah check-in hari ini, jadi update check-out
                    cursor.execute("""
                    UPDATE attendance 
                    SET check_out = CURRENT_TIMESTAMP 
                    WHERE id = ?
                    """, (existing['id'],))
                    
                    conn.commit()
                    conn.close()
                    return True, "check-out", existing['id']
                else:
                    # Belum check-in hari ini
                    now = datetime.now()
                    cursor.execute("""
                    INSERT INTO attendance (employee_id, check_in, date)
                    VALUES (?, ?, ?)
                    """, (employee_id, now, today))
                    
                    conn.commit()
                    last_id = cursor.lastrowid
                    conn.close()
                    return True, "check-in", last_id
            else:
                # Langsung update check-out terakhir
                cursor.execute("""
                SELECT * FROM attendance 
                WHERE employee_id = ? AND date = ? AND check_in IS NOT NULL
                ORDER BY check_in DESC LIMIT 1
                """, (employee_id, today))
                
                existing = cursor.fetchone()
                
                if existing:
                    cursor.execute("""
                    UPDATE attendance 
                    SET check_out = CURRENT_TIMESTAMP 
                    WHERE id = ?
                    """, (existing['id'],))
                    
                    conn.commit()
                    conn.close()
                    return True, "check-out", existing['id']
                else:
                    conn.close()
                    return False, "No check-in record found", None
                
        except Exception as e:
            return False, str(e), None
    
    def get_attendance_records(self, employee_id=None, start_date=None, end_date=None):
        """Mendapatkan rekaman absensi dengan filter"""
        query = """
        SELECT a.*, e.name, e.employee_id as employee_code
        FROM attendance a
        JOIN employees e ON a.employee_id = e.id
        WHERE 1=1
        """
        params = []
        
        if employee_id:
            query += " AND a.employee_id = ?"
            params.append(employee_id)
        
        if start_date:
            query += " AND a.date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND a.date <= ?"
            params.append(end_date)
        
        query += " ORDER BY a.date DESC, a.check_in DESC"
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(query, params)
        records = cursor.fetchall()
        
        result = [dict(record) for record in records]
        
        conn.close()
        return result
    
    def get_distinct_departments(self):
        """Mendapatkan daftar unik departemen"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT DISTINCT department FROM employees WHERE department IS NOT NULL AND department != '' ORDER BY department")
        departments = cursor.fetchall()
        
        result = [dept['department'] for dept in departments]
        
        conn.close()
        return result
    
    def get_distinct_positions(self):
        """Mendapatkan daftar unik posisi/jabatan"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT DISTINCT position FROM employees WHERE position IS NOT NULL AND position != '' ORDER BY position")
        positions = cursor.fetchall()
        
        result = [pos['position'] for pos in positions]
        
        conn.close()
        return result 