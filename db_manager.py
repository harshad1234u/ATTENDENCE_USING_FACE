"""
db_manager.py
─────────────
Database layer for the Face Recognition Attendance System.
Handles all MySQL CRUD operations for the `students` and
`attendance_logs` tables inside the `attendance_db` database.
"""

import pickle
from datetime import date, datetime

import bcrypt
import mysql.connector
from mysql.connector import Error
import numpy as np


class DatabaseManager:
    """Manages the MySQL connection and provides helper methods
    for student registration and attendance logging."""

    def __init__(
        self,
        host: str = "localhost",
        user: str = "root",
        password: str = "root",
        database: str = "attendance_db",
    ):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------
    def connect(self):
        """Establish a connection to the MySQL server."""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
            )
            if self.connection.is_connected():
                print("[DB] Connected to MySQL successfully.")
        except Error as e:
            print(f"[DB] Connection error: {e}")
            raise

    def close(self):
        """Close the active database connection."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("[DB] Connection closed.")

    def _ensure_connection(self):
        """Reconnect if the connection was lost."""
        if self.connection is None or not self.connection.is_connected():
            self.connect()

    # ------------------------------------------------------------------
    # Admin Authentication
    # ------------------------------------------------------------------
    def verify_admin(self, username: str, password: str) -> bool:
        """Verify admin login credentials using bcrypt."""
        self._ensure_connection()
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("SELECT password FROM admins WHERE username = %s", (username,))
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                # bcrypt.checkpw expects bytes
                hashed_pw = result["password"].encode('utf-8')
                return bcrypt.checkpw(password.encode('utf-8'), hashed_pw)
            return False
        except Error as e:
            print(f"[DB] Error verifying admin: {e}")
            return False

    # ------------------------------------------------------------------
    # Student CRUD
    # ------------------------------------------------------------------
    def add_student(
        self,
        roll_no: str,
        name: str,
        class_name: str,
        face_encoding: np.ndarray,
    ) -> bool:
        """Insert a new student with their face encoding.

        Parameters
        ----------
        roll_no : str
            Unique roll / enrollment number.
        name : str
            Full name of the student.
        class_name : str
            Class or section identifier.
        face_encoding : np.ndarray
            128-d face encoding from the `face_recognition` library.

        Returns
        -------
        bool
            True if the insert was successful, False otherwise.
        """
        self._ensure_connection()
        try:
            cursor = self.connection.cursor()
            # Serialize the numpy array to bytes using pickle
            encoding_blob = pickle.dumps(face_encoding)
            query = (
                "INSERT INTO students (roll_no, name, class_name, face_encoding) "
                "VALUES (%s, %s, %s, %s)"
            )
            cursor.execute(query, (roll_no, name, class_name, encoding_blob))
            self.connection.commit()
            cursor.close()
            print(f"[DB] Student '{name}' (Roll: {roll_no}) registered.")
            return True
        except mysql.connector.IntegrityError:
            # Duplicate primary key — student already registered
            print(f"[DB] Roll number '{roll_no}' is already registered.")
            return False
        except Error as e:
            print(f"[DB] Error adding student: {e}")
            return False

    def student_exists(self, roll_no: str) -> bool:
        """Check whether a student with the given roll number exists."""
        self._ensure_connection()
        cursor = self.connection.cursor()
        cursor.execute("SELECT 1 FROM students WHERE roll_no = %s", (roll_no,))
        result = cursor.fetchone()
        cursor.close()
        return result is not None

    def get_all_students(self) -> list:
        """Retrieve all students with their decoded face encodings.

        Returns
        -------
        list of dict
            Each dict contains keys: roll_no, name, class_name, face_encoding.
        """
        self._ensure_connection()
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute("SELECT roll_no, name, class_name, face_encoding FROM students")
        rows = cursor.fetchall()
        cursor.close()

        students = []
        for row in rows:
            students.append(
                {
                    "roll_no": row["roll_no"],
                    "name": row["name"],
                    "class_name": row["class_name"],
                    "face_encoding": pickle.loads(row["face_encoding"]),
                }
            )
        return students

    def update_student(self, roll_no: str, name: str, class_name: str) -> bool:
        """Update an existing student's name or class."""
        self._ensure_connection()
        try:
            cursor = self.connection.cursor()
            query = "UPDATE students SET name = %s, class_name = %s WHERE roll_no = %s"
            cursor.execute(query, (name, class_name, roll_no))
            self.connection.commit()
            updated = cursor.rowcount > 0
            cursor.close()
            if updated:
                print(f"[DB] Student '{roll_no}' updated.")
            return updated
        except Error as e:
            print(f"[DB] Error updating student: {e}")
            return False

    def delete_student(self, roll_no: str) -> bool:
        """Delete a student and their face encoding from the database."""
        self._ensure_connection()
        try:
            cursor = self.connection.cursor()
            query = "DELETE FROM students WHERE roll_no = %s"
            cursor.execute(query, (roll_no,))
            self.connection.commit()
            deleted = cursor.rowcount > 0
            cursor.close()
            if deleted:
                print(f"[DB] Student '{roll_no}' deleted.")
            return deleted
        except Error as e:
            print(f"[DB] Error deleting student: {e}")
            return False

    def search_students(self, query_str: str) -> list:
        """Search students by name or roll number."""
        self._ensure_connection()
        cursor = self.connection.cursor(dictionary=True)
        query = (
            "SELECT roll_no, name, class_name, face_encoding FROM students "
            "WHERE name LIKE %s OR roll_no LIKE %s"
        )
        like_pattern = f"%{query_str}%"
        cursor.execute(query, (like_pattern, like_pattern))
        rows = cursor.fetchall()
        cursor.close()

        students = []
        for row in rows:
            students.append(
                {
                    "roll_no": row["roll_no"],
                    "name": row["name"],
                    "class_name": row["class_name"],
                    "face_encoding": pickle.loads(row["face_encoding"]),
                }
            )
        return students

    # ------------------------------------------------------------------
    # Attendance CRUD
    # ------------------------------------------------------------------
    def log_attendance(self, roll_no: str) -> bool:
        """Log attendance for a student (once per day only).

        Uses INSERT IGNORE so a duplicate (roll_no, date) pair is
        silently ignored thanks to the UNIQUE constraint in the schema.

        Parameters
        ----------
        roll_no : str
            The student's roll number.

        Returns
        -------
        bool
            True if a *new* record was inserted, False if the student
            was already logged today or an error occurred.
        """
        self._ensure_connection()
        try:
            cursor = self.connection.cursor()
            now = datetime.now()
            query = (
                "INSERT IGNORE INTO attendance_logs (roll_no, date, time) "
                "VALUES (%s, %s, %s)"
            )
            cursor.execute(query, (roll_no, now.date(), now.time()))
            self.connection.commit()
            inserted = cursor.rowcount > 0
            cursor.close()
            if inserted:
                print(
                    f"[DB] Attendance logged for '{roll_no}' at "
                    f"{now.strftime('%H:%M:%S')}."
                )
            return inserted
        except Error as e:
            print(f"[DB] Error logging attendance: {e}")
            return False

    def is_attendance_logged_today(self, roll_no: str) -> bool:
        """Check if the student already has an attendance entry for today."""
        self._ensure_connection()
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT 1 FROM attendance_logs WHERE roll_no = %s AND date = %s",
            (roll_no, date.today()),
        )
        result = cursor.fetchone()
        cursor.close()
        return result is not None

    def get_attendance_today(self) -> list:
        """Fetch all attendance records for today.

        Returns
        -------
        list of dict
            Each dict has keys: roll_no, name, class_name, time.
        """
        self._ensure_connection()
        cursor = self.connection.cursor(dictionary=True)
        query = (
            "SELECT a.roll_no, s.name, s.class_name, a.time "
            "FROM attendance_logs a "
            "JOIN students s ON a.roll_no = s.roll_no "
            "WHERE a.date = %s "
            "ORDER BY a.time ASC"
        )
        cursor.execute(query, (date.today(),))
        rows = cursor.fetchall()
        cursor.close()
        return rows
