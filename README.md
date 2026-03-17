# Face Recognition Attendance Management System

A GUI-based attendance system that uses real-time face recognition to automatically log student attendance.

## Tech Stack

| Component       | Library                          |
| --------------- | -------------------------------- |
| GUI             | `customtkinter`                  |
| Computer Vision | `opencv-python`, `face-recognition` |
| Database        | `mysql-connector-python` (MySQL) |
| Data Handling   | `numpy`, `pickle`, `Pillow`      |

## Prerequisites

1. **Python 3.10+**
2. **MySQL Server** running locally
3. **CMake** and a **C++ compiler** (required to build `dlib`, a dependency of `face-recognition`)
   - On Windows: install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the "Desktop development with C++" workload
   - Then: `pip install cmake dlib`

## Setup

### 1. Create the Database

Open MySQL Workbench (or CLI) and run:

```sql
source schema.sql;
```

This creates the `attendance_db` database with `students` and `attendance_logs` tables.

### 2. Configure Database Credentials

Edit the `DB_CONFIG` dictionary at the top of `app.py`:

```python
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "YOUR_PASSWORD_HERE",
    "database": "attendance_db",
}
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python app.py
```

## Usage

### Registering a Student
1. Click **"Open Registration"** from the main menu
2. Fill in Name, Roll Number, and Class
3. Click **"Capture Face"** — a webcam window opens
4. Position your face and press **SPACE** to capture (or **Q** to cancel)
5. Click **"Register Student"** to save

### Taking Attendance
1. Click **"Open Attendance"** from the main menu
2. The live camera feed will detect and recognise registered faces
3. Attendance is automatically logged (once per student per day)
4. The right panel shows today's attendance in real time
