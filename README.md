# Face Recognition Attendance Management System

A GUI-based attendance system that uses real-time face recognition to automatically log student attendance.

## Tech Stack

| Component       | Library                          |
| --------------- | -------------------------------- |
| GUI             | `customtkinter`                  |
| Computer Vision | `opencv-python`, `face-recognition` |
| Database        | `mysql-connector-python` (MySQL) |
| Data Handling   | `numpy`, `Pillow`                |
| Config          | `python-dotenv`                  |

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

This creates the `attendance_db` database with `students`, `attendance_logs`, and `admins` tables, and seeds a default admin account (`admin` / `admin123`).

### 2. Configure Database Credentials

Copy `.env.example` to `.env` and fill in your MySQL credentials:

```bash
cp .env.example .env
```

Then edit `.env`:

```ini
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_mysql_password_here
DB_NAME=attendance_db
```

> **Never commit `.env` to version control.** It is listed in `.gitignore`.

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
1. Log in via the **Admin Panel** (default: `admin` / `admin123`)
2. Click **"Register New Student"**
3. Fill in Name, Roll Number, and Class
4. Click **"Capture Face"** — a webcam window opens
5. Position your face and press **SPACE** to capture (or **Q** to cancel)
6. Click **"Register Student"** to save

### Taking Attendance
1. Click **"Open Attendance"** from the main menu
2. The live camera feed will detect and recognise registered faces
3. Attendance is automatically logged (once per student per day)
4. The right panel shows today's attendance in real time

### Managing Students (Admin Panel)
1. Log in via the **Admin Panel**
2. Click **"Manage Students"** to view, search, edit, or delete students

### Exporting Attendance to CSV (Admin Panel)
1. Log in via the **Admin Panel**
2. Click **"Export Attendance"**
3. Enter a start and end date (YYYY-MM-DD format)
4. Click **"Choose File & Export"** and pick where to save the CSV file
