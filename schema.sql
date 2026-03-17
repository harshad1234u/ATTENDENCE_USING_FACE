-- ============================================================
-- Face Recognition Attendance Management System
-- Database Schema
-- ============================================================

-- Create the database (if it doesn't already exist)
CREATE DATABASE IF NOT EXISTS attendance_db;
USE attendance_db;

-- -----------------------------------------------------------
-- Table: students
-- Stores registered student info and their face encodings.
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS students (
    roll_no     VARCHAR(20)  PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,
    class_name  VARCHAR(50)  NOT NULL,
    face_encoding LONGBLOB   NOT NULL,          -- pickled 128-d numpy array
    registered_at DATETIME   DEFAULT CURRENT_TIMESTAMP
);

-- -----------------------------------------------------------
-- Table: attendance_logs
-- Stores one attendance entry per student per day.
-- The UNIQUE constraint prevents duplicate daily entries.
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS attendance_logs (
    log_id   INT          AUTO_INCREMENT PRIMARY KEY,
    roll_no  VARCHAR(20)  NOT NULL,
    date     DATE         NOT NULL,
    time     TIME         NOT NULL,
    FOREIGN KEY (roll_no) REFERENCES students(roll_no)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    UNIQUE KEY unique_daily_attendance (roll_no, date)
);

-- -----------------------------------------------------------
-- Table: admins
-- Stores admin credentials for the Admin Panel.
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS admins (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    username    VARCHAR(50)  UNIQUE NOT NULL,
    password    VARCHAR(255) NOT NULL  -- bcrypt hashed
);

-- Seed a default admin account (username: admin, password: admin123)
-- bcrypt hash for 'admin123'
INSERT IGNORE INTO admins (username, password) VALUES ('admin', '$2b$12$L/X.A1eA/H220f1V4QWb3OpN/xK58o3ZJ9h0GvQk0X4k2IcGzM.3a');
