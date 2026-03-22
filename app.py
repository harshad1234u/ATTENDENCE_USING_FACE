"""
app.py
──────
Main entry-point for the Face Recognition Attendance Management System.

Uses `customtkinter` for the GUI, `opencv-python` + `face_recognition`
for computer vision, and MySQL via `db_manager.py` for persistence.
"""

import csv
import os
import threading
import tkinter as tk
from datetime import datetime, date
from tkinter import messagebox, filedialog

import cv2
import customtkinter as ctk
import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageTk

from db_manager import DatabaseManager
from face_utils import FaceRecognition

# Load environment variables from .env (if present)
load_dotenv()

# ══════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "root"),
    "database": os.getenv("DB_NAME", "attendance_db"),
}

# customtkinter appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ══════════════════════════════════════════════════════════════
# Main Application Class
# ══════════════════════════════════════════════════════════════
class AttendanceApp(ctk.CTk):
    """Root GUI window — acts as the main menu and manages child frames."""

    WIDTH = 900
    HEIGHT = 620

    def __init__(self):
        super().__init__()

        self.title("Face Recognition Attendance System")
        self.geometry(f"{self.WIDTH}x{self.HEIGHT}")
        self.resizable(False, False)

        # Database manager (shared across all modules)
        self.db = DatabaseManager(**DB_CONFIG)
        try:
            self.db.connect()
        except Exception as e:
            messagebox.showerror(
                "Database Error",
                f"Cannot connect to MySQL.\n\n{e}\n\n"
                "Please make sure MySQL is running and the database "
                "'attendance_db' exists (run schema.sql first).",
            )
            self.destroy()
            return

        # Face recognition utility
        self.face_rec = FaceRecognition()

        # Container frame — all pages are stacked here
        self.container = ctk.CTkFrame(self, corner_radius=0)
        self.container.pack(fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        # Page registry
        self.frames: dict[str, ctk.CTkFrame] = {}
        for PageClass in (MainMenuPage, RegistrationPage, LiveAttendancePage, AdminLoginPage, AdminDashboardPage, StudentManagementPage, ExportAttendancePage):
            page = PageClass(parent=self.container, controller=self)
            self.frames[PageClass.__name__] = page
            page.grid(row=0, column=0, sticky="nsew")

        self.show_frame("MainMenuPage")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def show_frame(self, name: str):
        """Raise the frame with the given class name to the front."""
        frame = self.frames[name]
        frame.tkraise()
        # If the page has an on_show hook, call it
        if hasattr(frame, "on_show"):
            frame.on_show()

    def _on_close(self):
        """Cleanup on window close."""
        # Stop any active webcam feeds
        for frame in self.frames.values():
            if hasattr(frame, "stop"):
                frame.stop()
        self.db.close()
        self.destroy()


# ══════════════════════════════════════════════════════════════
# Page 1 — Main Menu
# ══════════════════════════════════════════════════════════════
class MainMenuPage(ctk.CTkFrame):
    """Home screen with two primary action buttons."""

    def __init__(self, parent, controller: AttendanceApp):
        super().__init__(parent, corner_radius=0)
        self.controller = controller

        # ── Header ────────────────────────────────────────────
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(pady=(60, 10))

        ctk.CTkLabel(
            header,
            text="📋  Attendance Management System",
            font=ctk.CTkFont(size=28, weight="bold"),
        ).pack()

        ctk.CTkLabel(
            header,
            text="Face Recognition Powered  •  Real-Time Tracking",
            font=ctk.CTkFont(size=14),
            text_color="gray",
        ).pack(pady=(4, 0))

        # ── Action Cards ─────────────────────────────────────
        card_frame = ctk.CTkFrame(self, fg_color="transparent")
        card_frame.pack(expand=True)

        # Admin Panel button
        reg_card = ctk.CTkFrame(card_frame, width=320, height=220, corner_radius=16)
        reg_card.pack(side="left", padx=30)
        reg_card.pack_propagate(False)

        ctk.CTkLabel(
            reg_card, text="🔐", font=ctk.CTkFont(size=48)
        ).pack(pady=(30, 8))
        ctk.CTkLabel(
            reg_card,
            text="Admin Panel",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack()
        ctk.CTkLabel(
            reg_card,
            text="Manage students & system settings",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        ).pack(pady=(2, 12))
        ctk.CTkButton(
            reg_card,
            text="Open Admin Panel",
            width=200,
            height=40,
            corner_radius=10,
            command=lambda: controller.show_frame("AdminLoginPage"),
        ).pack()

        # Live attendance button
        att_card = ctk.CTkFrame(card_frame, width=320, height=220, corner_radius=16)
        att_card.pack(side="left", padx=30)
        att_card.pack_propagate(False)

        ctk.CTkLabel(
            att_card, text="📸", font=ctk.CTkFont(size=48)
        ).pack(pady=(30, 8))
        ctk.CTkLabel(
            att_card,
            text="Start Live Attendance",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack()
        ctk.CTkLabel(
            att_card,
            text="Real-time face recognition feed",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        ).pack(pady=(2, 12))
        ctk.CTkButton(
            att_card,
            text="Open Attendance",
            width=200,
            height=40,
            corner_radius=10,
            fg_color="#28a745",
            hover_color="#218838",
            command=lambda: controller.show_frame("LiveAttendancePage"),
        ).pack()

        # ── Footer ────────────────────────────────────────────
        ctk.CTkLabel(
            self,
            text="Powered by OpenCV & face_recognition",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        ).pack(side="bottom", pady=16)


# ══════════════════════════════════════════════════════════════
# Page 2 — Student Registration
# ══════════════════════════════════════════════════════════════
class RegistrationPage(ctk.CTkFrame):
    """Form to register a new student with face capture."""

    def __init__(self, parent, controller: AttendanceApp):
        super().__init__(parent, corner_radius=0)
        self.controller = controller
        self.captured_encoding: np.ndarray | None = None
        self.captured_image: np.ndarray | None = None

        # ── Back button ───────────────────────────────────────
        top_bar = ctk.CTkFrame(self, fg_color="transparent")
        top_bar.pack(fill="x", padx=20, pady=(14, 0))
        ctk.CTkButton(
            top_bar,
            text="← Back to Dashboard",
            width=150,
            fg_color="transparent",
            border_width=1,
            command=lambda: controller.show_frame("AdminDashboardPage"),
        ).pack(side="left")

        # ── Title ─────────────────────────────────────────────
        ctk.CTkLabel(
            self,
            text="Register New Student",
            font=ctk.CTkFont(size=24, weight="bold"),
        ).pack(pady=(10, 20))

        # ── Content area ──────────────────────────────────────
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=40)

        # Left side — form
        form_frame = ctk.CTkFrame(content, width=360)
        form_frame.pack(side="left", fill="y", padx=(0, 20))
        form_frame.pack_propagate(False)

        pad = {"padx": 20, "pady": (10, 0), "anchor": "w"}
        entry_pad = {"padx": 20, "pady": (2, 0)}

        ctk.CTkLabel(form_frame, text="Full Name", font=ctk.CTkFont(size=13)).pack(**pad)
        self.name_entry = ctk.CTkEntry(form_frame, width=300, placeholder_text="e.g. John Doe")
        self.name_entry.pack(**entry_pad)

        ctk.CTkLabel(form_frame, text="Roll Number", font=ctk.CTkFont(size=13)).pack(**pad)
        self.roll_entry = ctk.CTkEntry(form_frame, width=300, placeholder_text="e.g. CSE2024001")
        self.roll_entry.pack(**entry_pad)

        ctk.CTkLabel(form_frame, text="Class / Section", font=ctk.CTkFont(size=13)).pack(**pad)
        self.class_entry = ctk.CTkEntry(form_frame, width=300, placeholder_text="e.g. CS-A")
        self.class_entry.pack(**entry_pad)

        self.capture_btn = ctk.CTkButton(
            form_frame,
            text="📷  Capture Face",
            width=300,
            height=42,
            corner_radius=10,
            command=self._capture_face,
        )
        self.capture_btn.pack(padx=20, pady=(24, 0))

        self.status_label = ctk.CTkLabel(
            form_frame, text="", font=ctk.CTkFont(size=12), wraplength=280
        )
        self.status_label.pack(padx=20, pady=(8, 0))

        self.register_btn = ctk.CTkButton(
            form_frame,
            text="✅  Register Student",
            width=300,
            height=42,
            corner_radius=10,
            fg_color="#28a745",
            hover_color="#218838",
            state="disabled",
            command=self._register_student,
        )
        self.register_btn.pack(padx=20, pady=(16, 20))

        # Right side — face preview
        preview_frame = ctk.CTkFrame(content, width=400)
        preview_frame.pack(side="left", fill="both", expand=True)
        preview_frame.pack_propagate(False)

        ctk.CTkLabel(
            preview_frame, text="Face Preview", font=ctk.CTkFont(size=15, weight="bold")
        ).pack(pady=(16, 4))

        self.preview_label = ctk.CTkLabel(preview_frame, text="No image captured yet")
        self.preview_label.pack(expand=True)

    # ------------------------------------------------------------------
    def on_show(self):
        """Reset the form every time the page is shown."""
        self.name_entry.delete(0, "end")
        self.roll_entry.delete(0, "end")
        self.class_entry.delete(0, "end")
        self.captured_encoding = None
        self.captured_image = None
        self.status_label.configure(text="", text_color="white")
        self.register_btn.configure(state="disabled")
        self.preview_label.configure(image=None, text="No image captured yet")

    # ------------------------------------------------------------------
    def _capture_face(self):
        """Open the webcam, capture a single frame, and extract the encoding."""
        self.status_label.configure(text="Opening camera…", text_color="cyan")
        self.update_idletasks()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.status_label.configure(
                text="❌ Cannot open webcam.", text_color="red"
            )
            return

        # Show a small live preview window so the user can position themselves
        messagebox.showinfo(
            "Capture Instructions",
            "A camera window will open.\n\n"
            "Position your face clearly in front of the camera,\n"
            "then press 'SPACE' to capture or 'Q' to cancel.",
        )

        captured_frame = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Mirror for natural feel
            display = cv2.flip(frame, 1)
            cv2.putText(
                display,
                "Press SPACE to capture | Q to cancel",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.imshow("Capture Face", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                captured_frame = cv2.flip(frame, 1)
                break
            elif key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        if captured_frame is None:
            self.status_label.configure(text="Capture cancelled.", text_color="orange")
            return

        # Process the captured frame
        encoding, location, msg = self.controller.face_rec.capture_face_encoding(
            captured_frame
        )

        if encoding is None:
            self.status_label.configure(text=f"❌ {msg}", text_color="red")
            return

        # Success — store encoding & show preview
        self.captured_encoding = encoding
        self.captured_image = captured_frame

        # Draw bounding box on preview
        top, right, bottom, left = location
        preview = captured_frame.copy()
        cv2.rectangle(preview, (left, top), (right, bottom), (0, 200, 0), 2)

        # Convert for tkinter display
        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(preview_rgb)
        pil_img = pil_img.resize((360, 270), Image.LANCZOS)
        tk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(360, 270))

        self.preview_label.configure(image=tk_img, text="")
        self.preview_label.image = tk_img  # keep reference

        self.status_label.configure(text=f"✅ {msg}", text_color="#28a745")
        self.register_btn.configure(state="normal")

    # ------------------------------------------------------------------
    def _register_student(self):
        """Validate form fields and save the student to the database."""
        name = self.name_entry.get().strip()
        roll_no = self.roll_entry.get().strip()
        class_name = self.class_entry.get().strip()

        if not name or not roll_no or not class_name:
            messagebox.showwarning("Missing Fields", "Please fill in all fields.")
            return

        if self.captured_encoding is None:
            messagebox.showwarning("No Face", "Please capture a face first.")
            return

        # Check for duplicate roll number
        if self.controller.db.student_exists(roll_no):
            messagebox.showerror(
                "Duplicate",
                f"Roll number '{roll_no}' is already registered.",
            )
            return

        success = self.controller.db.add_student(
            roll_no, name, class_name, self.captured_encoding
        )

        if success:
            messagebox.showinfo(
                "Success",
                f"Student '{name}' (Roll: {roll_no}) registered successfully!",
            )
            self.on_show()  # reset form
        else:
            messagebox.showerror("Error", "Failed to register student. Check console.")


# ══════════════════════════════════════════════════════════════
# Page 3 — Live Attendance
# ══════════════════════════════════════════════════════════════
class LiveAttendancePage(ctk.CTkFrame):
    """Real-time webcam feed with face recognition & auto attendance."""

    def __init__(self, parent, controller: AttendanceApp):
        super().__init__(parent, corner_radius=0)
        self.controller = controller
        self.cap: cv2.VideoCapture | None = None
        self.running = False
        self._after_id = None

        # Known faces cache (loaded on page show)
        self.known_encodings: list[np.ndarray] = []
        self.known_metadata: list[dict] = []  # [{roll_no, name}, …]

        # ── Top bar ───────────────────────────────────────────
        top_bar = ctk.CTkFrame(self, fg_color="transparent")
        top_bar.pack(fill="x", padx=20, pady=(14, 0))

        ctk.CTkButton(
            top_bar,
            text="← Back to Menu",
            width=140,
            fg_color="transparent",
            border_width=1,
            command=self._go_back,
        ).pack(side="left")

        self.feed_status = ctk.CTkLabel(
            top_bar,
            text="● LIVE",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#28a745",
        )
        self.feed_status.pack(side="right")

        # ── Title ─────────────────────────────────────────────
        ctk.CTkLabel(
            self,
            text="Live Attendance Feed",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).pack(pady=(6, 8))

        # ── Body ──────────────────────────────────────────────
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=20, pady=(0, 14))

        # Left — video feed
        video_frame = ctk.CTkFrame(body, width=580)
        video_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        video_frame.pack_propagate(False)

        self.video_label = ctk.CTkLabel(video_frame, text="Camera starting…")
        self.video_label.pack(expand=True)

        # Right — attendance log panel
        log_panel = ctk.CTkFrame(body, width=260)
        log_panel.pack(side="right", fill="y")
        log_panel.pack_propagate(False)

        ctk.CTkLabel(
            log_panel,
            text="Today's Attendance",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(pady=(12, 6))

        self.log_textbox = ctk.CTkTextbox(log_panel, width=240, state="disabled")
        self.log_textbox.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self.count_label = ctk.CTkLabel(
            log_panel, text="Total: 0", font=ctk.CTkFont(size=12)
        )
        self.count_label.pack(pady=(0, 10))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def on_show(self):
        """Called every time the page is raised to front."""
        self._load_known_faces()
        self._refresh_log_panel()
        self._start_feed()

    def stop(self):
        """Stop the webcam feed gracefully."""
        self.running = False
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

    def _go_back(self):
        """Stop the feed and return to the main menu."""
        self.stop()
        self.controller.show_frame("MainMenuPage")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_known_faces(self):
        """Pull all registered students from the DB into memory."""
        students = self.controller.db.get_all_students()
        self.known_encodings = [s["face_encoding"] for s in students]
        self.known_metadata = [
            {"roll_no": s["roll_no"], "name": s["name"]} for s in students
        ]
        print(f"[Live] Loaded {len(students)} known face(s).")

    def _refresh_log_panel(self):
        """Update the side panel with today's attendance records."""
        records = self.controller.db.get_attendance_today()
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")

        if not records:
            self.log_textbox.insert("end", "No attendance yet today.\n")
        else:
            for i, rec in enumerate(records, 1):
                time_str = str(rec["time"])[:8]  # HH:MM:SS
                self.log_textbox.insert(
                    "end",
                    f"{i}. {rec['name']}\n"
                    f"   Roll: {rec['roll_no']}  |  {time_str}\n\n",
                )

        self.log_textbox.configure(state="disabled")
        self.count_label.configure(text=f"Total: {len(records)}")

    # ------------------------------------------------------------------
    # Video feed loop
    # ------------------------------------------------------------------
    def _start_feed(self):
        """Open the webcam and begin the tkinter-compatible update loop."""
        if self.running:
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.video_label.configure(text="❌ Cannot open webcam.")
            self.feed_status.configure(text="● OFFLINE", text_color="red")
            return

        self.running = True
        self.feed_status.configure(text="● LIVE", text_color="#28a745")
        self._process_frame()

    def _process_frame(self):
        """Read one frame, run recognition, update the GUI, and schedule
        the next iteration.  This runs on the main thread via `after()`
        to stay compatible with tkinter's event loop."""
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.video_label.configure(text="Camera feed lost.")
            self.feed_status.configure(text="● OFFLINE", text_color="red")
            self.running = False
            return

        frame = cv2.flip(frame, 1)

        # Run face recognition
        results = self.controller.face_rec.recognize_faces(
            frame, self.known_encodings, self.known_metadata
        )

        # Auto-log attendance for recognised students
        new_logged = False
        for r in results:
            if r["roll_no"] is not None:
                was_new = self.controller.db.log_attendance(r["roll_no"])
                if was_new:
                    new_logged = True

        if new_logged:
            self._refresh_log_panel()

        # Draw bounding boxes & labels
        annotated = self.controller.face_rec.draw_results(frame, results)

        # Add timestamp overlay
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(
            annotated, ts, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
        )

        # Convert to CTkImage for display
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img = pil_img.resize((560, 420), Image.LANCZOS)
        tk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(560, 420))

        self.video_label.configure(image=tk_img, text="")
        self.video_label.image = tk_img  # prevent garbage collection

        # Schedule next frame (~30 ms ≈ 33 FPS)
        self._after_id = self.after(30, self._process_frame)


# ══════════════════════════════════════════════════════════════
# Admin Pages
# ══════════════════════════════════════════════════════════════
class AdminLoginPage(ctk.CTkFrame):
    """Secure login screen for administrators."""
    def __init__(self, parent, controller: AttendanceApp):
        super().__init__(parent, corner_radius=0)
        self.controller = controller

        # Back button
        top_bar = ctk.CTkFrame(self, fg_color="transparent")
        top_bar.pack(fill="x", padx=20, pady=(14, 0))
        ctk.CTkButton(
            top_bar,
            text="← Back to Menu",
            width=140,
            fg_color="transparent",
            border_width=1,
            command=lambda: controller.show_frame("MainMenuPage"),
        ).pack(side="left")

        # Form container
        form = ctk.CTkFrame(self, width=360, height=400, corner_radius=16)
        form.pack(expand=True)
        form.pack_propagate(False)

        ctk.CTkLabel(form, text="🔐 Admin Login", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(40, 30))

        ctk.CTkLabel(form, text="Username", font=ctk.CTkFont(size=13)).pack(padx=30, anchor="w")
        self.user_entry = ctk.CTkEntry(form, width=300)
        self.user_entry.pack(padx=30, pady=(2, 16))

        ctk.CTkLabel(form, text="Password", font=ctk.CTkFont(size=13)).pack(padx=30, anchor="w")
        self.pass_entry = ctk.CTkEntry(form, width=300, show="*")
        self.pass_entry.pack(padx=30, pady=(2, 24))

        self.error_label = ctk.CTkLabel(form, text="", text_color="red", font=ctk.CTkFont(size=12))
        self.error_label.pack(pady=(0, 10))

        ctk.CTkButton(
            form, text="Login", width=300, height=40, font=ctk.CTkFont(weight="bold"),
            command=self._login
        ).pack(padx=30)
    
    def on_show(self):
        self.user_entry.delete(0, "end")
        self.pass_entry.delete(0, "end")
        self.error_label.configure(text="")

    def _login(self):
        username = self.user_entry.get().strip()
        password = self.pass_entry.get()
        if not username or not password:
            self.error_label.configure(text="Please enter both username and password")
            return
        
        if self.controller.db.verify_admin(username, password):
            self.controller.show_frame("AdminDashboardPage")
        else:
            self.error_label.configure(text="Invalid credentials")


class AdminDashboardPage(ctk.CTkFrame):
    """Dashboard shown to authenticated admins."""
    def __init__(self, parent, controller: AttendanceApp):
        super().__init__(parent, corner_radius=0)
        self.controller = controller

        # Top bar with logout
        top_bar = ctk.CTkFrame(self, fg_color="transparent")
        top_bar.pack(fill="x", padx=20, pady=(14, 0))
        ctk.CTkLabel(top_bar, text="Admin Dashboard", font=ctk.CTkFont(size=18, weight="bold")).pack(side="left")
        ctk.CTkButton(
            top_bar, text="Logout", width=100, fg_color="#dc3545", hover_color="#c82333",
            command=lambda: controller.show_frame("MainMenuPage")
        ).pack(side="right")

        # Actions container
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(expand=True)

        # Register button
        reg_btn = ctk.CTkButton(
            container, text="🧑‍🎓\nRegister New Student", width=220, height=120,
            font=ctk.CTkFont(size=16, weight="bold"),
            command=lambda: controller.show_frame("RegistrationPage")
        )
        reg_btn.pack(side="left", padx=20)

        # Manage button
        manage_btn = ctk.CTkButton(
            container, text="⚙️\nManage Students", width=220, height=120,
            font=ctk.CTkFont(size=16, weight="bold"),
            command=lambda: controller.show_frame("StudentManagementPage")
        )
        manage_btn.pack(side="left", padx=20)

        # Export button
        export_btn = ctk.CTkButton(
            container, text="📤\nExport Attendance", width=220, height=120,
            font=ctk.CTkFont(size=16, weight="bold"),
            command=lambda: controller.show_frame("ExportAttendancePage")
        )
        export_btn.pack(side="left", padx=20)


class StudentManagementPage(ctk.CTkFrame):
    """View, search, edit, and delete students."""
    def __init__(self, parent, controller: AttendanceApp):
        super().__init__(parent, corner_radius=0)
        self.controller = controller

        # Top bar
        top_bar = ctk.CTkFrame(self, fg_color="transparent")
        top_bar.pack(fill="x", padx=20, pady=(14, 10))
        ctk.CTkButton(
            top_bar, text="← Back to Dashboard", width=150, fg_color="transparent", border_width=1,
            command=lambda: controller.show_frame("AdminDashboardPage")
        ).pack(side="left")
        ctk.CTkLabel(top_bar, text="Student Management", font=ctk.CTkFont(size=18, weight="bold")).pack(side="right")

        # Toolbar (Search + Actions)
        toolbar = ctk.CTkFrame(self, fg_color="transparent")
        toolbar.pack(fill="x", padx=20, pady=10)

        self.search_entry = ctk.CTkEntry(toolbar, width=250, placeholder_text="Search by Name or Roll No...")
        self.search_entry.pack(side="left", padx=(0, 10))
        self.search_entry.bind("<KeyRelease>", lambda e: self._refresh_data())

        ctk.CTkButton(toolbar, text="Search", width=80, command=self._refresh_data).pack(side="left")

        ctk.CTkButton(
            toolbar, text="Delete Selected", width=120, fg_color="#dc3545", hover_color="#c82333",
            command=self._delete_student
        ).pack(side="right", padx=(10, 0))

        ctk.CTkButton(
            toolbar, text="Edit Selected", width=120,
            command=self._edit_student
        ).pack(side="right")

        # Table (ttk.Treeview)
        # We use a standard tkinter Treeview inside a CTkFrame because CTk doesn't have a native table yet
        from tkinter import ttk
        table_frame = ctk.CTkFrame(self)
        table_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", background="#2b2b2b", foreground="white", rowheight=30, fieldbackground="#2b2b2b", borderwidth=0)
        style.map('Treeview', background=[('selected', '#1f538d')])
        style.configure("Treeview.Heading", background="#333333", foreground="white", relief="flat", font=('Helvetica', 11, 'bold'))
        style.map("Treeview.Heading", background=[('active', '#3e3e3e')])

        self.tree = ttk.Treeview(table_frame, columns=("Roll No", "Name", "Class"), show="headings", selectmode="browse")
        self.tree.heading("Roll No", text="Roll No")
        self.tree.heading("Name", text="Name")
        self.tree.heading("Class", text="Class / Section")

        self.tree.column("Roll No", width=150, anchor="center")
        self.tree.column("Name", width=300, anchor="w")
        self.tree.column("Class", width=150, anchor="center")

        scroll_y = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll_y.set)

        self.tree.pack(fill="both", expand=True, side="left")
        scroll_y.pack(side="right", fill="y")

    def on_show(self):
        self.search_entry.delete(0, "end")
        self._refresh_data()

    def _refresh_data(self):
        query = self.search_entry.get().strip()
        if query:
            students = self.controller.db.search_students(query)
        else:
            students = self.controller.db.get_all_students()

        # Clear existing
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Insert new
        for s in students:
            self.tree.insert("", "end", values=(s["roll_no"], s["name"], s["class_name"]))

    def _get_selected(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a student from the list.")
            return None
        return self.tree.item(selected[0])["values"]

    def _edit_student(self):
        student = self._get_selected()
        if not student: return
        roll_no, name, class_name = student
        
        # Open edit dialog
        dialog = EditStudentDialog(self, roll_no, name, class_name)
        self.wait_window(dialog)
        self._refresh_data()

    def _delete_student(self):
        student = self._get_selected()
        if not student: return
        roll_no, name, _ = student

        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to completely remove\n{name} ({roll_no})\nand their face data? This cannot be undone.", icon="warning"):
            if self.controller.db.delete_student(roll_no):
                messagebox.showinfo("Success", "Student deleted successfully.")
                self._refresh_data()
            else:
                messagebox.showerror("Error", "Failed to delete student.")


class EditStudentDialog(ctk.CTkToplevel):
    def __init__(self, parent: StudentManagementPage, roll_no: str, name: str, class_name: str):
        super().__init__(parent)
        self.parent_page = parent
        self.roll_no = roll_no
        self.title("Edit Student")
        self.geometry("400x320")
        self.resizable(False, False)
        # Make modal
        self.transient(parent.winfo_toplevel())
        self.grab_set()

        ctk.CTkLabel(self, text=f"Edit Student: {roll_no}", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(20, 20))

        form = ctk.CTkFrame(self, fg_color="transparent")
        form.pack(fill="both", expand=True, padx=30)

        ctk.CTkLabel(form, text="Full Name", anchor="w").pack(fill="x")
        self.name_entry = ctk.CTkEntry(form)
        self.name_entry.insert(0, name)
        self.name_entry.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(form, text="Class / Section", anchor="w").pack(fill="x")
        self.class_entry = ctk.CTkEntry(form)
        self.class_entry.insert(0, class_name)
        self.class_entry.pack(fill="x", pady=(0, 20))

        btn_frame = ctk.CTkFrame(form, fg_color="transparent")
        btn_frame.pack(fill="x")
        
        ctk.CTkButton(btn_frame, text="Cancel", width=100, fg_color="gray", hover_color="#555555", command=self.destroy).pack(side="left")
        ctk.CTkButton(btn_frame, text="Save Changes", width=120, command=self._save).pack(side="right")
        
    def _save(self):
        new_name = self.name_entry.get().strip()
        new_class = self.class_entry.get().strip()
        
        if not new_name or not new_class:
            messagebox.showerror("Error", "Fields cannot be empty.", parent=self)
            return

        if self.parent_page.controller.db.update_student(self.roll_no, new_name, new_class):
            self.destroy()
        else:
            messagebox.showerror("Error", "Failed to update database.", parent=self)

class ExportAttendancePage(ctk.CTkFrame):
    """Export attendance records for a date range to a CSV file."""

    def __init__(self, parent, controller: AttendanceApp):
        super().__init__(parent, corner_radius=0)
        self.controller = controller

        # Top bar
        top_bar = ctk.CTkFrame(self, fg_color="transparent")
        top_bar.pack(fill="x", padx=20, pady=(14, 0))
        ctk.CTkButton(
            top_bar, text="← Back to Dashboard", width=150,
            fg_color="transparent", border_width=1,
            command=lambda: controller.show_frame("AdminDashboardPage"),
        ).pack(side="left")
        ctk.CTkLabel(
            top_bar, text="Export Attendance", font=ctk.CTkFont(size=18, weight="bold")
        ).pack(side="right")

        # Form
        form = ctk.CTkFrame(self, width=460, corner_radius=16)
        form.pack(expand=True, pady=20)
        form.pack_propagate(False)

        ctk.CTkLabel(
            form, text="📤  Export to CSV", font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(28, 20))

        row_frame = ctk.CTkFrame(form, fg_color="transparent")
        row_frame.pack(padx=30, fill="x")

        # Start date
        start_col = ctk.CTkFrame(row_frame, fg_color="transparent")
        start_col.pack(side="left", expand=True, fill="x", padx=(0, 10))
        ctk.CTkLabel(start_col, text="Start Date (YYYY-MM-DD)", anchor="w").pack(fill="x")
        self.start_entry = ctk.CTkEntry(start_col, placeholder_text="e.g. 2024-01-01")
        self.start_entry.pack(fill="x", pady=(2, 0))

        # End date
        end_col = ctk.CTkFrame(row_frame, fg_color="transparent")
        end_col.pack(side="left", expand=True, fill="x")
        ctk.CTkLabel(end_col, text="End Date (YYYY-MM-DD)", anchor="w").pack(fill="x")
        self.end_entry = ctk.CTkEntry(end_col, placeholder_text="e.g. 2024-12-31")
        self.end_entry.pack(fill="x", pady=(2, 0))

        self.status_label = ctk.CTkLabel(
            form, text="", font=ctk.CTkFont(size=12), wraplength=380
        )
        self.status_label.pack(pady=(14, 0))

        ctk.CTkButton(
            form, text="Choose File & Export", width=300, height=42,
            corner_radius=10, fg_color="#28a745", hover_color="#218838",
            command=self._export,
        ).pack(padx=30, pady=(12, 28))

    def on_show(self):
        today = date.today().isoformat()
        self.start_entry.delete(0, "end")
        self.start_entry.insert(0, today)
        self.end_entry.delete(0, "end")
        self.end_entry.insert(0, today)
        self.status_label.configure(text="", text_color="white")

    def _export(self):
        start_str = self.start_entry.get().strip()
        end_str = self.end_entry.get().strip()

        # Validate dates
        try:
            start_date = date.fromisoformat(start_str)
            end_date = date.fromisoformat(end_str)
        except ValueError:
            self.status_label.configure(
                text="❌ Invalid date format. Use YYYY-MM-DD.", text_color="red"
            )
            return

        if start_date > end_date:
            self.status_label.configure(
                text="❌ Start date must be on or before end date.", text_color="red"
            )
            return

        # Ask user where to save the file
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"attendance_{start_str}_to_{end_str}.csv",
            title="Save Attendance CSV",
        )
        if not filepath:
            return  # user cancelled

        records = self.controller.db.get_attendance_by_date_range(start_date, end_date)

        try:
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Roll No", "Name", "Class", "Date", "Time"])
                for rec in records:
                    writer.writerow([
                        rec["roll_no"],
                        rec["name"],
                        rec["class_name"],
                        str(rec["date"]),
                        str(rec["time"])[:8],
                    ])
        except OSError as e:
            self.status_label.configure(
                text=f"❌ Could not write file: {e}", text_color="red"
            )
            return

        self.status_label.configure(
            text=f"✅ Exported {len(records)} record(s) to:\n{filepath}",
            text_color="#28a745",
        )


# ══════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = AttendanceApp()
    app.mainloop()
