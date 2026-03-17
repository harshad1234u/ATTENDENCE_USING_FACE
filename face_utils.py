"""
face_utils.py
─────────────
Face detection, encoding, and recognition utilities built on
top of the `face_recognition` and `opencv-python` libraries.
"""

from typing import List, Optional, Tuple

import cv2
import face_recognition
import numpy as np


class FaceRecognition:
    """Utility class for face detection, encoding, and recognition."""

    # Recognition tolerance — lower is stricter (default 0.6)
    TOLERANCE = 0.5
    # Model to use for face detection: "hog" (fast, CPU) or "cnn" (accurate, GPU)
    MODEL = "hog"

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------
    @staticmethod
    def capture_face_encoding(
        frame: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], str]:
        """Detect faces in a frame and return the encoding if exactly one
        face is found.

        Parameters
        ----------
        frame : np.ndarray
            BGR image captured from the webcam.

        Returns
        -------
        encoding : np.ndarray or None
            128-d face encoding, or None on failure.
        location : tuple(top, right, bottom, left) or None
            Pixel coordinates of the face bounding box.
        message : str
            Human-readable status string.
        """
        # Convert BGR → RGB (face_recognition expects RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face locations
        locations = face_recognition.face_locations(rgb, model=FaceRecognition.MODEL)

        if len(locations) == 0:
            return None, None, "No face detected. Please look at the camera."
        if len(locations) > 1:
            return (
                None,
                None,
                f"{len(locations)} faces detected. Only one person should be in frame.",
            )

        # Exactly one face — compute encoding
        encodings = face_recognition.face_encodings(rgb, locations)
        return encodings[0], locations[0], "Face captured successfully!"

    # ------------------------------------------------------------------
    # Live recognition helpers
    # ------------------------------------------------------------------
    @staticmethod
    def recognize_faces(
        frame: np.ndarray,
        known_encodings: List[np.ndarray],
        known_metadata: List[dict],
    ) -> List[dict]:
        """Recognise faces in a live frame against the known database.

        Parameters
        ----------
        frame : np.ndarray
            BGR image from the webcam.
        known_encodings : list of np.ndarray
            List of 128-d face encodings from the database.
        known_metadata : list of dict
            Parallel list with keys 'roll_no' and 'name' for each encoding.

        Returns
        -------
        list of dict
            Each dict: {name, roll_no, location} for every detected face.
            Unknown faces return name="Unknown" and roll_no=None.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Downsample for speed (process at ¼ resolution)
        small_rgb = cv2.resize(rgb, (0, 0), fx=0.25, fy=0.25)

        locations = face_recognition.face_locations(small_rgb, model=FaceRecognition.MODEL)
        encodings = face_recognition.face_encodings(small_rgb, locations)

        results = []
        for encoding, location in zip(encodings, locations):
            name = "Unknown"
            roll_no = None

            if known_encodings:
                # Compare against all known faces
                distances = face_recognition.face_distance(known_encodings, encoding)
                best_idx = int(np.argmin(distances))

                if distances[best_idx] <= FaceRecognition.TOLERANCE:
                    name = known_metadata[best_idx]["name"]
                    roll_no = known_metadata[best_idx]["roll_no"]

            # Scale location back to original size (×4)
            top, right, bottom, left = [v * 4 for v in location]

            results.append(
                {
                    "name": name,
                    "roll_no": roll_no,
                    "location": (top, right, bottom, left),
                }
            )

        return results

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    @staticmethod
    def draw_results(frame: np.ndarray, results: List[dict]) -> np.ndarray:
        """Draw bounding boxes and labels on the frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image to annotate.
        results : list of dict
            Output from `recognize_faces`.

        Returns
        -------
        np.ndarray
            Annotated frame (same object, mutated in-place).
        """
        for r in results:
            top, right, bottom, left = r["location"]

            if r["roll_no"]:
                # Known student — green box
                color = (0, 200, 0)
                label = f"{r['name']} ({r['roll_no']})"
            else:
                # Unknown face — red box
                color = (0, 0, 255)
                label = "Unknown"

            # Bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Label background
            cv2.rectangle(
                frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED
            )
            cv2.putText(
                frame,
                label,
                (left + 6, bottom - 8),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

        return frame
