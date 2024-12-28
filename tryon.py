import cv2
import mediapipe as mp
import numpy as np
import os
import torch
from torchvision import transforms
from typing import List, Tuple
import math


class AdvancedHairstyleTryOn:
    def __init__(self, hairstyles_dir: str):
        """
        Initialize Advanced Hairstyle Try-On system with deep learning and AR enhancements.

        :param hairstyles_dir: Directory containing hairstyle images with transparency.
        """
        # Load Mediapipe Face Mesh for accurate landmark detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.9,
            min_tracking_confidence=0.9
        )

        # Load and preprocess hairstyle images
        self.hairstyles = self._load_hairstyles(hairstyles_dir)

        # Transformation pipeline for deep learning models
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_hairstyles(self, directory: str) -> List[np.ndarray]:
        """
        Load hairstyle images from the directory.

        :param directory: Directory containing hairstyle images.
        :return: List of preprocessed hairstyle images.
        """
        hairstyles = []
        for filename in os.listdir(directory):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(directory, filename)
                hairstyle = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

                if hairstyle is not None:
                    # Ensure the hairstyle has an alpha channel
                    if hairstyle.shape[2] == 3:
                        hairstyle = cv2.cvtColor(hairstyle, cv2.COLOR_BGR2BGRA)

                    # Enhance edges for blending
                    hairstyle = self._refine_hairstyle_edges(hairstyle)
                    hairstyles.append(hairstyle)
        return hairstyles

    def _refine_hairstyle_edges(self, hairstyle: np.ndarray) -> np.ndarray:
        """
        Refine hairstyle edges for better blending.

        :param hairstyle: Hairstyle image with alpha channel.
        :return: Refined hairstyle.
        """
        alpha = hairstyle[:, :, 3]
        alpha = cv2.GaussianBlur(alpha, (5, 5), sigmaX=2, sigmaY=2)
        hairstyle[:, :, 3] = alpha
        return hairstyle

    def _get_forehead_roi(self, landmarks: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        """
        Get the Region of Interest (ROI) for placing the hairstyle.

        :param landmarks: Detected facial landmarks.
        :return: ROI coordinates.
        """
        forehead_points = [
            landmarks[10],  # Center forehead
            landmarks[234],  # Left forehead
            landmarks[454],  # Right forehead
        ]

        x_coords = [p[0] for p in forehead_points]
        y_coords = [p[1] for p in forehead_points]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min = min(y_coords) - 70
        y_max = min(y_coords) + 50

        return x_min, y_min, x_max, y_max

    def _overlay_hairstyle(self, frame: np.ndarray, hairstyle: np.ndarray,
                           roi: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Overlay hairstyle on the video frame with smooth blending.

        :param frame: Original frame.
        :param hairstyle: Selected hairstyle image.
        :param roi: Region of Interest for overlay.
        :return: Frame with hairstyle overlay.
        """
        x_min, y_min, x_max, y_max = roi
        w, h = x_max - x_min, y_max - y_min

        # Resize the hairstyle to fit the ROI
        resized_hairstyle = cv2.resize(hairstyle, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # Extract alpha channel for blending
        alpha_mask = resized_hairstyle[:, :, 3] / 255.0
        hairstyle_rgb = resized_hairstyle[:, :, :3]

        # Perform blending
        roi_frame = frame[y_min:y_max, x_min:x_max]
        blended = (roi_frame * (1 - alpha_mask[:, :, np.newaxis]) +
                   hairstyle_rgb * alpha_mask[:, :, np.newaxis])

        frame[y_min:y_max, x_min:x_max] = blended.astype(np.uint8)
        return frame

    def process_frame(self, frame: np.ndarray, hairstyle_index: int) -> np.ndarray:
        """
        Process the frame to overlay the selected hairstyle.

        :param frame: Input video frame.
        :param hairstyle_index: Index of the hairstyle to apply.
        :return: Processed frame.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            landmark_points = [
                (int(landmark.x * w), int(landmark.y * h))
                for landmark in landmarks.landmark
            ]

            roi = self._get_forehead_roi(landmark_points)
            hairstyle = self.hairstyles[hairstyle_index]
            frame = self._overlay_hairstyle(frame, hairstyle, roi)

        return frame

    def run(self):
        """
        Run the virtual try-on system.
        """
        cap = cv2.VideoCapture(0)

        current_hairstyle_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            processed_frame = self.process_frame(frame, current_hairstyle_index)

            cv2.putText(processed_frame,
                        "Press 'n': Next Hairstyle, 'q': Quit",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)

            cv2.imshow('Advanced Hairstyle Try-On', processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                current_hairstyle_index = (current_hairstyle_index + 1) % len(self.hairstyles)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    HAIRSTYLES_DIR = "hairstyles"  # Replace with the path to your hairstyle images

    tryon_system = AdvancedHairstyleTryOn(HAIRSTYLES_DIR)
    tryon_system.run()
