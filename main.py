from tkinter import *
from tkinter import messagebox
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import threading
import google.generativeai as genai
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import io
import base64
import subprocess  # For running tryon.py
from typing import List, Tuple
import subprocess
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk


import json

class AdvancedHairstyleTryOnF:
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

        # Initialize adjustment variables with default values
        self.width_multiplier = 1.0
        self.height_multiplier = 1.0
        self.x_offset = 0
        self.y_offset = 0

        # Try to load saved settings if they exist
        try:
            with open('hairstyle_settings.json', 'r') as f:
                settings = json.load(f)
                self.width_multiplier = settings['width_multiplier']
                self.height_multiplier = settings['height_multiplier']
                self.x_offset = settings['x_offset']
                self.y_offset = settings['y_offset']
        except FileNotFoundError:
            # If no saved settings exist, keep the default values
            pass

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

        # Resize the hairstyle to fit the ROI dynamically
        resized_hairstyle = cv2.resize(hairstyle, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # Extract alpha channel for blending
        alpha_mask = resized_hairstyle[:, :, 3] / 255.0
        hairstyle_rgb = resized_hairstyle[:, :, :3]

        # Extract the region of interest (ROI) from the frame, ensuring boundaries are valid
        roi_frame = frame[max(0, y_min):min(frame.shape[0], y_max),
                    max(0, x_min):min(frame.shape[1], x_max)]

        # Ensure ROI frame dimensions match resized hairstyle
        roi_h, roi_w = roi_frame.shape[:2]
        resized_hairstyle = cv2.resize(resized_hairstyle, (roi_w, roi_h), interpolation=cv2.INTER_LANCZOS4)
        alpha_mask = resized_hairstyle[:, :, 3] / 255.0
        hairstyle_rgb = resized_hairstyle[:, :, :3]

        # Perform blending
        blended = (roi_frame * (1 - alpha_mask[:, :, np.newaxis]) +
                   hairstyle_rgb * alpha_mask[:, :, np.newaxis])

        frame[max(0, y_min):min(frame.shape[0], y_max),
        max(0, x_min):min(frame.shape[1], x_max)] = blended.astype(np.uint8)
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

            roi = self._get_hairline_roi(landmark_points)
            hairstyle = self.hairstyles[hairstyle_index]
            frame = self._overlay_hairstyle(frame, hairstyle, roi)

        return frame

    def _get_hairline_roi(self, landmarks: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        """
        Get the Region of Interest (ROI) for placing the hairstyle dynamically.
        Width increases normally, height increases 2x during size adjustment.
        """
        # Points from the face landmarks
        left_side = landmarks[234]
        right_side = landmarks[454]
        top_mid = landmarks[10]  # Forehead midpoint
        bottom_mid = landmarks[152]  # Chin center

        # Calculate the dimensions for the head
        head_width = int(math.dist((left_side[0], left_side[1]), (right_side[0], right_side[1])))
        head_height = int(math.dist((top_mid[0], top_mid[1]), (bottom_mid[0], bottom_mid[1])))

        # Add adjustment variables with separate multipliers for width and height
        self.width_multiplier = getattr(self, 'width_multiplier', 1.0)
        self.height_multiplier = getattr(self, 'height_multiplier', 1.0)
        self.x_offset = getattr(self, 'x_offset', 0)
        self.y_offset = getattr(self, 'y_offset', 0)

        # Calculate adjusted ROI dimensions
        # Width increases normally
        hairline_x_min = int(left_side[0] - head_width * 0.5 * self.width_multiplier) + self.x_offset
        hairline_x_max = int(right_side[0] + head_width * 0.5 * self.width_multiplier) + self.x_offset

        # Height increases with double multiplier (2x effect on height)
        hairline_y_min = int(top_mid[1] - head_height * 0.2) + self.y_offset  # Keep top margin fixed
        hairline_y_max = int(bottom_mid[1] + head_height * 2.0 * self.height_multiplier) + self.y_offset

        return hairline_x_min, hairline_y_min, hairline_x_max, hairline_y_max

    def run(self):
        """
        Run the virtual try-on system with keyboard controls.
        Width increases 1x, height increases 2x when pressing 'b'.
        """
        cap = cv2.VideoCapture(0)
        current_hairstyle_index = 0

        # Initialize adjustment variables if not already set
        if not hasattr(self, 'width_multiplier'):
            self.width_multiplier = 1.0
        if not hasattr(self, 'height_multiplier'):
            self.height_multiplier = 1.0
        if not hasattr(self, 'x_offset'):
            self.x_offset = 0
        if not hasattr(self, 'y_offset'):
            self.y_offset = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            processed_frame = self.process_frame(frame, current_hairstyle_index)

            # Display controls
            controls = [
                "b: Bigger (2x height, 1x width)",
                "v: Smaller",
                "l: Move Left",
                "r: Move Right",
                "u: Move Up",
                "d: Move Down",
                "s: Save",
                "n: Next Style",
                "q: Quit"
            ]

            y = 30
            for text in controls:
                cv2.putText(processed_frame, text, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y += 25

            cv2.imshow('Advanced Hairstyle Try-On', processed_frame)

            key = cv2.waitKey(1) & 0xFF

            # Handle keyboard controls
            if key == ord('q'):
                break
            elif key == ord('b'):  # Bigger - height increases 2x compared to width
                self.width_multiplier += 0.1  # Normal width increase
                self.height_multiplier += 0.2  # Double height increase
            elif key == ord('v'):  # Smaller
                self.width_multiplier = max(0.1, self.width_multiplier - 0.1)
                self.height_multiplier = max(0.1, self.height_multiplier - 0.2)
            elif key == ord('l'):  # Left
                self.x_offset -= 10
            elif key == ord('r'):  # Right
                self.x_offset += 10
            elif key == ord('u'):  # Up
                self.y_offset -= 10
            elif key == ord('d'):  # Down
                self.y_offset += 10
            elif key == ord('s'):  # Save settings
                with open('hairstyle_settings.json', 'w') as f:
                    json.dump({
                        'width_multiplier': self.width_multiplier,
                        'height_multiplier': self.height_multiplier,
                        'x_offset': self.x_offset,
                        'y_offset': self.y_offset
                    }, f)
            elif key == ord('n'):  # Next hairstyle
                current_hairstyle_index = (current_hairstyle_index + 1) % len(self.hairstyles)

        cap.release()
        cv2.destroyAllWindows()


# Global state management
class GlobalState:
    def __init__(self):
        self.captured_image_path = None
        self.is_logged_in = False

# Create global state instance
global_state = GlobalState()

# Google Gemini API Configuration
genai.configure(api_key="AIzaSyBK48Ys3UfJrIU1X1dOuYboXd0APpFhaIw")
model = genai.GenerativeModel("gemini-1.5-flash")


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

        # Position and size adjustment parameters
        self.offset_x = 0
        self.offset_y = 0
        self.scale_factor = 1.0

        # Dictionary to store saved configurations for each hairstyle
        self.saved_configs = {}

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

    def _get_hairline_roi(self, landmarks: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        """
        Get the Region of Interest (ROI) for placing the hairstyle dynamically above the forehead,
        accounting for user adjustments.

        :param landmarks: Detected facial landmarks.
        :return: ROI coordinates for hairline placement.
        """
        # Points from the hairline
        left_side = landmarks[234]
        right_side = landmarks[454]
        top_mid = landmarks[10]
        bottom_mid = landmarks[152]

        # Calculate the dimensions for the hairline ROI dynamically
        head_width = int(math.dist((left_side[0], left_side[1]), (right_side[0], right_side[1])))
        head_height = int(math.dist((top_mid[0], top_mid[1]), (bottom_mid[0], bottom_mid[1])))

        # Apply scale factor and offsets to the ROI
        scaled_width = head_width * self.scale_factor
        scaled_height = head_height * self.scale_factor

        hairline_x_min = int(left_side[0] - scaled_width * 0.3) + self.offset_x
        hairline_y_min = int(top_mid[1] - scaled_height * 0.4) + self.offset_y
        hairline_x_max = int(right_side[0] + scaled_width * 0.3) + self.offset_x
        hairline_y_max = int(top_mid[1] + scaled_height * 0.5) + self.offset_y

        return hairline_x_min, hairline_y_min, hairline_x_max, hairline_y_max

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

        # Ensure valid dimensions
        if w <= 0 or h <= 0:
            return frame

        # Clip ROI to frame boundaries
        y_min = max(0, y_min)
        y_max = min(frame.shape[0], y_max)
        x_min = max(0, x_min)
        x_max = min(frame.shape[1], x_max)

        # Resize the hairstyle to fit the ROI dynamically
        try:
            resized_hairstyle = cv2.resize(hairstyle, (x_max - x_min, y_max - y_min),
                                           interpolation=cv2.INTER_LANCZOS4)
        except cv2.error:
            return frame

        # Extract alpha channel for blending
        alpha_mask = resized_hairstyle[:, :, 3] / 255.0
        hairstyle_rgb = resized_hairstyle[:, :, :3]

        # Perform blending
        roi_frame = frame[y_min:y_max, x_min:x_max]
        if roi_frame.shape[:2] != alpha_mask.shape[:2]:
            return frame

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

            roi = self._get_hairline_roi(landmark_points)
            hairstyle = self.hairstyles[hairstyle_index]
            frame = self._overlay_hairstyle(frame, hairstyle, roi)

        return frame

    def save_current_config(self, hairstyle_index: int):
        """
        Save the current position and scale configuration for the current hairstyle.
        """
        self.saved_configs[hairstyle_index] = {
            'offset_x': self.offset_x,
            'offset_y': self.offset_y,
            'scale_factor': self.scale_factor
        }

    def load_config(self, hairstyle_index: int):
        """
        Load saved configuration for the current hairstyle if it exists.
        """
        if hairstyle_index in self.saved_configs:
            config = self.saved_configs[hairstyle_index]
            self.offset_x = config['offset_x']
            self.offset_y = config['offset_y']
            self.scale_factor = config['scale_factor']

    def run(self):
        """
        Run the virtual try-on system with enhanced controls.
        """
        cap = cv2.VideoCapture(0)
        current_hairstyle_index = 0

        # Load any saved configuration for the first hairstyle
        self.load_config(current_hairstyle_index)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            processed_frame = self.process_frame(frame, current_hairstyle_index)

            # Display controls
            controls_text = [
                "Controls:",
                "n: Next Hairstyle",
                "l/r: Left/Right",
                "u/d: Up/Down",
                "b/v: Bigger/Smaller",
                "s: Save Config",
                "q: Quit"
            ]

            y_position = 30
            for text in controls_text:
                cv2.putText(processed_frame, text, (10, y_position),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_position += 25

            cv2.imshow('Advanced Hairstyle Try-On', processed_frame)

            key = cv2.waitKey(1) & 0xFF

            # Handle key controls
            if key == ord('q'):
                break
            elif key == ord('n'):
                current_hairstyle_index = (current_hairstyle_index + 1) % len(self.hairstyles)
                self.load_config(current_hairstyle_index)
            elif key == ord('l'):
                self.offset_x -= 5
            elif key == ord('r'):
                self.offset_x += 5
            elif key == ord('u'):
                self.offset_y -= 5
            elif key == ord('d'):
                self.offset_y += 5
            elif key == ord('b'):
                self.scale_factor += 0.1
            elif key == ord('v'):
                self.scale_factor = max(0.1, self.scale_factor - 0.1)
            elif key == ord('s'):
                self.save_current_config(current_hairstyle_index)

        cap.release()
        cv2.destroyAllWindows()

class ViewData:
    def __init__(self):

        def face():
            """
            Classifies hair length from a captured image using a pre-trained CNN model.
            Handles image preprocessing and displays classification results.
            """
            # Check if model file exists
            model_path = os.path.join('models', 'cnnCat2.h5')
            if not os.path.exists(model_path):
                messagebox.showerror("Error", "Model file not found. Please ensure 'models/cnnCat2.h5' exists.")
                return

            # Check if an image has been captured
            if not global_state.captured_image_path:
                messagebox.showwarning("Warning", "Please capture an image first!")
                return

            try:
                # Load the pre-trained model
                model = load_model(model_path)

                # Load and preprocess the captured image
                with Image.open(global_state.captured_image_path) as image:
                    # Resize to match the model's input size
                    image = image.resize((224, 224))
                    image_array = np.array(image)

                    # Handle different image channels
                    if len(image_array.shape) == 2:  # Grayscale
                        image_array = np.stack((image_array,) * 3, axis=-1)
                    elif len(image_array.shape) == 3:
                        if image_array.shape[2] == 4:  # RGBA
                            image_array = image_array[:, :, :3]  # Convert to RGB
                        elif image_array.shape[2] != 3:  # Invalid number of channels
                            raise ValueError(f"Unexpected number of channels: {image_array.shape[2]}")

                    # Normalize pixel values to [0, 1]
                    image_array = image_array.astype(np.float32) / 255.0

                    # Add batch dimension
                    image_array = np.expand_dims(image_array, axis=0)

                # Make prediction
                predictions = model.predict(image_array, verbose=0)
                # Fix: Use the correct indexing for predictions
                predicted_class = np.argmax(predictions[0])  # Changed from axis=1

                # Map prediction to label
                class_labels = {
                    0: "Short Hair",
                    1: "Medium Hair",
                    2: "Long Hair"
                }
                prediction_label = class_labels.get(predicted_class, "Unknown")

                # Calculate confidence score - Fixed indexing
                confidence = float(predictions[0][predicted_class]) * 100

                # Display the result with confidence
                result_message = (
                    f"Predicted Hairstyle: {prediction_label}\n"
                    f"Confidence: {confidence:.2f}%"
                )
                messagebox.showinfo("Hair Classifier Result", result_message)

            except ImportError as e:
                messagebox.showerror(
                    "Error",
                    "Required libraries not installed. Please install tensorflow using: pip install tensorflow"
                )
                print(f"Import error details: {str(e)}")

            except FileNotFoundError as e:
                messagebox.showerror(
                    "Error",
                    f"Image file not found: {str(e)}"
                )

            except ValueError as e:
                messagebox.showerror(
                    "Error",
                    f"Invalid image format: {str(e)}"
                )

            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"Unexpected error: {str(e)}"
                )
                print(f"Detailed error: {str(e)}")  # For debugging

        def extraction():
            threading.Thread(target=capture_image).start()

        def capture_image():
            cap = cv2.VideoCapture(0)
            cv2.namedWindow("Capture Image")

            while True:
                ret, frame = cap.read()
                if not ret:
                    messagebox.showerror("Error", "Failed to access camera.")
                    break

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, "Press 'c' to capture", (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "Press 'q' to exit", (10, 60), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow("Capture Image", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('c'):
                    global_state.captured_image_path = "captured_image.jpg"
                    cv2.imwrite(global_state.captured_image_path, frame)
                    messagebox.showinfo("Success", "Image captured successfully!")
                    break
                elif key == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        def recommend_style():
            if not global_state.captured_image_path:
                messagebox.showwarning("Warning", "Please capture an image first!")
                return

            try:
                # Load the image for Gemini
                image = Image.open(global_state.captured_image_path)

                # Prepare the prompt
                prompt = """
                Please analyze this image and provide a comprehensive hairstyle recommendation covering:

                1. Face Shape Analysis:
                   - Identify face shape
                   - Key facial features

                2. Current Hair Analysis:
                   - Current length and texture
                   - Hair type and condition

                3. Personalized Recommendations:
                   - 2-3 specific hairstyle suggestions
                   - Why each style would suit them
                   - Length and layering recommendations

                4. Styling Tips:
                   - Daily maintenance advice
                   - Styling products to consider
                   - Tools recommended

                5. Additional Suggestions:
                   - Color recommendations if applicable
                   - Accessories that would complement
                   - Professional maintenance schedule

                Please provide detailed, practical recommendations that consider both style and maintenance.
                """

                # Generate content using the image file
                response = model.generate_content([prompt, image])
                suggestion = response.text

                # Create scrollable recommendation window
                suggestion_window = Toplevel()
                suggestion_window.title("Professional Style Recommendation")
                suggestion_window.geometry("800x600")
                suggestion_window.configure(bg='#f5f5f5')

                # Create main frame with scrollbar
                main_frame = Frame(suggestion_window, bg='#f5f5f5')
                main_frame.pack(fill=BOTH, expand=1, padx=20, pady=20)

                # Add scrollbar
                canvas = Canvas(main_frame, bg='#f5f5f5')
                scrollbar = Scrollbar(main_frame, orient=VERTICAL, command=canvas.yview)
                scrollable_frame = Frame(canvas, bg='#f5f5f5')

                scrollable_frame.bind(
                    "<Configure>",
                    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
                )

                canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
                canvas.configure(yscrollcommand=scrollbar.set)

                # Add title and content
                Label(scrollable_frame,
                      text="Your Personalized Hairstyle Analysis",
                      font="Helvetica 16 bold",
                      bg='#f5f5f5',
                      fg='#333333').pack(pady=10)

                # Add recommendation text
                text_widget = Text(scrollable_frame,
                                   wrap=WORD,
                                   width=70,
                                   height=25,
                                   font="Helvetica 11",
                                   padx=15,
                                   pady=15,
                                   bg='white')
                text_widget.insert(END, suggestion)
                text_widget.config(state=DISABLED)
                text_widget.pack(pady=10)

                # Pack scrollbar and canvas
                scrollbar.pack(side=RIGHT, fill=Y)
                canvas.pack(side=LEFT, fill=BOTH, expand=1)

                # Create button frame
                button_frame = Frame(suggestion_window, bg='#f5f5f5')
                button_frame.pack(pady=10)

                # Add Chat button
                Button(button_frame,
                       text="Open Chat",
                       command=lambda: create_chat_window(global_state.captured_image_path),
                       font="Helvetica 11 bold",
                       bg="#2196F3",
                       fg="white",
                       padx=20,
                       pady=5).pack(side=LEFT, padx=10)

                # Add Close button
                Button(button_frame,
                       text="Close",
                       command=suggestion_window.destroy,
                       font="Helvetica 11 bold",
                       bg="#4CAF50",
                       fg="white",
                       padx=20,
                       pady=5).pack(side=LEFT, padx=10)

            except Exception as e:
                messagebox.showerror("Error", f"Error generating recommendation: {str(e)}")
                print(f"Detailed error: {str(e)}")

        # Virtual Try-On function to run tryon.py
        def vm():
            try:
                # Ensure HAIRSTYLES_DIR is set to the correct directory path
                HAIRSTYLES_DIR = "hairstyles"  # Replace with the actual path if necessary

                # Assuming AdvancedHairstyleTryOn class is already defined or imported
                tryon_system = AdvancedHairstyleTryOn(HAIRSTYLES_DIR)
                tryon_system.run()

                messagebox.showinfo("Success", "Virtual Try-On completed successfully!")

            except Exception as e:
                messagebox.showerror("Error", f"Error running Virtual Try-On: {str(e)}")

        def vf():
            try:
                # Ensure HAIRSTYLES_DIR is set to the correct directory path
                HAIRSTYLES_DIR = "hf"  # Replace with the actual path if necessary

                # Assuming AdvancedHairstyleTryOn class is already defined or imported
                tryon_system = AdvancedHairstyleTryOnF(HAIRSTYLES_DIR)
                tryon_system.run()

                messagebox.showinfo("Success", "Virtual Try-On completed successfully!")

            except Exception as e:
                messagebox.showerror("Error", f"Error running Virtual Try-On: {str(e)}")

        def show_hairstyle_folders():
            """
            Display subfolders within the 'styles' directory, and allow viewing images in the selected subfolder.
            """
            import os
            from tkinter import filedialog, Listbox, Label, Toplevel, Frame, Button, LEFT, messagebox
            from PIL import Image, ImageTk

            # Directory containing hairstyle folders
            STYLES_DIR = "styles"  # Replace with the correct path if necessary

            # Create a new window for browsing folders
            folder_window = Toplevel()
            folder_window.title("Hairstyle Folders")
            folder_window.geometry("600x400")
            folder_window.configure(bg='#f5f5f5')

            Label(folder_window, text="Select the Style to view hairstyles:", font="Helvetica 12 bold",
                  bg="#f5f5f5").pack(pady=10)

            # Create a listbox to display subfolders
            listbox = Listbox(folder_window, font="Helvetica 10", width=50, height=15)
            listbox.pack(pady=10)

            # Populate the listbox with subfolders
            if os.path.exists(STYLES_DIR):
                subfolders = [f for f in os.listdir(STYLES_DIR) if os.path.isdir(os.path.join(STYLES_DIR, f))]
                for folder in subfolders:
                    listbox.insert(END, folder)
            else:
                messagebox.showerror("Error", f"Directory '{STYLES_DIR}' does not exist.")
                folder_window.destroy()
                return

            def open_subfolder(event):
                """
                Display images from the selected subfolder.
                """
                selected_folder = listbox.get(listbox.curselection())
                folder_path = os.path.join(STYLES_DIR, selected_folder)

                # Create a new window to display images
                image_window = Toplevel()
                image_window.title(f"Images in {selected_folder}")
                image_window.geometry("800x600")
                image_window.configure(bg='#f5f5f5')

                # Get all image files in the selected folder
                image_files = [f for f in os.listdir(folder_path) if
                               f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

                if not image_files:
                    messagebox.showinfo("No Images", f"No images found in folder '{selected_folder}'.")
                    image_window.destroy()
                    return

                # Display images one by one
                current_index = [0]  # Use a mutable object to allow updating index

                def show_image():
                    """
                    Show the current image in the folder with proper aspect ratio.
                    """
                    try:
                        img_path = os.path.join(folder_path, image_files[current_index[0]])
                        print(f"Loading image from: {img_path}")
                        img = Image.open(img_path)
                        img = img.convert("RGBA")  # Ensures proper handling of PNG transparency

                        # Resize while maintaining aspect ratio
                        max_size = (840, 560)  # Increased display size by 40%
                        img = img.resize((int(img.size[0] * 1.4), int(img.size[1] * 1.4)), Image.Resampling.LANCZOS)

                        # Create a PhotoImage object for display
                        img_tk = ImageTk.PhotoImage(img)

                        # Update the label with the new image
                        img_label.config(image=img_tk)
                        img_label.image = img_tk  # Keep a reference to avoid garbage collection

                        # Remove file extension and update the title
                        file_name = os.path.splitext(image_files[current_index[0]])[0]
                        img_title.config(text=f"{file_name} ({current_index[0] + 1}/{len(image_files)})")
                    except Exception as e:
                        print(f"Error loading image: {e}")
                        messagebox.showerror("Image Load Error", f"Unable to load image: {e}")

                def next_image():
                    """
                    Show the next image in the folder.
                    """
                    if current_index[0] < len(image_files) - 1:
                        current_index[0] += 1
                        show_image()

                def prev_image():
                    """
                    Show the previous image in the folder.
                    """
                    if current_index[0] > 0:
                        current_index[0] -= 1
                        show_image()

                # Add image display components
                img_label = Label(image_window, bg='#f5f5f5')
                img_label.pack(pady=10)

                img_title = Label(image_window, text="", font="Helvetica 10", bg='#f5f5f5')
                img_title.pack(pady=5)

                # Add navigation buttons
                nav_frame = Frame(image_window, bg='#f5f5f5')
                nav_frame.pack(pady=10)

                Button(nav_frame, text="Previous", command=prev_image, font="Helvetica 10", bg="#2196F3",
                       fg="white").pack(side=LEFT, padx=10)
                Button(nav_frame, text="Next", command=next_image, font="Helvetica 10", bg="#4CAF50", fg="white").pack(
                    side=LEFT, padx=10)

                # Show the first image initially
                show_image()

            # Bind double-click on the listbox to open the subfolder
            listbox.bind("<Double-1>", open_subfolder)

        # Main window setup
        self.ws = tk.Toplevel()
        self.ws.title("COIFFURE")
        self.ws.maxsize(width=1100, height=800)
        self.ws.minsize(width=1100, height=800)
        self.ws.configure(bg='#99ddff')

        # Prevent window from being destroyed when clicking X
        self.ws.protocol("WM_DELETE_WINDOW", lambda: self.ws.iconify())

        # Load and set background image
        image1 = Image.open("4.jpg")
        img = image1.resize((1100, 800))
        self.test = ImageTk.PhotoImage(img)
        label1 = tk.Label(self.ws, image=self.test)
        label1.place(x=1, y=1)

        # Add UI elements
        window_width = 1000
        spacing = window_width // 6  # Dividing the window into 6 equal sections

        # Label centered at the top
        tk.Label(self.ws, text='COIFFURE', bg="#ffb366", font='verdana 15 bold').place(x=450, y=120)

        # Buttons evenly spaced
        tk.Button(self.ws, text="Virtual Try-On Male", font='Verdana 10 bold', bg="#ffcc00", command=vm).place(
            x=spacing, y=230)
        tk.Button(self.ws, text="Hairstyles", font='Verdana 10 bold', bg="#ffcc00",
                  command=show_hairstyle_folders).place(x=spacing * 2, y=230)
        tk.Button(self.ws, text="Capture Process", font='Verdana 10 bold', bg="#99ff99", command=extraction).place(
            x=spacing * 3, y=230)
        tk.Button(self.ws, text="Style Recommend", font='Verdana 10 bold', bg="#99ff99", command=recommend_style).place(
            x=spacing * 4, y=230)
        tk.Button(self.ws, text="Virtual Try-On Female", font='Verdana 10 bold', bg="#ffcc00", command=vf).place(
            x=spacing * 5, y=230)

        self.ws.mainloop()



def create_chat_window(image_path):
    chat_window = Toplevel()
    chat_window.title("Hairstyle Chat Assistant")
    chat_window.geometry("1000x700")
    chat_window.configure(bg='#f5f5f5')

    # Create split layout
    right_frame = Frame(chat_window, bg='#f5f5f5')
    right_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=10, pady=10)

    # Create chat display area
    chat_display = Text(right_frame, wrap=WORD, width=50, height=25, font=("Helvetica", 11), bg='#f5f5f5')
    chat_display.pack(fill=BOTH, expand=True, pady=(0, 10))
    chat_display.config(state=DISABLED)

    # Create input area
    input_frame = Frame(right_frame, bg='#f5f5f5')
    input_frame.pack(fill=X, side=BOTTOM)

    chat_input = Text(input_frame, wrap=WORD, height=3, font=("Helvetica", 11), bg='#ffffff')
    chat_input.pack(side=LEFT, fill=X, expand=True, padx=(0, 10))

    def send_message():
        user_message = chat_input.get("1.0", END).strip()
        if user_message:
            # Clear input
            chat_input.delete("1.0", END)

            # Update chat display with user message
            chat_display.config(state=NORMAL)
            chat_display.insert(END, f"You: {user_message}\n", 'user_message')
            chat_display.insert(END, "\n")

            try:
                # Load the image using Pillow
                image = Image.open(image_path)

                # Create context-aware prompt
                prompt = f"""Based on the hairstyle image provided, please address the following question/comment:
                {user_message}

                Please provide specific advice related to the person's hair in the image."""

                # Generate the response using the image and prompt
                response = model.generate_content([prompt, image])

                assistant_response = response.text

                # Update chat display with assistant response
                chat_display.insert(END, f"Assistant: {assistant_response}\n", 'assistant_message')
                chat_display.insert(END, "\n")
                chat_display.see(END)
            except Exception as e:
                chat_display.insert(END,
                                    "Assistant: Sorry, I encountered an error processing your request. Please try again.\n\n")
                print(f"Error in chat: {str(e)}")

            chat_display.config(state=DISABLED)

    # Add send button
    send_button = Button(input_frame, text="Send", command=send_message,
                         bg="#4CAF50", fg="white", font=("Helvetica", 11, "bold"),
                         padx=20)
    send_button.pack(side=RIGHT)

    # Bind Enter key to send message
    def handle_enter(event):
        if not event.state & 0x1:  # Check if Shift is not pressed
            send_message()
            return "break"  # Prevents default behavior

    chat_input.bind("<Return>", handle_enter)

    # Define styles for user and assistant messages
    chat_display.tag_configure('user_message', foreground='blue', justify='left')
    chat_display.tag_configure('assistant_message', foreground='green', justify='left')

# Run the application
if __name__ == "__main__":
    ViewData()
