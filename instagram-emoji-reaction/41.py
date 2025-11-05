#!/usr/bin/env python3
"""
Sideways Hand Movement Detector
Detects when both hands move sideways and displays a GIF in multiple windows
"""

import cv2
import mediapipe as mp
import time
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
SIDEWAYS_THRESHOLD = 0.1  # Minimum horizontal movement to trigger (lowered for easier detection)
HISTORY_SIZE = 15  # Number of frames to track
NUM_WINDOWS = 20  # How many windows to display the GIF in (increased!)
GIF_PATH = "images/did-unc-snap-unc.gif"
GIF_WINDOW_WIDTH = 300
GIF_WINDOW_HEIGHT = 225
GIF_FPS = 15  # Playback speed for GIF
WINDOW_OPEN_DELAY = 0.05  # Delay between opening each window (seconds)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Hand position history for tracking movement
left_hand_history = []
right_hand_history = []

# Two hands detection timing
two_hands_detected_time = None
TWO_HANDS_DELAY = 1  # Wait 0.5 seconds after detecting 2 hands

# Cooldown to prevent multiple triggers
last_trigger_time = 0
COOLDOWN_SECONDS = 3

# GIF display state
gif_windows_active = False
gif_frames = []
gif_window_names = []

def load_gif_frames(gif_path):
    """Load all frames from a GIF file"""
    gif = Image.open(gif_path)
    frames = []
    try:
        while True:
            frame = gif.convert('RGB')
            frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            frame_resized = cv2.resize(frame_cv, (GIF_WINDOW_WIDTH, GIF_WINDOW_HEIGHT))
            frames.append(frame_resized)
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    return frames

def create_gif_windows(num_windows):
    """Create multiple windows for displaying GIFs quickly"""
    window_names = []
    for i in range(num_windows):
        window_name = f'GIF Window {i+1}'
        window_names.append(window_name)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, GIF_WINDOW_WIDTH, GIF_WINDOW_HEIGHT)
        # Position windows in a grid pattern (5 columns)
        x_offset = (i % 5) * (GIF_WINDOW_WIDTH + 20)
        y_offset = (i // 5) * (GIF_WINDOW_HEIGHT + 30)
        cv2.moveWindow(window_name, x_offset + 50, y_offset + 50)

    return window_names

def display_gif_in_windows(frames, window_names, start_immediately=True):
    """Display GIF frames in all windows, starting as windows appear"""
    if not frames or not window_names:
        return

    frame_delay = int(1000 / GIF_FPS)  # Delay in milliseconds

    # If starting immediately, show first frame in all windows right away
    if start_immediately:
        for window_name in window_names:
            try:
                cv2.imshow(window_name, frames[0])
            except:
                pass
        cv2.waitKey(1)

    # Play GIF 3 times in each window
    for loop in range(3):
        for frame in frames:
            for window_name in window_names:
                try:
                    cv2.imshow(window_name, frame)
                except:
                    pass  # Window might be closed

            if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
                break

def close_gif_windows(window_names):
    """Close all GIF windows"""
    for window_name in window_names:
        try:
            cv2.destroyWindow(window_name)
        except:
            pass

print("🚀 Starting Two Hands Detector...")
print("📋 Instructions:")
print("   - Show both hands to the camera")
print("   - After 1 second, the GIF will display in multiple windows!")
print("   - Press 'q' to quit")
print()

# Load GIF frames at startup
print("📂 Loading GIF...")
try:
    gif_frames = load_gif_frames(GIF_PATH)
    print(f"✅ Loaded {len(gif_frames)} frames from {GIF_PATH}")
except Exception as e:
    print(f"❌ Error loading GIF: {e}")
    exit()

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    exit()

# Create window
cv2.namedWindow('Two Hands Detector', cv2.WINDOW_NORMAL)

with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2
) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("⚠️ Ignoring empty camera frame.")
            continue

        # Flip for mirror view
        frame = cv2.flip(frame, 1)

        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Process hands
        results = hands.process(image_rgb)

        # Draw hand landmarks
        image_rgb.flags.writeable = True
        frame_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_bgr,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        # Detect two hands and trigger after delay
        current_time = time.time()

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 2:
            # Start timer when 2 hands are first detected
            if two_hands_detected_time is None:
                two_hands_detected_time = current_time
                print(f"👐 Two hands detected! Triggering in {TWO_HANDS_DELAY} second(s)...")

        # Once timer is started, keep checking even if hands are removed
        if two_hands_detected_time is not None:
            time_since_detection = current_time - two_hands_detected_time
            if time_since_detection >= TWO_HANDS_DELAY:
                # Time has passed, trigger immediately
                print("✅ TRIGGERING GIF DISPLAY!")
                print(f"🎬 Creating {NUM_WINDOWS} windows and starting playback...")

                # Create all windows quickly
                gif_window_names = create_gif_windows(NUM_WINDOWS)

                print(f"✅ Windows created! Playing GIF...")
                # Display GIF in all windows (starts immediately)
                display_gif_in_windows(gif_frames, gif_window_names, start_immediately=True)

                # Close windows after playback
                close_gif_windows(gif_window_names)

                last_trigger_time = current_time

                # Reset two hands timer
                two_hands_detected_time = None

        # Display status
        status = "Show both hands to camera"
        color = (255, 255, 255)
        cv2.putText(frame_bgr, status, (10, CAMERA_HEIGHT - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Show number of hands detected
        num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        cv2.putText(frame_bgr, f'Hands: {num_hands}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display frame
        cv2.imshow('Two Hands Detector', frame_bgr)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
print("👋 Shutting down...")
cap.release()
cv2.destroyAllWindows()
print("✅ Application closed successfully!")
