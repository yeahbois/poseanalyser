#!/usr/bin/env python3
"""
Monkey Gesture Reactor - A real-time camera-based monkey gesture display application
Uses MediaPipe for hand detection and face mesh detection
Cycles between three monkey images/animations based on your gestures:
- Finger to mouth (shh gesture)
- Raised finger (pointing up)
- Tongue out and moving side to side
"""

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# --- SETUP AND INITIALIZATION ---

# Initialize MediaPipe modules
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- CONFIGURATION CONSTANTS ---
# MacBook Pro screen is typically 1440x900 or 1680x1050, so half would be around 720x450 or 840x525
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 450
EMOJI_WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

# Performance settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
GIF_FPS = 10  # Target FPS for GIF animation (frames per second)

# Helper function to load GIF frames
def load_gif_frames(gif_path):
    gif = Image.open(gif_path)
    frames = []
    try:
        while True:
            frame = gif.convert('RGB')
            frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            frame_resized = cv2.resize(frame_cv, EMOJI_WINDOW_SIZE)
            frames.append(frame_resized)
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    return frames

# Helper function to load static images
def load_static_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"{image_path} could not be loaded")
    img_resized = cv2.resize(img, EMOJI_WINDOW_SIZE)
    return img_resized

# --- LOAD AND PREPARE IMAGES AND ANIMATIONS ---
try:
    # Load monkey images from images folder
    monkey_finger_mouth_image = load_static_image("images/monkey_finger_mouth.jpeg")
    monkey_finger_raise_image = load_static_image("images/monkey_finger_raise.jpg")

    # Load monkey GIF animation
    monkey_mouth_frames = load_gif_frames("images/monkey_mouth.gif")
    if len(monkey_mouth_frames) == 0:
        raise FileNotFoundError("images/monkey_mouth.gif has no frames or could not be loaded")

    print(f"✅ All monkey images loaded successfully!")
    print(f"   - Monkey finger mouth image loaded")
    print(f"   - Monkey finger raise image loaded")
    print(f"   - Monkey mouth GIF: {len(monkey_mouth_frames)} frames")

except Exception as e:
    print("❌ Error loading images! Make sure they are in the correct folder and named properly.")
    print(f"Error details: {e}")
    print("\nExpected files in 'images/' folder:")
    print("- images/monkey_finger_mouth.jpeg (finger to mouth)")
    print("- images/monkey_finger_raise.jpg (finger raised)")
    print("- images/monkey_mouth.gif (tongue out animation)")
    exit()

# Create a blank image for cases where an emoji is missing
blank_emoji = np.zeros((EMOJI_WINDOW_SIZE[0], EMOJI_WINDOW_SIZE[1], 3), dtype=np.uint8)

# --- MAIN LOGIC ---

# Start webcam capture
print("🎥 Starting webcam capture...")
cap = cv2.VideoCapture(0)

# Set camera resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)

# Check if webcam is available
if not cap.isOpened():
    print("❌ Error: Could not open webcam. Make sure your camera is connected and not being used by another application.")
    exit()

# Initialize named windows with specific sizes
cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
cv2.namedWindow('Animation Output', cv2.WINDOW_NORMAL)

# Set window sizes and positions
cv2.resizeWindow('Camera Feed', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.resizeWindow('Animation Output', WINDOW_WIDTH, WINDOW_HEIGHT)

# Position windows side by side
cv2.moveWindow('Camera Feed', 100, 100)
cv2.moveWindow('Animation Output', WINDOW_WIDTH + 150, 100)

print("🚀 Starting monkey gesture detection...")
print("📋 Monkey Gestures:")
print("   - Press 'q' to quit")
print("   🐵 GESTURES:")
print("      • Put finger to mouth = Shh monkey 🤫")
print("      • Raise index finger up = Finger raise monkey ☝️")
print("      • Raise both hands to middle OR stick tongue out = Tongue out monkey 👅")
print("   Default: Finger to mouth monkey")

# Animation tracking variables
import time
current_animation = "MONKEY_FINGER_MOUTH"  # Default state
animation_frame_index = 0
last_gif_update = time.time()
gif_frame_delay = 1.0 / GIF_FPS  # Delay between GIF frames

# Tongue tracking for side-to-side detection
tongue_x_history = []
TONGUE_HISTORY_SIZE = 10  # Track last 10 frames

# Instantiate MediaPipe models
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh, \
     mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("⚠️  Ignoring empty camera frame.")
            continue

        # Flip the frame horizontally for a mirror-like display
        frame = cv2.flip(frame, 1)

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (320, 240))

        # Convert the BGR image to RGB for MediaPipe
        image_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # To improve performance, mark the image as not writeable
        image_rgb.flags.writeable = False

        # --- DETECTION LOGIC ---

        # Process all detections first (on smaller frame for speed)
        results_pose = pose.process(image_rgb)
        results_hands = hands.process(image_rgb)
        results_face = face_mesh.process(image_rgb)

        # Default state - cycle through monkey images
        detected_state = "MONKEY_FINGER_MOUTH"  # Default to first monkey image

        # 1. Check for finger to mouth gesture
        if results_hands.multi_hand_landmarks and results_face.multi_face_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                # Get index finger tip position
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Get mouth center from face landmarks (using lip landmarks)
                face_landmarks = results_face.multi_face_landmarks[0]
                # Mouth center is approximately at landmark 13 (upper lip) and 14 (lower lip)
                mouth_top = face_landmarks.landmark[13]
                mouth_bottom = face_landmarks.landmark[14]
                mouth_center_x = (mouth_top.x + mouth_bottom.x) / 2
                mouth_center_y = (mouth_top.y + mouth_bottom.y) / 2

                # Calculate distance between finger tip and mouth
                distance = ((index_finger_tip.x - mouth_center_x)**2 + (index_finger_tip.y - mouth_center_y)**2)**0.5

                # If finger is close to mouth (threshold of 0.15 in normalized coordinates)
                if distance < 0.15:
                    detected_state = "MONKEY_FINGER_MOUTH"
                    print("✅ MONKEY: Finger to mouth detected!")
                    break

        # 2. Check for raised finger gesture
        if detected_state == "MONKEY_FINGER_MOUTH" and results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                # Get finger landmarks
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

                # Check if index finger is extended and pointing up
                # Index should be higher than wrist and other fingers should be down
                index_extended = index_tip.y < index_mcp.y - 0.1
                index_high = index_tip.y < wrist.y - 0.15
                middle_not_extended = middle_tip.y > index_tip.y + 0.05

                if index_extended and index_high and middle_not_extended:
                    detected_state = "MONKEY_FINGER_RAISE"
                    print("✅ MONKEY: Raised finger detected!")
                    break

        # 3. Check for tongue out gesture (EASY MODE: both hands up OR tongue detected)
        if detected_state == "MONKEY_FINGER_MOUTH":
            # Option 1: Both hands raised to middle of screen or higher
            if results_hands.multi_hand_landmarks:
                num_hands = len(results_hands.multi_hand_landmarks)

                if num_hands >= 2:
                    # Get both hands' positions
                    hand_y_positions = []
                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                        hand_y_positions.append(wrist.y)

                    # Check if both hands are raised (y < 0.65 means upper 65% of screen - very lenient)
                    both_hands_up = all(y < 0.65 for y in hand_y_positions)

                    if both_hands_up:
                        detected_state = "MONKEY_TONGUE_OUT"
                        print("✅ MONKEY: Both hands up - Tongue out activated!")

            # Option 2: Tongue out with side-to-side movement (original detection)
            if detected_state == "MONKEY_FINGER_MOUTH" and results_face.multi_face_landmarks:
                face_landmarks = results_face.multi_face_landmarks[0]

                # Get mouth landmarks - check if mouth is open and detect horizontal movement
                # Using lips landmarks to detect mouth opening
                upper_lip = face_landmarks.landmark[13]
                lower_lip = face_landmarks.landmark[14]
                mouth_left = face_landmarks.landmark[61]
                mouth_right = face_landmarks.landmark[291]

                # Calculate mouth opening (vertical distance)
                mouth_height = abs(upper_lip.y - lower_lip.y)

                # Calculate mouth horizontal center
                mouth_center_x = (mouth_left.x + mouth_right.x) / 2

                # Track tongue position (approximated by mouth opening position)
                if mouth_height > 0.02:  # Mouth is open
                    tongue_x_history.append(mouth_center_x)
                    if len(tongue_x_history) > TONGUE_HISTORY_SIZE:
                        tongue_x_history.pop(0)

                    # Check for side-to-side movement
                    if len(tongue_x_history) >= TONGUE_HISTORY_SIZE:
                        x_min = min(tongue_x_history)
                        x_max = max(tongue_x_history)
                        x_range = x_max - x_min

                        # If there's significant horizontal movement
                        if x_range > 0.015:
                            detected_state = "MONKEY_TONGUE_OUT"
                            print("✅ MONKEY: Tongue out detected!")
                else:
                    # Clear history if mouth is closed
                    tongue_x_history.clear()

        current_animation = detected_state
        
        # --- DISPLAY LOGIC ---

        # Control GIF frame rate
        current_time = time.time()
        if current_time - last_gif_update >= gif_frame_delay:
            animation_frame_index += 1
            last_gif_update = current_time

        # Select the animation to display based on the detected state
        if current_animation == "MONKEY_FINGER_MOUTH":
            # Show monkey finger to mouth image
            display_frame = monkey_finger_mouth_image
            state_name = "🐵 Shh... Finger to Mouth"
        elif current_animation == "MONKEY_FINGER_RAISE":
            # Show monkey raised finger image
            display_frame = monkey_finger_raise_image
            state_name = "🐵 Finger Raised"
        elif current_animation == "MONKEY_TONGUE_OUT":
            # Show monkey tongue out GIF animation
            display_frame = monkey_mouth_frames[animation_frame_index % len(monkey_mouth_frames)]
            state_name = "🐵 Tongue Out!"
        else:
            # Default fallback - show finger to mouth
            display_frame = monkey_finger_mouth_image
            state_name = "🐵 Shh... Finger to Mouth"

        # Resize camera frame to match window size
        camera_frame_resized = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

        # Add the status text to the main camera feed
        cv2.putText(camera_frame_resized, f'STATE: {state_name}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Add instructions text
        cv2.putText(camera_frame_resized, 'Press "q" to quit', (10, WINDOW_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the camera feed and animation
        cv2.imshow('Camera Feed', camera_frame_resized)
        cv2.imshow('Animation Output', display_frame)

        # Exit loop if 'q' is pressed (increased wait time for smoother video)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# --- CLEANUP ---
print("👋 Shutting down...")
cap.release()
cv2.destroyAllWindows()
print("✅ Application closed successfully!")