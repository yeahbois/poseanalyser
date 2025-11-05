import os
import time
import numpy as np
import cv2
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from PIL import Image
from io import BytesIO
import base64
import threading
from queue import Queue
import mediapipe as mp

# --- Flask and SocketIO Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'emoji-reaction-app'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# --- MediaPipe Initialization ---
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# --- State and Configuration ---
client_sessions = {}
frame_queue = Queue(maxsize=5)
result_queue = Queue(maxsize=5)

# --- Load Reaction Images ---
# Store images in memory to avoid reading from disk on every request
reaction_images = {}
def load_reaction_images():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        images_dir = os.path.join(base_dir, 'images')

        # Load default image
        img_path = os.path.join(images_dir, 'monkey_finger_mouth.jpeg')
        reaction_images['default'] = Image.open(img_path)

        # Load raised finger image
        img_path = os.path.join(images_dir, 'monkey_finger_raise.jpg')
        reaction_images['raised_finger'] = Image.open(img_path)

        # Load tongue out GIF frames
        gif_path = os.path.join(images_dir, 'monkey_mouth.gif')
        gif = Image.open(gif_path)
        frames = []
        for i in range(gif.n_frames):
            gif.seek(i)
            frame = gif.convert('RGBA')
            frames.append(frame)
        reaction_images['tongue_out'] = frames
        print("✅ Reaction images and GIFs loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading reaction images: {e}")

# --- Frame Processing Thread ---
def process_frames():
    while True:
        if not frame_queue.empty():
            frame_data = frame_queue.get()
            session_id = frame_data['session_id']

            try:
                # Decode base64 image
                img_data = frame_data['image'].split(',')[1]
                image = Image.open(BytesIO(base64.b64decode(img_data)))
                frame = np.array(image)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # --- Gesture Detection Logic ---
                reaction_key = 'default' # Default reaction

                # Process with MediaPipe
                hands_results = hands.process(frame_rgb)
                face_results = face_mesh.process(frame_rgb)
                pose_results = pose.process(frame_rgb)

                h, w, _ = frame.shape

                # 1. Finger to Mouth Gesture (Highest Priority)
                if hands_results.multi_hand_landmarks and face_results.multi_face_landmarks:
                    for hand_landmarks in hands_results.multi_hand_landmarks:
                        # Index finger tip
                        finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                        for face_landmarks in face_results.multi_face_landmarks:
                            # Mouth landmarks (e.g., upper and lower lip)
                            mouth_top = face_landmarks.landmark[13] # Upper lip
                            mouth_bottom = face_landmarks.landmark[14] # Lower lip

                            # Calculate distance
                            dist_top = np.sqrt((finger_tip.x - mouth_top.x)**2 + (finger_tip.y - mouth_top.y)**2)
                            dist_bottom = np.sqrt((finger_tip.x - mouth_bottom.x)**2 + (finger_tip.y - mouth_bottom.y)**2)

                            if dist_top < 0.1 or dist_bottom < 0.1:
                                reaction_key = 'default' # 'finger_to_mouth' is the default
                                break
                        if reaction_key == 'default':
                            break

                # 2. Raised Finger Gesture
                if reaction_key == 'default' and hands_results.multi_hand_landmarks:
                    for hand_landmarks in hands_results.multi_hand_landmarks:
                        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
                        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

                        # Check if index finger is pointing up
                        if index_tip.y < index_pip.y and index_pip.y < wrist.y:
                            reaction_key = 'raised_finger'
                            break

                # 3. Thumbs Up
                if reaction_key == 'default' and hands_results.multi_hand_landmarks:
                    for hand_landmarks in hands_results.multi_hand_landmarks:
                        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                        thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]

                        if thumb_tip.y < thumb_ip.y:
                            reaction_key = 'thumbs_up'
                            break

                # 4. Crying
                if reaction_key == 'default' and hands_results.multi_hand_landmarks and face_results.multi_face_landmarks:
                    for hand_landmarks in hands_results.multi_hand_landmarks:
                         for face_landmarks in face_results.multi_face_landmarks:
                            # Check if hands are near eyes
                            left_eye = face_landmarks.landmark[33]
                            right_eye = face_landmarks.landmark[263]
                            left_hand = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                            right_hand = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                            if (np.sqrt((left_hand.x - left_eye.x)**2 + (left_hand.y - left_eye.y)**2) < 0.1 or
                                np.sqrt((right_hand.x - right_eye.x)**2 + (right_hand.y - right_eye.y)**2) < 0.1):
                                reaction_key = 'crying'
                                break
                         if reaction_key == 'crying':
                            break

                # 5. Tongue Out / Both Hands Up
                if reaction_key == 'default':
                    # Both hands up check
                    if pose_results.pose_landmarks:
                        left_hand = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                        right_hand = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                        nose = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

                        if left_hand.y < nose.y and right_hand.y < nose.y:
                             reaction_key = 'tongue_out'

                # --- Prepare and Send Result ---
                result = {
                    'reaction': reaction_key,
                    'session_id': session_id,
                    'timestamp': frame_data.get('timestamp', time.time() * 1000)
                }
                result_queue.put(result)

            except Exception as e:
                print(f"❌ Error processing frame: {e}")
                result_queue.put({'error': str(e), 'session_id': session_id})
        else:
            time.sleep(0.001)

# --- Result Sending Thread ---
def send_results():
    gif_frame_index = 0
    while True:
        if not result_queue.empty():
            result = result_queue.get()
            session_id = result.get('session_id')

            if session_id in client_sessions:
                if 'error' in result:
                    socketio.emit('error', {'message': result['error']}, room=session_id)
                else:
                    reaction_key = result['reaction']
                    image_to_send = None

                    if reaction_key == 'tongue_out':
                        # Cycle through GIF frames
                        gif_frames = reaction_images.get('tongue_out', [])
                        if gif_frames:
                           image_to_send = gif_frames[gif_frame_index % len(gif_frames)]
                           gif_frame_index += 1
                    else:
                        image_to_send = reaction_images.get(reaction_key)
                        gif_frame_index = 0 # Reset gif

                    if image_to_send:
                        buffered = BytesIO()
                        image_to_send.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

                        socketio.emit('result', {
                            'image': 'data:image/png;base64,' + img_str,
                            'timestamp': result['timestamp']
                        }, room=session_id)
        else:
            time.sleep(0.01) # Sleep a bit longer here

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

# --- SocketIO Events ---
@socketio.on('connect')
def handle_connect():
    client_sessions[request.sid] = True
    print(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    if request.sid in client_sessions:
        del client_sessions[request.sid]
    print(f"Client disconnected: {request.sid}")

@socketio.on('frame')
def handle_frame(data):
    data['session_id'] = request.sid
    if not frame_queue.full():
        frame_queue.put(data)

# --- Main ---
if __name__ == '__main__':
    load_reaction_images()

    # Start background threads
    processing_thread = threading.Thread(target=process_frames, daemon=True)
    result_thread = threading.Thread(target=send_results, daemon=True)
    processing_thread.start()
    result_thread.start()

    print("🚀 Server running on http://0.0.0.0:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
