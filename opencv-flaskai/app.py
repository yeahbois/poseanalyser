import os
import time
import numpy as np
import cv2
from flask import Flask, render_template_string, request
from flask_socketio import SocketIO, emit, disconnect
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import base64
import threading
from queue import Queue
import uuid
import torch

os.environ["OMP_NUM_THREADS"] = "1"

app = Flask(__name__)
app.config['SECRET_KEY'] = 'yolov8-termux-fps'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

print("Loading YOLOv8n (detection only)...")
model = YOLO("yolov8n.pt")

model.fuse()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"Using device: {device}")

dummy = np.zeros((320, 320, 3), dtype=np.uint8)
_ = model(dummy, verbose=False)
print("✅ Ready!")

client_sessions = {}

frame_queue = Queue(maxsize=5)  
result_queue = Queue(maxsize=5)  

processing = False
last_frame_time = 0
detection_fps = 0
frame_count = 0
last_fps_update = time.time()
processing_times = []

def process_frames():
    global processing, detection_fps, frame_count, last_fps_update, processing_times
    
    while True:
        if not frame_queue.empty():
            frame_data = frame_queue.get()
            processing = True
            
            try:
                start_time = time.time()
                
                img_data = frame_data['image'].split(',')[1]
                image = Image.open(BytesIO(base64.b64decode(img_data)))
                frame = np.array(image)
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

                results = model(frame, verbose=False, imgsz=320, conf=0.4)
                res = results[0]

                annotated = frame.copy()
                
                for box, cls, conf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{model.names[int(cls)]}: {conf:.2f}"
                    
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
                    
                    cv2.putText(annotated, label, (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                _, buffer = cv2.imencode('.webp', annotated, [int(cv2.IMWRITE_WEBP_QUALITY), 70])
                annotated_b64 = base64.b64encode(buffer).decode('utf-8')
                
                frame_count += 1
                current_time = time.time()
                if current_time - last_fps_update >= 1.0:
                    detection_fps = frame_count
                    frame_count = 0
                    last_fps_update = current_time
              
                processing_time = (time.time() - start_time) * 1000
                processing_times.append(processing_time)
                if len(processing_times) > 10:
                    processing_times.pop(0)
                
                avg_processing_time = sum(processing_times) / len(processing_times)
                
                result_queue.put({
                    'annotated': annotated_b64,
                    'timestamp': frame_data.get('timestamp', time.time() * 1000),
                    'detection_fps': detection_fps,
                    'session_id': frame_data['session_id'],
                    'processing_time': avg_processing_time
                })
                
                print(f"Processing time: {processing_time:.2f}ms, Detection FPS: {detection_fps}")
            except Exception as e:
                print("❌ Error in process_frames:", e)
                result_queue.put({
                    'error': str(e),
                    'session_id': frame_data['session_id']
                })
            finally:
                processing = False
        else:
            time.sleep(0.001)

processing_thread = threading.Thread(target=process_frames, daemon=True)
processing_thread.start()

def send_results():
    while True:
        if not result_queue.empty():
            result = result_queue.get()
            session_id = result.get('session_id')
            
            if session_id in client_sessions:
                if 'error' in result:
                    socketio.emit('error', {'message': result['error']}, room=session_id)
                else:
                    socketio.emit('result', result, room=session_id)
        else:
            time.sleep(0.001)

result_thread = threading.Thread(target=send_results, daemon=True)
result_thread.start()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
  <title>YOLOv8 Real-Time (FPS)</title>
  <style>
    body { font-family: Arial, text-align: center; background: #f0f0f0; padding: 10px; margin: 0; }
    h2 { color: #2c3e50; margin: 10px 0; }
    .container { display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; }
    video, img { border: 2px solid #3498db; margin: 5px; width: 426px; height: 320px; object-fit: cover; }
    button { padding: 10px 20px; font-size: 16px; background: #27ae60; color: white; border: none; border-radius: 4px; margin: 10px; }
    .status { font-weight: bold; margin: 5px; }
    .fps { background: #2c3e50; color: white; padding: 4px 8px; border-radius: 4px; display: inline-block; margin-top: 5px; }
    .stats { display: flex; justify-content: center; gap: 20px; margin-top: 10px; }
    .error { color: red; margin: 10px; }
    .controls { display: flex; justify-content: center; gap: 10px; margin: 10px; }
    .slider-container { display: flex; align-items: center; gap: 10px; margin: 10px; }
    input[type="range"] { width: 150px; }
  </style>
</head>
<body>
  <h2>YOLOv8 Real-Time Detection</h2>
  <div class="container">
    <div>
      <div>📹 Camera</div>
      <video id="video" autoplay playsinline muted></video>
    </div>
    <div>
      <div>🤖 Detection</div>
      <img id="result" />
    </div>
  </div>
  <br>
  <div class="controls">
    <button id="start">Start Detection</button>
    <button id="stop" disabled>Stop Detection</button>
  </div>
  <div class="slider-container">
    <label for="quality">Quality:</label>
    <input type="range" id="quality" min="0.3" max="0.9" step="0.1" value="0.6">
    <span id="qualityValue">0.6</span>
  </div>
  <div class="slider-container">
    <label for="fps">Target FPS:</label>
    <input type="range" id="fps" min="10" max="30" step="5" value="20">
    <span id="fpsValue">20</span>
  </div>
  <div class="status" id="status">Click to start</div>
  <div class="error" id="error"></div>
  <div class="stats">
    <div class="fps" id="fps">Camera FPS: --</div>
    <div class="fps" id="detection_fps">Detection FPS: --</div>
    <div class="fps" id="latency">Latency: --ms</div>
    <div class="fps" id="processing">Processing: --ms</div>
  </div>

  <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
  <script>
    const video = document.getElementById('video');
    const resultImg = document.getElementById('result');
    const statusEl = document.getElementById('status');
    const errorEl = document.getElementById('error');
    const fpsEl = document.getElementById('fps');
    const detectionFpsEl = document.getElementById('detection_fps');
    const latencyEl = document.getElementById('latency');
    const processingEl = document.getElementById('processing');
    const startBtn = document.getElementById('start');
    const stopBtn = document.getElementById('stop');
    const qualitySlider = document.getElementById('quality');
    const qualityValue = document.getElementById('qualityValue');
    const fpsSlider = document.getElementById('fps');
    const fpsValue = document.getElementById('fpsValue');

    let stream = null;
    let socket = null;
    let isActive = false;
    let frameCount = 0;
    let lastTime = Date.now();
    let fps = 0;
    let skipFrames = 0;
    let processingTime = 0;
    let frameInterval = 50; // Default to 20 FPS
    let sessionId = null;
    let imageQuality = 0.6;

    // Update quality value display
    qualitySlider.addEventListener('input', () => {
      imageQuality = parseFloat(qualitySlider.value);
      qualityValue.textContent = imageQuality.toFixed(1);
    });

    // Update FPS value display and interval
    fpsSlider.addEventListener('input', () => {
      const targetFps = parseInt(fpsSlider.value);
      fpsValue.textContent = targetFps;
      frameInterval = 1000 / targetFps;
    });

    async function startCamera() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ 
          video: { 
            width: { ideal: 640 }, 
            height: { ideal: 480 },
            facingMode: 'environment' // Use rear camera on mobile if available
          } 
        });
        video.srcObject = stream;
        video.width = 426;
        video.height = 320;
        statusEl.textContent = "✅ Camera ready";
        return true;
      } catch (err) {
        statusEl.textContent = "❌ Camera error: " + (err.message || 'denied');
        errorEl.textContent = "Camera access denied or not available";
        console.error(err);
        return false;
      }
    }

    function stopCamera() {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
        video.srcObject = null;
      }
    }

    function captureAndSend() {
      if (!isActive || !stream) return;

      // Skip frames if processing is taking too long
      if (skipFrames > 0) {
        skipFrames--;
        setTimeout(captureAndSend, frameInterval);
        return;
      }

      const startTime = performance.now();
      
      const canvas = document.createElement('canvas');
      canvas.width = 426;
      canvas.height = 320;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, 426, 320);
      
      // Use dynamic quality based on slider
      const dataUrl = canvas.toDataURL('image/webp', imageQuality);
      
      socket?.emit('frame', { 
        image: dataUrl,
        timestamp: startTime,
        session_id: sessionId
      });

      // Update FPS
      frameCount++;
      const now = Date.now();
      if (now - lastTime >= 1000) {
        fps = frameCount;
        fpsEl.textContent = `Camera FPS: ${fps}`;
        frameCount = 0;
        lastTime = now;
      }

      // Use setTimeout with target frame interval for consistent FPS
      setTimeout(captureAndSend, frameInterval);
    }

    startBtn.onclick = async () => {
      if (!await startCamera()) return;

      socket = io();
      
      socket.on('connect', () => {
        sessionId = socket.id;
        console.log('Connected with session ID:', sessionId);
      });
      
      socket.on('connect_error', (err) => {
        statusEl.textContent = "🔌 Connection failed";
        errorEl.textContent = "Failed to connect to server";
        console.error("Socket.IO error:", err);
      });
      
      socket.on('error', (data) => {
        errorEl.textContent = "Server error: " + data.message;
        console.error("Server error:", data.message);
      });
      
      socket.on('result', (data) => {
        // Display WebP image
        resultImg.src = 'data:image/webp;base64,' + data.annotated;
        resultImg.width = 426;
        resultImg.height = 320;
        
        // Update detection FPS from server
        if (data.detection_fps !== undefined) {
          detectionFpsEl.textContent = `Detection FPS: ${data.detection_fps}`;
        }
        
        // Update processing time
        if (data.processing_time !== undefined) {
          processingEl.textContent = `Processing: ${data.processing_time.toFixed(0)}ms`;
        }
        
        // Calculate latency
        const latency = performance.now() - data.timestamp;
        latencyEl.textContent = `Latency: ${latency.toFixed(0)}ms`;
        
        // If latency is too high, skip frames
        if (latency > 100) {
          skipFrames = Math.floor(latency / 100);
          // Adjust frame interval based on latency
          frameInterval = Math.max(50, Math.min(100, latency));
        } else {
          // Try to reach target FPS
          frameInterval = 1000 / parseInt(fpsSlider.value);
        }
      });

      isActive = true;
      startBtn.disabled = true;
      stopBtn.disabled = false;
      statusEl.textContent = "📡 Processing...";
      errorEl.textContent = "";
      captureAndSend();
    };

    stopBtn.onclick = () => {
      isActive = false;
      stopCamera();
      if (socket) {
        socket.disconnect();
        socket = null;
      }
      startBtn.disabled = false;
      stopBtn.disabled = true;
      statusEl.textContent = "Stopped";
      resultImg.src = "";
    };
  </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

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
    global last_frame_time
    
    data['session_id'] = request.sid
    
    if not frame_queue.full():
        frame_queue.put(data)
        last_frame_time = time.time()
    else:
        print("Frame queue is full, dropping frame")

if __name__ == '__main__':
    print("Server running on http://0.0.0.0:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)