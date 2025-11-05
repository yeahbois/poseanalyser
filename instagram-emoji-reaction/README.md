# Instagram Emoji Reaction 🐵

A real-time camera-based monkey gesture detector that uses MediaPipe to detect hand gestures and facial expressions, then displays corresponding animated monkey reactions.

## Features

- **Finger to Mouth Gesture** 🤫: Put finger to mouth → displays monkey shh image
- **Raised Finger Gesture** ☝️: Raise index finger → displays monkey pointing up image
- **Tongue Out Animation** 👅: Raise both hands OR stick tongue out and move side-to-side → displays animated monkey GIF
- **Real-time Processing**: Live camera feed with instant monkey reactions
- **Animated GIF Support**: Plays smooth animations for tongue out gesture

## Requirements

- Python 3.12 (Homebrew: `brew install python@3.12`)
- macOS or Linux with a webcam
- Required Python packages (see `requirements.txt`)

## Installation

1. **Clone or download this project**

2. **Create a virtual environment (Python 3.12) and install dependencies:**
   ```bash
   # macOS: ensure Python 3.12 is installed
   brew install python@3.12

   # Create and activate a virtual environment
   python3.12 -m venv emoji_env
   source emoji_env/bin/activate

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Ensure you have the monkey images in the `images/` directory:**
   - `images/monkey_finger_mouth.jpeg` - Finger to mouth monkey
   - `images/monkey_finger_raise.jpg` - Raised finger monkey
   - `images/monkey_mouth.gif` - Tongue out animated monkey

## Usage

1. **Run the application:**
   ```bash
   # Option A: use helper script (recommended)
   ./run.sh

   # Option B: run manually
   source emoji_env/bin/activate
   python 41.py
   ```

2. **Two windows will open:**
   - **Camera Feed**: Shows your live camera with detection status
   - **Animation Output**: Displays the corresponding monkey reaction based on your gestures

3. **Gestures and Controls:**
   - Press `q` to quit the application
   - **Finger to Mouth** 🤫: Put your index finger to your mouth (detected by distance between finger tip and mouth)
   - **Raised Finger** ☝️: Extend and raise your index finger up above wrist (other fingers curled)
   - **Tongue Out** 👅: Either raise both hands to middle screen OR stick tongue out and move it side-to-side
   - **Default State**: Finger to mouth monkey

## How It Works

The application uses three MediaPipe solutions:

1. **Hand Detection**: Detects finger positions and hand gestures using hand landmarks
2. **Face Mesh Detection**: Analyzes mouth shape and position for tongue detection
3. **Pose Detection** (for two hands detection): Monitors hand positions for both hands up gesture

### Detection Priority
1. **Finger to Mouth** (highest priority) - Detected when finger tip is close to mouth
2. **Raised Finger** - Detected when index finger is extended and raised, other fingers down
3. **Tongue Out** - Can be triggered two ways:
   - **Easy Mode**: Raise both hands to middle screen (more lenient detection)
   - **Original**: Stick tongue out and move it horizontally with mouth open

## Customization

### Adjusting Gesture Sensitivity

Edit the threshold values in `41.py`:
- **Finger to mouth**: Line 185 - `distance < 0.15` (decrease for more sensitive)
- **Raised finger**: Lines 200-202 - Adjust the `y` position thresholds
- **Tongue detection**: Line 260 - `x_range > 0.015` (decrease for more sensitive tongue movement)
- **Both hands up**: Line 224 - `y < 0.65` (increase for more lenient detection)

### Changing Monkey Images

Replace the image files with your own in the `images/` folder:
- `images/monkey_finger_mouth.jpeg` - Your finger to mouth image
- `images/monkey_finger_raise.jpg` - Your raised finger image
- `images/monkey_mouth.gif` - Your tongue out animated GIF

### Other Customization Options

- **Window Size**: Modify `WINDOW_WIDTH` and `WINDOW_HEIGHT` (lines 26-27)
- **GIF Playback Speed**: Adjust `GIF_FPS` (line 33) for animation frame rate
- **Camera Resolution**: Change `CAMERA_WIDTH` and `CAMERA_HEIGHT` (lines 31-32)

## Troubleshooting

### Camera Issues (macOS)
- If you see "not authorized to capture video", grant Camera access for your terminal/editor:
  - System Settings → Privacy & Security → Camera → enable for Terminal/VS Code/iTerm
- Quit and relaunch the terminal/editor after changing permissions
- Ensure no other app is using the camera
- Try different camera indices by changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in line 90

### Images Not Loading
- Verify image files are in the `images/` directory
- Check file names match exactly: `monkey_finger_mouth.jpeg`, `monkey_finger_raise.jpg`, `monkey_mouth.gif`
- Ensure image files are not corrupted
- For GIF files, verify they can be opened with Image.open()

### Detection Issues
- **Finger to mouth not detected**: Ensure your finger is close enough to your mouth (try adjusting threshold on line 185)
- **Raised finger not detected**: Make sure index finger is clearly extended and raised above wrist, other fingers curled down
- **Tongue out not detected**: Try the easier "both hands up" mode instead of tongue movement
- Ensure good lighting on your face and hands
- Keep your face and hands clearly visible in the camera

## Technical Details

- **Framework**: OpenCV for camera capture and display
- **AI Models**: MediaPipe Hand, FaceMesh, and Pose solutions
- **Image Processing**: Real-time RGB conversion and landmark detection with GIF animation support
- **Performance**: Optimized for real-time processing with confidence thresholds and frame rate control
- **Supported Formats**: JPEG, PNG, GIF (animated and static)

## Dependencies

Dependencies are pinned in `requirements-lock.txt` for reproducibility.

Main direct dependencies:
- `opencv-python>=4.8.0` - Computer vision library for camera capture and image processing
- `mediapipe>=0.10.13` - Hand, FaceMesh, and Pose detection
- `numpy>=1.24.0` - Numerical computing
- `Pillow>=10.0.0` - GIF loading and image manipulation

Install with: `pip install -r requirements.txt`

## Project Structure

```
instagram-emoji-reaction/
├── 41.py                  # Main monkey gesture reactor (run this)
├── emoji_reactor.py       # Original emoji reactor (legacy)
├── run.sh                 # Helper script to run the app
├── requirements.txt        # Python dependencies
├── emoji_env/             # Virtual environment (created on setup)
└── images/                # Image assets
    ├── monkey_finger_mouth.jpeg
    ├── monkey_finger_raise.jpg
    └── monkey_mouth.gif
```

## License

This project is for educational and personal use. Please ensure you have appropriate permissions for any images you use.
