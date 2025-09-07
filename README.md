# Face Recognition Attendance System

A comprehensive face recognition-based attendance system with multiple deployment options and advanced AI models.

## 🚀 Features

- **Multiple Recognition Engines**: Support for InsightFace, DeepFace, FaceNet, and traditional OpenCV
- **Real-time Recognition**: Live video processing with high accuracy
- **Student Management**: Add, delete, and manage student records
- **Attendance Logging**: Automatic attendance tracking with timestamps
- **Multiple Interfaces**: Command-line, GUI, and Web interfaces
- **Ensemble Recognition**: Combines multiple AI models for enhanced accuracy
- **Adaptive Learning**: Continuously improves recognition accuracy

## 📋 System Requirements

### Basic Requirements
- Python 3.8 or higher
- Webcam or camera device
- 4GB RAM minimum (8GB recommended)
- Windows 10/11, macOS, or Linux

### Optional (for enhanced performance)
- CUDA-compatible GPU for faster AI model inference
- SSD storage for faster database operations

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd attendence
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
Notes:
- If you have issues with `dlib` or GPU packages, install CPU-only first. InsightFace and Torch are optional; the system falls back to OpenCV if not available.
- On Windows, you might need Visual Studio Build Tools for `dlib`. If binaries are available for your Python version: `pip install dlib==19.24.2 --only-binary :all:`

### 3. Download AI Models (Optional but Recommended)
```bash
python -m attendance_system download
```

## 🎯 How to Run

### Method 1: Simple Main System
```bash
python main.py
```
This runs the basic DeepFace + InsightFace attendance system with a simple menu interface.

### Method 2: Advanced Attendance System
```bash
# For student registration
python -m attendance_system register --camera 0

# For attendance recognition
python -m attendance_system recognize --camera 0 --threshold 0.5

# Download required models
python -m attendance_system download
```

### Method 3: GUI Interface
```bash
python attendance_system/gui.py
```
Provides a graphical user interface for easy interaction.

### Method 4: Streamlit Web Interface (Recommended)
```bash
streamlit run web_app.py
```
Opens the Streamlit web app with camera-based Add, Take Attendance, Delete, and List pages.

### Method 5: Legacy Flask App (Optional)
```bash
python attendance_system/app.py
```
Runs a legacy Flask server (limited compared to Streamlit app).

## 📁 Project Structure

```
attendence/
├── main.py                          # Simple main system
├── attendance_system/               # Advanced system
│   ├── __main__.py                 # Command-line interface
│   ├── gui.py                      # GUI interface
│   ├── app.py                      # Web interface
│   ├── recognition/                # Recognition modules
│   ├── registration/               # Registration modules
│   ├── models/                     # AI model files
│   └── config/                     # Configuration files
├── database/                       # Data storage
│   ├── attendance/                 # Attendance logs
│   ├── embeddings/                 # Face embeddings
│   └── images/                     # Student images
└── requirements.txt                # Dependencies
```

## 🔧 Configuration

### Recognition Threshold
- **0.3-0.5**: High sensitivity (may have false positives)
- **0.5-0.7**: Balanced (recommended)
- **0.7-0.9**: High precision (may miss some matches)

### Camera Settings
- Default: Camera index 0
- Resolution: 640x480 (configurable)
- Frame rate: 30 FPS

## 📊 Usage Guide

### 1. Adding Students
- Streamlit: Go to Add Student, enter ID and Name, click Submit to open camera, align your face, then click Capture & Save.
- CLI: Use the menu in `python main.py` and follow prompts.

### 2. Taking Attendance
1. Run the recognition system
2. Students approach the camera
3. System recognizes faces in real-time
4. Attendance is automatically logged
5. Results displayed on screen

### 3. Managing Data
- **List Students**: View all registered students
- **Delete Student**: Remove specific student records
- **Set Threshold**: Adjust recognition sensitivity
- **View Logs**: Check attendance history

## 🔄 Updating Requirements

### Update All Dependencies
```bash
# Update all dependencies
pip install --upgrade -r requirements.txt

# Update requirements.txt with current versions
pip freeze > requirements.txt
```

### Update Specific Components
```bash
# Update AI/ML dependencies
pip install --upgrade torch torchvision onnxruntime insightface face-recognition

# Update web dependencies
pip install --upgrade flask flask-cors flask-socketio

# Update core dependencies
pip install --upgrade opencv-python numpy scipy
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Camera Not Found
```bash
# Check available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"
```

#### 2. AI Models Not Loading
```bash
# Download models manually
python -m attendance_system download --insightface
```

#### 3. Memory Issues
- Reduce frame resolution
- Close other applications
- Use CPU-only mode for AI models

#### 4. Recognition Accuracy Issues
- Adjust recognition threshold
- Add more face samples per student
- Ensure good lighting conditions
- Clean camera lens

### Performance Optimization

#### For Better Speed
```bash
# Use GPU acceleration (if available)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Reduce frame processing
# Set lower resolution in camera settings
```

#### For Better Accuracy
```bash
# Increase embedding samples per student
# Use ensemble recognition mode
# Ensure consistent lighting
```

## 📈 Advanced Features

### Ensemble Recognition
Combines multiple AI models for superior accuracy:
- InsightFace for primary recognition
- FaceNet for verification
- Traditional methods as fallback

### Adaptive Learning
- Continuously updates face embeddings
- Improves recognition over time
- Handles aging and appearance changes

### Anti-Spoofing
- Detects photo/video attacks
- Liveness detection
- Depth-based verification

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing issues
3. Create a new issue with detailed information
4. Include system specifications and error logs

## 🔄 Version History

- **v2.0**: Added ensemble recognition and web interface
- **v1.5**: Enhanced GUI and adaptive learning
- **v1.0**: Basic face recognition attendance system

---

**Note**: This system is designed for educational and research purposes. For production deployment, ensure compliance with privacy laws and data protection regulations. 