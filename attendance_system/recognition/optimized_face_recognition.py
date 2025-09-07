"""
Optimized Face Recognition Module
Uses InsightFace with DeepFace-like features for maximum accuracy
"""

import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
import threading
from collections import deque
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("⚠️ InsightFace not available - using OpenCV DNN only")
import onnxruntime

class OptimizedFaceRecognition:
    def __init__(self):
        # Initialize InsightFace with optimized settings
        if INSIGHTFACE_AVAILABLE:
            try:
                self.face_analyzer = FaceAnalysis(providers=['CPUExecutionProvider'])
                self.face_analyzer.prepare(ctx_id=0, det_size=(320, 320))
                self.use_insightface = True
                print("✅ InsightFace initialized successfully")
            except Exception as e:
                print(f"⚠️ InsightFace initialization failed: {e}")
                self.use_insightface = False
        else:
            print("⚠️ InsightFace not available - skipping InsightFace initialization")
            self.use_insightface = False
        
        # Initialize OpenCV DNN face detector
        try:
            self.opencv_dnn_detector = cv2.dnn.readNetFromCaffe(
                "attendance_system/models/opencv_dnn/deploy.prototxt",
                "attendance_system/models/opencv_dnn/res10_300x300_ssd_iter_140000.caffemodel"
            )
            self.use_opencv_dnn = True
            print("✅ OpenCV DNN initialized successfully")
        except Exception as e:
            print(f"⚠️ OpenCV DNN initialization failed: {e}")
            self.use_opencv_dnn = False
        
        # Performance optimizations
        self.frame_buffer = deque(maxlen=1)  # Reduced buffer
        self.processing_lock = threading.Lock()
        self.last_recognition_time = 0
        self.recognition_interval = 0.3  # Faster processing
        
        # Recognition settings
        self.recognition_threshold = 0.7  # Higher threshold for accuracy
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Available models
        self.available_models = {
            "1": {"name": "InsightFace", "type": "insightface", "enabled": self.use_insightface},
            "2": {"name": "OpenCV DNN", "type": "opencv_dnn", "enabled": self.use_opencv_dnn},
            "3": {"name": "OpenCV Cascade", "type": "opencv_cascade", "enabled": True}
        }
        
        # Camera optimization settings
        self.camera_settings = {
            'width': 640,
            'height': 480,
            'fps': 30,
            'buffer_size': 1,
            'fourcc': cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        }
        
        print("🤖 Optimized Face Recognition initialized!")

    def optimize_camera(self, cap):
        """Optimize camera settings for minimal lag"""
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_settings['width'])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_settings['height'])
            cap.set(cv2.CAP_PROP_FPS, self.camera_settings['fps'])
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self.camera_settings['buffer_size'])
            cap.set(cv2.CAP_PROP_FOURCC, self.camera_settings['fourcc'])
            
            # Additional optimizations
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
            cap.set(cv2.CAP_PROP_CONTRAST, 0.5)
        except Exception as e:
            print(f"Camera optimization warning: {e}")

    def get_camera_simple(self):
        """Simple camera initialization"""
        print("🔍 Trying simple camera initialization...")
        
        camera_indices = [0, 1, -1]
        
        for camera_idx in camera_indices:
            try:
                print(f"Trying camera index {camera_idx}...")
                cap = cv2.VideoCapture(camera_idx)
                
                if cap.isOpened():
                    import time
                    time.sleep(0.5)  # Shorter wait
                    
                    success_count = 0
                    for attempt in range(5):  # Fewer tests
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            success_count += 1
                            print(f"Frame {attempt+1}: Success")
                        else:
                            print(f"Frame {attempt+1}: Failed")
                        time.sleep(0.1)  # Shorter delay
                    
                    print(f"Success rate: {success_count}/5 frames")
                    
                    if success_count >= 3:  # Lower threshold
                        print(f"✅ Camera {camera_idx} initialization successful!")
                        return cap
                    else:
                        print(f"❌ Camera {camera_idx} opened but unstable")
                        cap.release()
                else:
                    print(f"❌ Failed to open camera {camera_idx}")
                    cap.release()
                    
            except Exception as e:
                print(f"❌ Error with camera {camera_idx}: {e}")
                try:
                    cap.release()
                except:
                    pass
        
        return None

    def get_camera(self):
        """Get optimized camera"""
        cap = self.get_camera_simple()
        if cap is not None:
            self.optimize_camera(cap)
            return cap
        
        print("❌ No working camera found!")
        return None

    def get_available_models(self):
        """Get list of available models"""
        return self.available_models

    def detect_faces_opencv_dnn(self, frame):
        """Detect faces using OpenCV DNN"""
        faces = []
        try:
            height, width = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.opencv_dnn_detector.setInput(blob)
            detections = self.opencv_dnn_detector.forward()
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    x, y, x2, y2 = box.astype(int)
                    w, h = x2 - x, y2 - y
                    if w > 50 and h > 50:
                        faces.append((x, y, w, h, 'opencv_dnn'))
        except Exception as e:
            pass
        return faces

    def detect_faces_fast(self, frame):
        """Fast face detection with multiple methods"""
        faces = []
        
        # Try InsightFace first (most accurate)
        if self.use_insightface:
            try:
                insight_faces = self.face_analyzer.get(frame)
                for face in insight_faces:
                    if hasattr(face, 'bbox'):
                        bbox = face.bbox.astype(int)
                        x, y, x2, y2 = bbox
                        w, h = x2 - x, y2 - y
                        
                        if x >= 0 and y >= 0 and w > 50 and h > 50:
                            faces.append((x, y, w, h, 'insightface'))
            except Exception as e:
                pass
        
        # Try OpenCV DNN if no faces detected
        if not faces and self.use_opencv_dnn:
            faces = self.detect_faces_opencv_dnn(frame)
        
        # Fallback to OpenCV cascade if no faces detected
        if not faces:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                opencv_faces = self.face_cascade.detectMultiScale(
                    gray, 1.1, 3, minSize=(60, 60), maxSize=(300, 300)
                )
                for (x, y, w, h) in opencv_faces:
                    if w > 50 and h > 50:
                        faces.append((x, y, w, h, 'opencv_cascade'))
            except Exception as e:
                pass
        
        return faces

    def extract_embeddings_optimized(self, face_image):
        """Extract embeddings using InsightFace (optimized)"""
        embeddings = {}
        
        if self.use_insightface:
            try:
                # Resize for better performance
                resized_img = cv2.resize(face_image, (112, 112))
                faces = self.face_analyzer.get(resized_img)
                
                if len(faces) > 0 and hasattr(faces[0], 'embedding'):
                    face_embedding = faces[0].embedding
                    if face_embedding is not None:
                        embeddings["insightface"] = face_embedding
            except Exception as e:
                pass
        
        return embeddings

    def calculate_similarity_fast(self, emb1, emb2):
        """Fast similarity calculation"""
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Use dot product for speed
        dot_product = np.dot(emb1, emb2)
        norm_a = np.linalg.norm(emb1)
        norm_b = np.linalg.norm(emb2)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return max(0.0, dot_product / (norm_a * norm_b))

    def find_face_optimized(self, face_image, stored_embeddings, student_ids, embedding_types):
        """Find matching face using optimized method with model-specific embeddings"""
        if face_image is None:
            return None, 0.0
        
        # Extract embeddings for current face
        query_embeddings = self.extract_embeddings_optimized(face_image)
        if not query_embeddings:
            return None, 0.0
        
        best_match = None
        best_score = -1
        
        # Compare with stored embeddings of the same type
        for i, stored_emb in enumerate(stored_embeddings):
            embedding_type = embedding_types[i] if i < len(embedding_types) else "unknown"
            
            # Only compare embeddings of the same type
            if embedding_type == "insightface" and "insightface" in query_embeddings:
                if stored_emb.shape == query_embeddings["insightface"].shape:
                    similarity = self.calculate_similarity_fast(query_embeddings["insightface"], stored_emb)
                    if similarity > best_score:
                        best_score = similarity
                        best_match = student_ids[i]
            
            elif embedding_type == "opencv_dnn":
                # For OpenCV DNN, use simplified features
                try:
                    face_resized = cv2.resize(face_image, (64, 64))
                    gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                    normalized = gray.flatten().astype(np.float32) / 255.0
                    if stored_emb.shape == normalized.shape:
                        similarity = self.calculate_similarity_fast(normalized, stored_emb)
                        if similarity > best_score:
                            best_score = similarity
                            best_match = student_ids[i]
                except:
                    pass
            
            elif embedding_type == "opencv_cascade":
                # For OpenCV Cascade, use histogram features
                try:
                    face_resized = cv2.resize(face_image, (64, 64))
                    gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
                    hist_normalized = hist.flatten() / np.sum(hist)
                    if stored_emb.shape == hist_normalized.shape:
                        similarity = self.calculate_similarity_fast(hist_normalized, stored_emb)
                        if similarity > best_score:
                            best_score = similarity
                            best_match = student_ids[i]
                except:
                    pass
        
        return best_match, best_score

    def process_frame_optimized(self, frame, stored_embeddings, student_ids, embedding_types):
        """Optimized frame processing"""
        current_time = time.time()
        
        # Skip processing if too frequent
        if current_time - self.last_recognition_time < self.recognition_interval:
            return [], frame
        
        # Detect faces
        faces = self.detect_faces_fast(frame)
        recognition_results = []
        
        # Process each face
        for face_data in faces:
            x, y, w, h, detection_method = face_data
            
            # Skip very small faces
            if w < 60 or h < 60:
                continue
            
            # Extract face region
            face_img = frame[y:y+h, x:x+w]
            
            # Perform recognition
            student_id, confidence = self.find_face_optimized(
                face_img, stored_embeddings, student_ids, embedding_types
            )
            
            recognition_results.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'student_id': student_id,
                'confidence': confidence,
                'detection_method': detection_method
            })
        
        self.last_recognition_time = current_time
        return recognition_results, frame

    def draw_recognition_results(self, frame, results, students_data):
        """Draw recognition results on frame"""
        for result in results:
            x, y, w, h = result['x'], result['y'], result['w'], result['h']
            student_id = result['student_id']
            confidence = result['confidence']
            detection_method = result['detection_method']
            
            if student_id and confidence >= self.recognition_threshold:
                name = students_data.get(student_id, {}).get('name', 'Unknown')
                color = (0, 255, 0)  # Green for recognized
                label = f"{name} ({confidence:.1%})"
            else:
                color = (0, 0, 255)  # Red for unknown
                label = f"Unknown ({confidence:.1%})"
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw detection method indicator
            method_color = (255, 255, 0) if detection_method == 'insightface' else (255, 0, 255)
            cv2.circle(frame, (x+w-10, y+10), 5, method_color, -1)

    def run_real_time_recognition(self, stored_embeddings, student_ids, students_data):
        """Run optimized real-time recognition"""
        cap = self.get_camera()
        if cap is None:
            print("❌ No camera available")
            return
        
        frame_count = 0
        fps_start = time.time()
        recognized_students = set()
        
        print("🎯 Optimized Real-time Recognition Active - Press ESC to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading from camera")
                break
            
            frame = cv2.flip(frame, 1)
            frame_count += 1
            
            # Process frame
            results, processed_frame = self.process_frame_optimized(
                frame, stored_embeddings, student_ids
            )
            
            # Draw results
            self.draw_recognition_results(processed_frame, results, students_data)
            
            # Log recognized students
            for result in results:
                if (result['student_id'] and 
                    result['confidence'] >= self.recognition_threshold):
                    recognized_students.add(result['student_id'])
            
            # Display performance metrics
            current_time = time.time()
            fps = frame_count / (current_time - fps_start)
            
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Faces: {len(results)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(processed_frame, f"Recognized: {len(recognized_students)}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(processed_frame, "Optimized Recognition Active", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(processed_frame, "Press ESC to quit", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Optimized Face Recognition', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return recognized_students 