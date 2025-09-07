"""
DeepFace-based Face Recognition Module
Uses the best models for maximum accuracy and performance
"""

import cv2
import numpy as np
import json
import os
import time
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("⚠️ InsightFace not available - using OpenCV DNN only")
import onnxruntime as ort

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepFaceRecognition:
    def __init__(self, models_dir: str = "attendance_system/models"):
        """
        Initialize DeepFace recognition system with InsightFace as secondary model
        """
        self.models_dir = models_dir
        self.embeddings_file = "database/embeddings/deepface_embeddings.json"
        self.attendance_dir = "database/attendance"
        
        # Initialize models
        self.insightface_app = None
        self.face_detector = None
        self.recognition_models = {}
        
        # Recognition settings
        self.recognition_threshold = 0.6
        self.ensemble_threshold = 0.5
        self.face_detection_confidence = 0.5
        
        # Performance settings
        self.frame_skip = 2
        self.max_faces = 10
        self.min_face_size = 80
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize DeepFace and InsightFace models"""
        try:
            # Initialize InsightFace
            self.insightface_app = FaceAnalysis(
                name='buffalo_l',
                providers=['CPUExecutionProvider']
            )
            self.insightface_app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("✅ InsightFace model loaded successfully")
            
            # Initialize OpenCV face detector as fallback
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            logger.info("✅ OpenCV face detector loaded")
            
        except Exception as e:
            logger.error(f"❌ Error initializing models: {e}")
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces using multiple methods for better accuracy
        """
        faces = []
        
        try:
            # Method 1: InsightFace detection
            if self.insightface_app:
                insight_faces = self.insightface_app.get(frame)
                for face in insight_faces:
                    bbox = face.bbox.astype(int)
                    faces.append({
                        'bbox': bbox,
                        'confidence': face.det_score,
                        'landmarks': face.kps,
                        'embedding': face.embedding,
                        'method': 'insightface'
                    })
            
            # Method 2: OpenCV cascade if no faces found
            if not faces:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                opencv_faces = self.face_detector.detectMultiScale(
                    gray, 1.1, 4, minSize=(self.min_face_size, self.min_face_size)
                )
                
                for (x, y, w, h) in opencv_faces:
                    faces.append({
                        'bbox': [x, y, x+w, y+h],
                        'confidence': 0.8,
                        'landmarks': None,
                        'embedding': None,
                        'method': 'opencv'
                    })
                    
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            
        return faces
    
    def extract_embeddings(self, frame: np.ndarray, faces: List[Dict]) -> List[Dict]:
        """
        Extract embeddings for detected faces using DeepFace and InsightFace
        """
        embeddings = []
        
        for face in faces:
            try:
                bbox = face['bbox']
                x1, y1, x2, y2 = bbox
                
                # Extract face region
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue
                
                # Resize for consistency
                face_img = cv2.resize(face_img, (160, 160))
                
                # Get InsightFace embedding if available
                if face.get('embedding') is not None:
                    insight_embedding = face['embedding']
                else:
                    # Extract using InsightFace
                    insight_faces = self.insightface_app.get(face_img)
                    if insight_faces:
                        insight_embedding = insight_faces[0].embedding
                    else:
                        insight_embedding = None
                
                # Create embedding record
                embedding_data = {
                    'bbox': bbox,
                    'confidence': face['confidence'],
                    'insightface_embedding': insight_embedding,
                    'timestamp': time.time()
                }
                
                embeddings.append(embedding_data)
                
            except Exception as e:
                logger.error(f"Error extracting embedding: {e}")
                continue
                
        return embeddings
    
    def load_embeddings(self) -> Dict:
        """Load stored embeddings"""
        embeddings = {}
        try:
            if os.path.exists(self.embeddings_file):
                with open(self.embeddings_file, 'r') as f:
                    embeddings = json.load(f)
                logger.info(f"✅ Loaded {len(embeddings)} student embeddings")
            else:
                logger.info("No existing embeddings found")
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
        return embeddings
    
    def save_embeddings(self, embeddings: Dict):
        """Save embeddings to file"""
        try:
            os.makedirs(os.path.dirname(self.embeddings_file), exist_ok=True)
            with open(self.embeddings_file, 'w') as f:
                json.dump(embeddings, f, indent=2)
            logger.info("✅ Embeddings saved successfully")
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            # Normalize embeddings
            embedding1_norm = embedding1 / np.linalg.norm(embedding1)
            embedding2_norm = embedding2 / np.linalg.norm(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1_norm, embedding2_norm)
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def recognize_faces(self, embeddings: List[Dict], stored_embeddings: Dict) -> List[Dict]:
        """
        Recognize faces using ensemble of DeepFace and InsightFace
        """
        recognitions = []
        
        for embedding_data in embeddings:
            best_match = None
            best_score = 0.0
            
            insight_embedding = embedding_data.get('insightface_embedding')
            if insight_embedding is None:
                continue
            
            # Compare with stored embeddings
            for student_id, student_data in stored_embeddings.items():
                student_insight_embedding = student_data.get('insightface_embedding')
                
                if student_insight_embedding is not None:
                    # Convert to numpy array if needed
                    if isinstance(student_insight_embedding, list):
                        student_insight_embedding = np.array(student_insight_embedding)
                    
                    # Calculate similarity
                    similarity = self.calculate_similarity(
                        insight_embedding, student_insight_embedding
                    )
                    
                    # Update best match
                    if similarity > best_score:
                        best_score = similarity
                        best_match = {
                            'student_id': student_id,
                            'name': student_data.get('name', 'Unknown'),
                            'similarity': similarity,
                            'bbox': embedding_data['bbox']
                        }
            
            # Apply threshold
            if best_match and best_match['similarity'] >= self.recognition_threshold:
                recognitions.append(best_match)
            else:
                # Add unknown face
                recognitions.append({
                    'student_id': 'unknown',
                    'name': 'Unknown',
                    'similarity': best_score if best_match else 0.0,
                    'bbox': embedding_data['bbox']
                })
        
        return recognitions
    
    def register_student(self, student_id: str, name: str, images: List[np.ndarray]) -> bool:
        """
        Register a new student with multiple face images
        """
        try:
            # Load existing embeddings
            embeddings = self.load_embeddings()
            
            # Extract embeddings from all images
            all_embeddings = []
            for img in images:
                faces = self.detect_faces(img)
                if faces:
                    img_embeddings = self.extract_embeddings(img, faces)
                    all_embeddings.extend(img_embeddings)
            
            if not all_embeddings:
                logger.error("No faces detected in registration images")
                return False
            
            # Use the best embedding (highest confidence)
            best_embedding = max(all_embeddings, key=lambda x: x['confidence'])
            
            # Store student data
            embeddings[student_id] = {
                'name': name,
                'insightface_embedding': best_embedding['insightface_embedding'].tolist(),
                'registration_date': datetime.now().isoformat(),
                'confidence': best_embedding['confidence']
            }
            
            # Save embeddings
            self.save_embeddings(embeddings)
            
            logger.info(f"✅ Student {name} (ID: {student_id}) registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error registering student: {e}")
            return False
    
    def take_attendance(self, frame: np.ndarray) -> List[Dict]:
        """
        Take attendance from a frame
        """
        try:
            # Detect faces
            faces = self.detect_faces(frame)
            if not faces:
                return []
            
            # Extract embeddings
            embeddings = self.extract_embeddings(frame, faces)
            if not embeddings:
                return []
            
            # Load stored embeddings
            stored_embeddings = self.load_embeddings()
            
            # Recognize faces
            recognitions = self.recognize_faces(embeddings, stored_embeddings)
            
            return recognitions
            
        except Exception as e:
            logger.error(f"Error in attendance taking: {e}")
            return []
    
    def log_attendance(self, recognitions: List[Dict]):
        """
        Log attendance to file
        """
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            attendance_file = os.path.join(self.attendance_dir, f"{today}.json")
            
            # Load existing attendance
            attendance_data = []
            if os.path.exists(attendance_file):
                with open(attendance_file, 'r') as f:
                    attendance_data = json.load(f)
            
            # Add new attendance records
            current_time = datetime.now().isoformat()
            for recognition in recognitions:
                if recognition['student_id'] != 'unknown':
                    attendance_record = {
                        'student_id': recognition['student_id'],
                        'name': recognition['name'],
                        'timestamp': current_time,
                        'similarity': recognition['similarity']
                    }
                    attendance_data.append(attendance_record)
            
            # Save attendance
            os.makedirs(self.attendance_dir, exist_ok=True)
            with open(attendance_file, 'w') as f:
                json.dump(attendance_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error logging attendance: {e}")
    
    def get_attendance_stats(self, date: str = None) -> Dict:
        """
        Get attendance statistics for a date
        """
        try:
            if date is None:
                date = datetime.now().strftime('%Y-%m-%d')
            
            attendance_file = os.path.join(self.attendance_dir, f"{date}.json")
            
            if not os.path.exists(attendance_file):
                return {'total': 0, 'students': [], 'date': date}
            
            with open(attendance_file, 'r') as f:
                attendance_data = json.load(f)
            
            # Count unique students
            unique_students = {}
            for record in attendance_data:
                student_id = record['student_id']
                if student_id not in unique_students:
                    unique_students[student_id] = {
                        'name': record['name'],
                        'first_seen': record['timestamp'],
                        'last_seen': record['timestamp'],
                        'count': 1
                    }
                else:
                    unique_students[student_id]['last_seen'] = record['timestamp']
                    unique_students[student_id]['count'] += 1
            
            return {
                'total': len(unique_students),
                'students': list(unique_students.values()),
                'date': date
            }
            
        except Exception as e:
            logger.error(f"Error getting attendance stats: {e}")
            return {'total': 0, 'students': [], 'date': date}
    
    def delete_student(self, student_id: str) -> bool:
        """
        Delete a student from the system
        """
        try:
            embeddings = self.load_embeddings()
            
            if student_id in embeddings:
                del embeddings[student_id]
                self.save_embeddings(embeddings)
                logger.info(f"✅ Student {student_id} deleted successfully")
                return True
            else:
                logger.warning(f"Student {student_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting student: {e}")
            return False
    
    def list_students(self) -> List[Dict]:
        """
        List all registered students
        """
        try:
            embeddings = self.load_embeddings()
            students = []
            
            for student_id, data in embeddings.items():
                students.append({
                    'id': student_id,
                    'name': data.get('name', 'Unknown'),
                    'registration_date': data.get('registration_date', 'Unknown')
                })
            
            return students
            
        except Exception as e:
            logger.error(f"Error listing students: {e}")
            return []
    
    def set_threshold(self, threshold: float):
        """Set recognition threshold"""
        self.recognition_threshold = max(0.1, min(0.9, threshold))
        logger.info(f"Recognition threshold set to {self.recognition_threshold}")
    
    def get_system_status(self) -> Dict:
        """Get system status and model information"""
        return {
            'models_loaded': {
                'insightface': self.insightface_app is not None,
                'opencv_detector': self.face_detector is not None
            },
            'settings': {
                'recognition_threshold': self.recognition_threshold,
                'face_detection_confidence': self.face_detection_confidence,
                'frame_skip': self.frame_skip
            },
            'storage': {
                'embeddings_file': self.embeddings_file,
                'attendance_dir': self.attendance_dir
            }
        } 