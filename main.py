#!/usr/bin/env python3
"""
DeepFace + InsightFace Attendance System
Simple, accurate, and working
"""

import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("⚠️ InsightFace not available - using OpenCV DNN only")

class DeepFaceAttendance:
    def __init__(self):
        self.data_file = "database/students_deepface.json"
        self.embeddings_file = "database/embeddings/deepface_embeddings.json"
        self.attendance_dir = "database/attendance"
        
        # Create directories
        os.makedirs("database", exist_ok=True)
        os.makedirs("database/embeddings", exist_ok=True)
        os.makedirs(self.attendance_dir, exist_ok=True)
        
        # Initialize InsightFace (DeepFace-like)
        if INSIGHTFACE_AVAILABLE:
            try:
                self.face_analyzer = FaceAnalysis(providers=['CPUExecutionProvider'])
                self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
                print("✅ InsightFace (DeepFace) loaded successfully")
            except Exception as e:
                print(f"❌ InsightFace failed: {e}")
                self.face_analyzer = None
        else:
            print("⚠️ InsightFace not available - using OpenCV fallback")
            self.face_analyzer = None
        
        # Fallback face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Settings
        self.recognition_threshold = 0.6
        self.max_embeddings = 15   # Maximum number of embeddings per student (changed to 15)
        self.min_embedding_update_interval = 5  # Minimum seconds between embedding updates
        self.students = {}
        self.embeddings = {}
        
        self.load_data()
        print("🤖 DeepFace Attendance System Ready!")

    def load_data(self):
        """Load students and embeddings"""
        # Load students
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    self.students = json.load(f)
            except:
                self.students = {}
        
        # Load embeddings
        if os.path.exists(self.embeddings_file):
            try:
                with open(self.embeddings_file, 'r') as f:
                    self.embeddings = json.load(f)
            except:
                self.embeddings = {}
        
        print(f"✅ Loaded {len(self.students)} students")

    def save_data(self):
        """Save students and embeddings"""
        with open(self.data_file, 'w') as f:
            json.dump(self.students, f, indent=2)
        
        with open(self.embeddings_file, 'w') as f:
            json.dump(self.embeddings, f, indent=2)

    def detect_faces(self, frame):
        """Detect faces using InsightFace or OpenCV"""
        faces = []
        
        # Try InsightFace first
        if self.face_analyzer:
            try:
                insight_faces = self.face_analyzer.get(frame)
                for face in insight_faces:
                    bbox = face.bbox.astype(int)
                    faces.append({
                        'bbox': bbox,
                        'embedding': face.embedding,
                        'confidence': face.det_score
                    })
            except:
                pass
        
        # Fallback to OpenCV
        if not faces:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            opencv_faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
            
            for (x, y, w, h) in opencv_faces:
                faces.append({
                    'bbox': [x, y, x+w, y+h],
                    'embedding': None,
                    'confidence': 0.8
                })
        
        return faces

    def extract_embedding(self, face_img):
        """Extract embedding using InsightFace or OpenCV fallback"""
        # Try InsightFace first if available
        if self.face_analyzer:
            try:
                # Resize for InsightFace
                face_resized = cv2.resize(face_img, (112, 112))
                faces = self.face_analyzer.get(face_resized)
                
                if faces and hasattr(faces[0], 'embedding'):
                    return faces[0].embedding
            except:
                pass
        
        # Fallback to simple OpenCV embedding
        try:
            # Resize to standard size
            face_resized = cv2.resize(face_img, (64, 64))
            
            # Convert to grayscale
            gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            
            # Normalize
            normalized = gray.astype(np.float32) / 255.0
            
            # Flatten to create embedding
            embedding = normalized.flatten()
            
            # Normalize embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None

    def calculate_similarity(self, emb1, emb2):
        """Calculate cosine similarity"""
        if emb1 is None or emb2 is None:
            return 0.0
        
        try:
            emb1_norm = emb1 / np.linalg.norm(emb1)
            emb2_norm = emb2 / np.linalg.norm(emb2)
            return float(np.dot(emb1_norm, emb2_norm))
        except:
            return 0.0

    def recognize_face(self, face_embedding):
        """Recognize face using stored embeddings"""
        if face_embedding is None:
            return None, 0.0
        
        best_match = None
        best_score = 0.0
        
        for student_id, student_data in self.embeddings.items():
            stored_embeddings = student_data.get('embeddings', [])
            
            # If there's no list of embeddings, check for single embedding (backward compatibility)
            if not stored_embeddings and 'embedding' in student_data:
                stored_embeddings = [student_data['embedding']]
            elif stored_embeddings and isinstance(stored_embeddings[0], dict):
                # Handle the new format with metadata
                stored_embedding_vectors = [emb['vector'] for emb in stored_embeddings]
                stored_embeddings = stored_embedding_vectors
            
            # Compare with all stored embeddings for this student
            student_best_score = 0.0
            for stored_embedding in stored_embeddings:
                # Convert to numpy if needed
                if isinstance(stored_embedding, list):
                    stored_embedding = np.array(stored_embedding)
                
                similarity = self.calculate_similarity(face_embedding, stored_embedding)
                student_best_score = max(student_best_score, similarity)
            
            # Keep best match across all students
            if student_best_score > best_score:
                best_score = student_best_score
                best_match = student_id
        
        return best_match, best_score

    def draw_face_box_with_name(self, frame, bbox, name, confidence, is_recognized=True):
        """Draw face box with name only (no confidence shown on overlay)."""
        x1, y1, x2, y2 = bbox
        
        # Choose color based on recognition status
        if is_recognized:
            color = (0, 255, 0)  # Green for recognized
            box_color = (0, 255, 0)
        else:
            color = (0, 0, 255)  # Red for unknown
            box_color = (0, 0, 255)
        
        # Draw rectangle around face
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        
        # Draw filled background for text (name only)
        text = f"{name}"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Background rectangle for text
        cv2.rectangle(frame, (x1, y1-text_height-10), (x1+text_width+10, y1), box_color, -1)
        
        # Draw text
        cv2.putText(frame, text, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw small indicator dot
        cv2.circle(frame, (x2-10, y1+10), 5, color, -1)

    def update_student_embedding(self, student_id, new_embedding, confidence=0.0):
        """Update student embedding by adding new embedding to the list"""
        if student_id not in self.embeddings:
            return False
        
        # Initialize embeddings list if it doesn't exist
        if 'embeddings' not in self.embeddings[student_id]:
            # Migrate from old format if necessary
            if 'embedding' in self.embeddings[student_id]:
                old_embedding = self.embeddings[student_id]['embedding']
                self.embeddings[student_id]['embeddings'] = [old_embedding]
                # Keep the old embedding key for backward compatibility
            else:
                self.embeddings[student_id]['embeddings'] = []
        
        # Convert numpy array to list for JSON serialization
        if isinstance(new_embedding, np.ndarray):
            new_embedding = new_embedding.tolist()
        
        # Add new embedding to the list
        embeddings = self.embeddings[student_id]['embeddings']
        
        # Check if this embedding is significantly different from existing ones
        is_unique = True
        avg_similarity = 0.0
        similarity_scores = []
        
        for existing_emb in embeddings:
            existing_emb_array = np.array(existing_emb)
            new_emb_array = np.array(new_embedding)
            similarity = self.calculate_similarity(existing_emb_array, new_emb_array)
            similarity_scores.append(similarity)
            
            # If very similar to existing embedding (>0.85 similarity), don't add
            if similarity > 0.85:
                is_unique = False
                
        # Calculate average similarity
        if similarity_scores:
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
        
        # Enhanced uniqueness assessment - more unique means lower similarity
        uniqueness_score = 1.0 - avg_similarity if similarity_scores else 1.0
        
        current_time = time.time()
        timestamp = datetime.now().isoformat()
        
        if is_unique:
            # Limit the number of embeddings per student
            if len(embeddings) >= self.max_embeddings:
                # Find the oldest embedding or least useful one
                oldest_index = 0
                # In future versions, could implement more sophisticated pruning
                
                # Remove oldest embedding (first in list)
                embeddings.pop(oldest_index)
            
            # Add new embedding with metadata
            embedding_entry = {
                'vector': new_embedding,
                'timestamp': timestamp,
                'confidence': float(confidence),
                'uniqueness': float(uniqueness_score)
            }
            
            # Add new embedding
            embeddings.append(embedding_entry)
            
            # Update timestamp
            self.embeddings[student_id]['last_updated'] = timestamp
            
            # Update diversity score - how many different appearances we have
            diversity = min(1.0, len(embeddings) / self.max_embeddings)
            self.embeddings[student_id]['diversity_score'] = diversity
            
            # Save to file
            self.save_data()
            print(f"\n✨ NEW APPEARANCE #{len(embeddings)} LEARNED for {self.students[student_id]['name']}")
            print(f"Uniqueness: {uniqueness_score:.2f} | Student Diversity: {diversity*100:.0f}%")
            return True
        else:
            print(f"\nℹ️ Similar appearance already stored for {self.students[student_id]['name']}")
            return False

    def add_student(self):
        """Add new student with automatic face capture"""
        print("\n📝 ADD STUDENT")
        print("-" * 30)
        
        student_id = input("Enter Student ID: ").strip()
        if not student_id:
            print("❌ Student ID required")
            return
        
        name = input("Enter Student Name: ").strip()
        if not name:
            print("❌ Student name required")
            return
        
        if student_id in self.students:
            print("❌ Student already exists")
            return
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera not found")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"📸 Capturing face for {name}")
        print("Look at camera - face will be captured automatically")
        
        captured_embedding = None
        capture_attempts = 0
        max_attempts = 50  # 5 seconds at 10fps
        
        while capture_attempts < max_attempts and captured_embedding is None:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            # Draw instructions
            cv2.putText(frame, f"Looking for face... ({capture_attempts}/{max_attempts})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Position your face in the camera", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw face rectangles and try to capture
            for face in faces:
                bbox = face['bbox']
                x1, y1, x2, y2 = bbox
                
                # Draw temporary box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, "Detecting...", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Try to extract embedding
                if face['embedding'] is not None:
                    captured_embedding = face['embedding']
                    print("✅ Face detected and embedding extracted!")
                    break
                else:
                    # Extract embedding from face region
                    face_img = frame[y1:y2, x1:x2]
                    if face_img.size > 0:
                        embedding = self.extract_embedding(face_img)
                        if embedding is not None:
                            captured_embedding = embedding
                            print("✅ Face detected and embedding extracted!")
                            break
            
            cv2.imshow('Face Capture', frame)
            
            key = cv2.waitKey(100) & 0xFF  # 100ms delay
            if key == 27:  # ESC
                break
            
            capture_attempts += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        if captured_embedding is not None:
            # Save student data
            self.students[student_id] = {
                'name': name,
                'registration_date': datetime.now().isoformat()
            }
            
            # Create metadata for the embedding
            embedding_entry = {
                'vector': captured_embedding.tolist(),
                'timestamp': datetime.now().isoformat(),
                'confidence': 1.0,  # Initial embedding is high confidence
                'uniqueness': 1.0   # First embedding is unique by definition
            }
            
            # Store embedding in new multi-embedding format
            self.embeddings[student_id] = {
                'name': name,
                'embeddings': [embedding_entry],  # List of embeddings with metadata
                'embedding': captured_embedding.tolist(),     # Keep for backward compatibility
                'registration_date': datetime.now().isoformat(),
                'diversity_score': 1/self.max_embeddings  # Starting diversity (1/15)
            }
            
            self.save_data()
            print(f"✅ Student {name} registered successfully!")
            print(f"System can store up to {self.max_embeddings} different appearances per student")
        else:
            print("❌ Failed to capture face - please try again")

    def take_attendance(self):
        """Take attendance with continuous face tracking"""
        print("\n📊 TAKING ATTENDANCE")
        print("-" * 30)
        print("Press ESC to stop")
        
        if not self.students:
            print("❌ No students registered")
            return
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera not found")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        recognized_students = set()
        frame_count = 0
        start_time = time.time()
        camera_start_time = start_time
        last_recognition_time = 0
        recognition_interval = 0.3  # Recognize every 0.3 seconds for better tracking
        
        # For embedding updates tracking
        last_embedding_update = {}
        
        # For displaying temporary notifications
        notification_text = ""
        notification_color = (255, 255, 255)
        notification_end_time = 0
        
        # Terminal header for recognition info
        print("\n" + "=" * 80)
        print(f"{'RECOGNITION RESULTS':<40}{'CONFIDENCE':<20}{'TIME TO RECOGNIZE':<20}")
        print("=" * 80)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame_count += 1
            current_time = time.time()
            
            # Detect faces every frame for continuous tracking
            faces = self.detect_faces(frame)
            
            # Process recognition at intervals
            if current_time - last_recognition_time >= recognition_interval:
                for face in faces:
                    bbox = face['bbox']
                    x1, y1, x2, y2 = bbox
                    
                    # Get embedding for recognition
                    face_embedding = face['embedding']
                    if face_embedding is None:
                        # Extract embedding from face region
                        face_img = frame[y1:y2, x1:x2]
                        if face_img.size > 0:
                            face_embedding = self.extract_embedding(face_img)
                    
                    # Recognize face
                    student_id, confidence = self.recognize_face(face_embedding)
                    
                    if student_id and confidence >= self.recognition_threshold:
                        name = self.students[student_id]['name']
                        
                        # Only log first recognition for this student
                        if student_id not in recognized_students:
                            recognition_time = current_time - camera_start_time
                            # Print recognition result in terminal
                            print(f"{name:<40}{confidence*100:.2f}%{recognition_time:.2f} seconds")
                        
                        recognized_students.add(student_id)
                        
                        # Log attendance
                        self.log_attendance(student_id, name)
                        
                        # Draw recognized face box
                        self.draw_face_box_with_name(frame, bbox, name, confidence, True)
                    elif student_id and 0.5 <= confidence < 0.7:
                        name = self.students[student_id]['name']
                        recognition_time = current_time - camera_start_time
                        
                        # Check if we've updated recently for this student
                        can_update = True
                        if student_id in last_embedding_update:
                            if current_time - last_embedding_update[student_id] < self.min_embedding_update_interval:
                                can_update = False
                        
                        # Automatically learn new appearance without asking
                        if can_update and face_embedding is not None:
                            print(f"\n🔄 Auto-Learning: {name} ({confidence:.1%}) - {recognition_time:.2f} seconds")
                            
                            if self.update_student_embedding(student_id, face_embedding, confidence):
                                # Set notification if a new embedding was added
                                notification_text = f"New appearance learned for {name}!"
                                notification_color = (0, 255, 0)  # Green
                                notification_end_time = current_time + 3  # Show for 3 seconds
                                # Track last update time
                                last_embedding_update[student_id] = current_time
                            
                            # Log attendance even with lower confidence
                            if student_id not in recognized_students:
                                print(f"{'AUTO-CONFIRMED:':<40}{confidence*100:.2f}%{recognition_time:.2f} seconds")
                            
                            recognized_students.add(student_id)
                            self.log_attendance(student_id, name)
                            
                        # Draw recognized face box (even if we're learning)
                        self.draw_face_box_with_name(frame, bbox, name, confidence, True)
                    else:
                        # Draw unknown face box
                        self.draw_face_box_with_name(frame, bbox, "Unknown", confidence, False)
                
                last_recognition_time = current_time
            else:
                # Draw face boxes without recognition (for continuous tracking)
                for face in faces:
                    bbox = face['bbox']
                    x1, y1, x2, y2 = bbox
                    
                    # Draw simple box for tracking
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(frame, "Tracking...", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Draw statistics
            current_time = time.time()
            fps = frame_count / (current_time - start_time)
            elapsed_time = current_time - camera_start_time
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Faces Detected: {len(faces)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Recognized: {len(recognized_students)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.putText(frame, "Auto-Learning Active", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Time: {elapsed_time:.2f}s", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press ESC to stop", (10, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display notification if active
            if current_time < notification_end_time:
                # Draw filled background for notification
                (text_width, text_height), _ = cv2.getTextSize(notification_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, 
                             (frame.shape[1]//2 - text_width//2 - 10, frame.shape[0] - 80),
                             (frame.shape[1]//2 + text_width//2 + 10, frame.shape[0] - 40),
                             (0, 0, 0), -1)
                
                # Draw notification text
                cv2.putText(frame, notification_text,
                           (frame.shape[1]//2 - text_width//2, frame.shape[0] - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, notification_color, 2)
            
            cv2.imshow('DeepFace Attendance', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n" + "=" * 80)
        print(f"✅ Attendance completed! {len(recognized_students)} students recognized")

    def log_attendance(self, student_id, name):
        """Log attendance"""
        today = datetime.now().strftime('%Y-%m-%d')
        attendance_file = os.path.join(self.attendance_dir, f"{today}.json")
        
        attendance_data = []
        if os.path.exists(attendance_file):
            try:
                with open(attendance_file, 'r') as f:
                    attendance_data = json.load(f)
            except:
                pass
        
        # Check if already logged today
        already_logged = any(record['student_id'] == student_id for record in attendance_data)
        
        if not already_logged:
            attendance_record = {
                'student_id': student_id,
                'name': name,
                'timestamp': datetime.now().isoformat()
            }
            attendance_data.append(attendance_record)
            
            with open(attendance_file, 'w') as f:
                json.dump(attendance_data, f, indent=2)

    def delete_student(self):
        """Delete student"""
        print("\n🗑️ DELETE STUDENT")
        print("-" * 30)
        
        if not self.students:
            print("❌ No students registered")
            return
        
        print("Registered students:")
        for i, (student_id, data) in enumerate(self.students.items(), 1):
            print(f"{i}. {data['name']} (ID: {student_id})")
        
        try:
            choice = int(input("\nEnter student number: ")) - 1
            student_list = list(self.students.items())
            
            if 0 <= choice < len(student_list):
                student_id, student_data = student_list[choice]
                confirm = input(f"Delete {student_data['name']}? (y/n): ").lower()
                
                if confirm == 'y':
                    del self.students[student_id]
                    if student_id in self.embeddings:
                        del self.embeddings[student_id]
                    
                    self.save_data()
                    print(f"✅ {student_data['name']} deleted")
                else:
                    print("❌ Deletion cancelled")
            else:
                print("❌ Invalid choice")
        except ValueError:
            print("❌ Please enter a valid number")

    def list_students(self):
        """List students"""
        print("\n📋 REGISTERED STUDENTS")
        print("-" * 30)
        
        if not self.students:
            print("❌ No students registered")
            return
        
        print(f"Total students: {len(self.students)}")
        print()
        
        for i, (student_id, data) in enumerate(self.students.items(), 1):
            print(f"{i}. {data['name']}")
            print(f"   ID: {student_id}")
            print(f"   Registered: {data['registration_date']}")
            print()

    def set_threshold(self):
        """Set recognition threshold"""
        print("\n⚙️ SET THRESHOLD")
        print("-" * 30)
        
        print(f"Current threshold: {self.recognition_threshold:.2f}")
        print("Higher = stricter, Lower = more lenient")
        
        try:
            new_threshold = float(input("Enter new threshold (0.1-0.9): "))
            if 0.1 <= new_threshold <= 0.9:
                self.recognition_threshold = new_threshold
                print(f"✅ Threshold set to {new_threshold:.2f}")
            else:
                print("❌ Must be between 0.1 and 0.9")
        except ValueError:
            print("❌ Please enter a valid number")

    def run(self):
        """Main menu"""
        while True:
            print("\n" + "=" * 40)
            print("🤖 DEEP FACE ATTENDANCE SYSTEM")
            print("=" * 40)
            print("1. Add Student")
            print("2. Take Attendance")
            print("3. Delete Student")
            print("4. List Students")
            print("5. Set Threshold")
            print("6. Exit")
            print("=" * 40)
            
            choice = input("Enter choice (1-6): ").strip()
            
            if choice == '1':
                self.add_student()
            elif choice == '2':
                self.take_attendance()
            elif choice == '3':
                self.delete_student()
            elif choice == '4':
                self.list_students()
            elif choice == '5':
                self.set_threshold()
            elif choice == '6':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice")

def main():
    print("🚀 Starting DeepFace Attendance System...")
    system = DeepFaceAttendance()
    system.run()

if __name__ == "__main__":
    main() 