"""
Face embedding extraction module for the registration system.
"""
import cv2
import numpy as np
import os
import time
from typing import List, Tuple, Dict, Optional, Union
import onnxruntime as ort
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("⚠️ InsightFace not available - falling back to OpenCV DNN")
from ..config.settings import (
    FACE_RECOGNITION_SIZE, FACE_CONFIDENCE_THRESHOLD,
    MODEL_DIR, USE_GPU
)
from ..config.model_config import MODELS, FACE_RECOGNITION_PARAMS
from ..utils.image_processing import (
    preprocess_face, detect_blur, align_face, crop_face
)

class FaceEmbeddingExtractor:
    """
    Extracts face embeddings from images using different face recognition models.
    Supports InsightFace ArcFace, MobileFaceNet, and other models.
    """
    
    def __init__(self, model_name="arcface"):
        """
        Initialize the face embedding extractor.
        
        Args:
            model_name (str): Name of the face recognition model to use
                              Options: "arcface", "facenet", "mobilefacenet", "opencv_simple"
        """
        # Switch to simple OpenCV-based embeddings if InsightFace not available and trying to use arcface
        if model_name == "arcface" and not INSIGHTFACE_AVAILABLE:
            print("⚠️ Switching to OpenCV simple embeddings (InsightFace not available)")
            model_name = "opencv_simple"
        
        self.model_name = model_name
        
        # Handle opencv_simple model (doesn't need external files)
        if model_name == "opencv_simple":
            self.model_config = {
                "size": (64, 64),
                "embedding_size": 4096  # 64*64 flattened
            }
        else:
            self.model_config = MODELS.get(model_name)
            if not self.model_config:
                raise ValueError(f"Model {model_name} not supported")
        
        # Get model parameters
        self.model_file = self.model_config.get("model_file", None)
        self.input_size = self.model_config.get("size", FACE_RECOGNITION_SIZE)
        self.embedding_size = self.model_config.get("embedding_size", 512)
        
        # Load recognition parameters
        self.recognition_params = FACE_RECOGNITION_PARAMS.get(model_name, {})
        self.normalize = self.recognition_params.get("normalization", True)
        self.norm_mean = self.recognition_params.get("normalization_mean", 127.5)
        self.norm_std = self.recognition_params.get("normalization_std", 128.0)
        
        # Initialize model
        self._load_model()
    
    def _load_model(self):
        """Load the face recognition model."""
        if self.model_name == "opencv_simple":
            # No external model file needed for simple OpenCV embeddings
            print("Using simple OpenCV-based face embeddings")
            return
        
        # Check if model exists
        if not os.path.exists(self.model_file):
            os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
            raise FileNotFoundError(f"Model file not found: {self.model_file}")
        
        if self.model_name == "arcface":
            # Use InsightFace for ArcFace
            self._load_insightface_model()
        else:
            # Use ONNX Runtime for other models
            self._load_onnx_model()
    
    def _load_insightface_model(self):
        """Load the InsightFace model."""
        if not INSIGHTFACE_AVAILABLE:
            print("⚠️ InsightFace not available - cannot load InsightFace model")
            raise ImportError("InsightFace not available")
        
        try:
            # Initialize FaceAnalysis app
            self.app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            print(f"Loaded InsightFace model: {self.model_name}")
        except Exception as e:
            print(f"Error loading InsightFace model: {e}")
            raise
    
    def _load_onnx_model(self):
        """Load an ONNX model using ONNX Runtime."""
        try:
            # Set execution provider based on configuration
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if USE_GPU else ['CPUExecutionProvider']
            
            # Create ONNX Runtime session
            self.session = ort.InferenceSession(self.model_file, providers=providers)
            
            # Get input and output details
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            print(f"Loaded ONNX model: {self.model_name}")
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            raise
    
    def preprocess_face_image(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess a face image for embedding extraction.
        
        Args:
            face_image (numpy.ndarray): Input face image
        
        Returns:
            numpy.ndarray: Preprocessed face image
        """
        # Resize to model input size
        resized_face = cv2.resize(face_image, self.input_size)
        
        # Convert to RGB if needed (InsightFace expects RGB)
        if self.model_name == "arcface":
            if len(resized_face.shape) == 3 and resized_face.shape[2] == 3:
                resized_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values if needed
        if self.normalize:
            resized_face = (resized_face - self.norm_mean) / self.norm_std
        
        # Add batch dimension if using ONNX
        if self.model_name != "arcface":
            resized_face = np.expand_dims(resized_face, axis=0)
            # ONNX models might expect NCHW format (batch, channels, height, width)
            if len(resized_face.shape) == 4:  # If it's a color image
                resized_face = np.transpose(resized_face, (0, 3, 1, 2))
        
        return resized_face
    
    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract face embedding from a face image.
        
        Args:
            face_image (numpy.ndarray): Input face image
        
        Returns:
            numpy.ndarray: Face embedding vector
        """
        if face_image is None or face_image.size == 0:
            raise ValueError("Invalid face image")
        
        # Check for blur
        is_blurry, blur_value = detect_blur(face_image)
        if is_blurry:
            print(f"Warning: Blurry image detected (blur value: {blur_value})")
        
        # Preprocess face
        preprocessed_face = self.preprocess_face_image(face_image)
        
        # Extract embedding based on model type
        if self.model_name == "arcface":
            return self._extract_insightface_embedding(face_image)
        elif self.model_name == "opencv_simple":
            return self._extract_opencv_simple_embedding(face_image)
        else:
            return self._extract_onnx_embedding(preprocessed_face)
    
    def _extract_insightface_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract embedding using InsightFace.
        
        Args:
            face_image (numpy.ndarray): Input face image
        
        Returns:
            numpy.ndarray: Face embedding vector
        """
        # InsightFace expects BGR images in its detect method
        # But we already handle this in preprocess_face_image
        # So we keep the face_image as is
        try:
            # Detect faces with InsightFace
            faces = self.app.get(face_image)
            
            # If no face is detected, return empty embedding
            if not faces:
                print("No face detected by InsightFace")
                return np.zeros(self.embedding_size)
            
            # Get embedding from the first face
            embedding = faces[0].embedding
            
            # Normalize embedding
            if embedding is not None:
                embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
        except Exception as e:
            print(f"Error extracting InsightFace embedding: {e}")
            return np.zeros(self.embedding_size)
    
    def _extract_onnx_embedding(self, preprocessed_face: np.ndarray) -> np.ndarray:
        """
        Extract embedding using ONNX model.
        
        Args:
            preprocessed_face (numpy.ndarray): Preprocessed face image
        
        Returns:
            numpy.ndarray: Face embedding vector
        """
        try:
            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: preprocessed_face})
            embedding = outputs[0][0]
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
        except Exception as e:
            print(f"Error extracting ONNX embedding: {e}")
            return np.zeros(self.embedding_size)
    
    def _extract_opencv_simple_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract embedding using simple OpenCV operations (fallback method).
        
        Args:
            face_image (numpy.ndarray): Input face image
        
        Returns:
            numpy.ndarray: Face embedding vector
        """
        try:
            # Resize to standard size
            resized_face = cv2.resize(face_image, self.input_size)
            
            # Convert to grayscale
            gray = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
            
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
            print(f"Error extracting OpenCV simple embedding: {e}")
            return np.zeros(self.embedding_size)
    
    def extract_embeddings_batch(self, face_images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract embeddings from multiple face images.
        
        Args:
            face_images (List[numpy.ndarray]): List of face images
        
        Returns:
            List[numpy.ndarray]: List of face embeddings
        """
        embeddings = []
        
        for face_image in face_images:
            try:
                embedding = self.extract_embedding(face_image)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error extracting embedding: {e}")
                # Append zero embedding for failed faces
                embeddings.append(np.zeros(self.embedding_size))
        
        return embeddings
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate similarity score between two face embeddings.
        
        Args:
            embedding1 (numpy.ndarray): First face embedding
            embedding2 (numpy.ndarray): Second face embedding
        
        Returns:
            float: Similarity score (cosine similarity)
        """
        # Normalize embeddings
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        embedding1 = embedding1 / norm1
        embedding2 = embedding2 / norm2
        
        # Calculate cosine similarity
        return np.dot(embedding1, embedding2)


class FaceDetector:
    """
    Face detector class that uses different face detection models.
    Supports RetinaFace, OpenCV DNN, and other models.
    """
    
    def __init__(self, model_name="retinaface"):
        """
        Initialize the face detector.
        
        Args:
            model_name (str): Name of the face detection model to use
                             Options: "retinaface", "opencv_dnn"
        """
        # Automatically switch to OpenCV DNN if InsightFace is not available
        if model_name == "retinaface" and not INSIGHTFACE_AVAILABLE:
            print("⚠️ Switching to OpenCV DNN face detector (InsightFace not available)")
            model_name = "opencv_dnn"
        
        self.model_name = model_name
        self.model_config = MODELS.get(model_name)
        self.app = None
        
        if not self.model_config:
            raise ValueError(f"Model {model_name} not supported")
        
        # Load face detector
        self._load_detector()
    
    def _load_detector(self):
        """Load the face detection model."""
        if self.model_name == "retinaface":
            if not INSIGHTFACE_AVAILABLE:
                print("⚠️ InsightFace not available - cannot load RetinaFace detector")
                raise ImportError("InsightFace not available for RetinaFace")
            
            try:
                # Initialize FaceAnalysis app for detection only
                self.app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
                self.app.prepare(ctx_id=0, det_size=(640, 640))
                print(f"Loaded {self.model_name} detector")
            except Exception as e:
                print(f"Error loading {self.model_name} detector: {e}")
                raise
        elif self.model_name == "opencv_dnn":
            try:
                # Load OpenCV DNN model
                model_file = self.model_config["model_file"]
                config_file = self.model_config["config_file"]
                
                if not os.path.exists(model_file) or not os.path.exists(config_file):
                    raise FileNotFoundError(f"Model or config file not found")
                
                self.net = cv2.dnn.readNetFromCaffe(config_file, model_file)
                
                # Use GPU if available and enabled
                if USE_GPU:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                
                print(f"Loaded {self.model_name} detector")
            except Exception as e:
                print(f"Error loading {self.model_name} detector: {e}")
                raise
    
    def detect_faces(self, image: np.ndarray, min_confidence: float = FACE_CONFIDENCE_THRESHOLD) -> List[Dict]:
        """
        Detect faces in an image.
        
        Args:
            image (numpy.ndarray): Input image
            min_confidence (float): Minimum confidence threshold
        
        Returns:
            List[Dict]: List of detected faces with bounding boxes and landmarks
        """
        if image is None:
            return []
        
        if self.model_name == "retinaface":
            return self._detect_retinaface(image, min_confidence)
        elif self.model_name == "opencv_dnn":
            return self._detect_opencv_dnn(image, min_confidence)
        else:
            raise ValueError(f"Unsupported detection model: {self.model_name}")
    
    def _detect_retinaface(self, image: np.ndarray, min_confidence: float) -> List[Dict]:
        """
        Detect faces using RetinaFace.
        
        Args:
            image (numpy.ndarray): Input image
            min_confidence (float): Minimum confidence threshold
        
        Returns:
            List[Dict]: List of detected faces
        """
        # Convert to RGB for InsightFace
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.app.get(rgb_image)
        
        # Filter and format results
        results = []
        for face in faces:
            if face.det_score < min_confidence:
                continue
            
            # Get bounding box
            x1, y1, x2, y2 = face.bbox.astype(int)
            w, h = x2 - x1, y2 - y1
            
            # Get landmarks if available
            landmarks = face.landmark if hasattr(face, 'landmark') else None
            
            # Convert landmarks to list if available
            landmark_points = []
            if landmarks is not None:
                for i in range(0, landmarks.shape[0]):
                    landmark_points.append((int(landmarks[i][0]), int(landmarks[i][1])))
            
            # Create result dictionary
            result = {
                "bbox": (x1, y1, w, h),
                "confidence": float(face.det_score),
                "landmarks": landmark_points
            }
            
            results.append(result)
        
        return results
    
    def _detect_opencv_dnn(self, image: np.ndarray, min_confidence: float) -> List[Dict]:
        """
        Detect faces using OpenCV DNN.
        
        Args:
            image (numpy.ndarray): Input image
            min_confidence (float): Minimum confidence threshold
        
        Returns:
            List[Dict]: List of detected faces
        """
        # Get model parameters
        model_params = self.model_config
        size = model_params.get("size", (300, 300))
        
        # Create blob from image
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, size, [104, 117, 123])
        
        # Set the input and forward pass
        self.net.setInput(blob)
        detections = self.net.forward()
        
        # Process detections
        results = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence < min_confidence:
                continue
            
            # Get coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # Calculate width and height
            width = x2 - x1
            height = y2 - y1
            
            # Skip invalid detections
            if width <= 0 or height <= 0:
                continue
            
            # Create result dictionary
            result = {
                "bbox": (x1, y1, width, height),
                "confidence": float(confidence),
                "landmarks": []  # OpenCV DNN doesn't provide landmarks
            }
            
            results.append(result)
        
        return results
