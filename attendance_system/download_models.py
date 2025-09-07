"""
Script to download required models for the face recognition attendance system.
"""
import os
import sys
import shutil
import argparse
import requests
from tqdm import tqdm
import zipfile
import tarfile
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("⚠️ InsightFace not available - using OpenCV DNN models only")
import urllib.request

from attendance_system.config.settings import MODEL_DIR
from attendance_system.config.model_config import MODEL_URLS, MODELS


def download_file(url, target_path, description=None):
    """
    Download a file with progress bar.
    
    Args:
        url (str): URL to download
        target_path (str): Path to save the file
        description (str): Description for the progress bar
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # Download with progress bar
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        desc = description or f"Downloading {os.path.basename(target_path)}"
        
        with open(target_path, 'wb') as f, tqdm(
                desc=desc,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def download_insightface_models():
    """Download InsightFace models."""
    print("\nDownloading InsightFace models...")
    
    if not INSIGHTFACE_AVAILABLE:
        print("⚠️ InsightFace not available - skipping InsightFace models")
        return False
    
    try:
        # Initialize FaceAnalysis to trigger the download
        app = FaceAnalysis(providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("InsightFace models downloaded successfully")
        return True
    except Exception as e:
        print(f"Error downloading InsightFace models: {e}")
        return False


def download_opencv_dnn_models():
    """Download OpenCV DNN face detection models."""
    print("\nDownloading OpenCV DNN models...")
    
    # URLs for OpenCV DNN models
    opencv_dnn_model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    opencv_dnn_config_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    
    # Target paths
    model_config = MODELS["opencv_dnn"]
    model_path = model_config["model_file"]
    config_path = model_config["config_file"]
    
    # Create directory
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Download model
    success1 = download_file(
        opencv_dnn_model_url, 
        model_path, 
        "Downloading OpenCV DNN model"
    )
    
    # Download config
    success2 = download_file(
        opencv_dnn_config_url, 
        config_path, 
        "Downloading OpenCV DNN config"
    )
    
    if success1 and success2:
        print("OpenCV DNN models downloaded successfully")
        return True
    else:
        print("Failed to download some OpenCV DNN model files")
        return False


def download_facenet_model():
    """Download FaceNet model."""
    print("\nDownloading FaceNet model...")
    
    # URL for FaceNet model
    facenet_url = "https://storage.googleapis.com/tensorflow/tf-keras-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
    
    # Target path
    model_path = MODELS["facenet"]["model_file"]
    
    # Create directory
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Download model
    success = download_file(
        facenet_url, 
        model_path, 
        "Downloading FaceNet model"
    )
    
    if success:
        print("FaceNet model downloaded successfully")
        return True
    else:
        print("Failed to download FaceNet model")
        return False


def download_all_models():
    """Download all required models."""
    print("Downloading all required models for the face recognition attendance system...")
    
    # Make sure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Download InsightFace models
    insightface_success = download_insightface_models()
    
    # Download OpenCV DNN models
    opencv_success = download_opencv_dnn_models()
    
    # Download FaceNet model
    facenet_success = download_facenet_model()
    
    # Download models from custom URLs
    custom_success = True
    for model_name, url in MODEL_URLS.items():
        if url and model_name in MODELS:
            model_path = MODELS[model_name]["model_file"]
            if not os.path.exists(model_path):
                print(f"\nDownloading {model_name} model...")
                success = download_file(url, model_path)
                if not success:
                    custom_success = False
    
    # Summary
    print("\nModel Download Summary:")
    print(f"- InsightFace models: {'Success' if insightface_success else 'Failed'}")
    print(f"- OpenCV DNN models: {'Success' if opencv_success else 'Failed'}")
    print(f"- FaceNet model: {'Success' if facenet_success else 'Failed'}")
    print(f"- Custom models: {'Success' if custom_success else 'Partial or Failed'}")
    
    return insightface_success and opencv_success and facenet_success and custom_success


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download required models for the face recognition attendance system")
    parser.add_argument("--insightface", action="store_true", help="Download only InsightFace models")
    parser.add_argument("--opencv", action="store_true", help="Download only OpenCV DNN models")
    parser.add_argument("--facenet", action="store_true", help="Download only FaceNet model")
    
    args = parser.parse_args()
    
    if args.insightface:
        download_insightface_models()
    elif args.opencv:
        download_opencv_dnn_models()
    elif args.facenet:
        download_facenet_model()
    else:
        download_all_models()


if __name__ == "__main__":
    main() 