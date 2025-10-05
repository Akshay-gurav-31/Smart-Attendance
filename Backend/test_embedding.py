#!/usr/bin/env python3
"""
Test script to diagnose embedding generation issues
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from modules.embeddings import EmbeddingModel
from api.face_detect_mediapipe import detect_faces_mediapipe

def test_model_loading():
    """Test if the model can be loaded"""
    print("=== Testing Model Loading ===")
    try:
        embedder = EmbeddingModel()
        print("✓ Model loaded successfully")
        return embedder
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return None

def test_face_detection():
    """Test face detection with a sample image"""
    print("\n=== Testing Face Detection ===")
    # Create a simple test image (you can replace this with a real image path)
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print(f"Created test image with shape: {test_img.shape}")
    
    try:
        faces = detect_faces_mediapipe(test_img)
        print(f"✓ Face detection completed, found {len(faces)} faces")
        return faces
    except Exception as e:
        print(f"✗ Face detection failed: {e}")
        return []

def test_embedding_generation(embedder, face_arrays):
    """Test embedding generation"""
    print("\n=== Testing Embedding Generation ===")
    if not face_arrays:
        print("No face arrays to test with")
        return None
    
    try:
        embedding = embedder.get_average_embedding_from_arrays(face_arrays)
        if embedding is not None:
            print(f"✓ Embedding generated successfully, shape: {embedding.shape}")
            return embedding
        else:
            print("✗ Embedding generation returned None")
            return None
    except Exception as e:
        print(f"✗ Embedding generation failed: {e}")
        return None

def main():
    print("Smart Attendance System - Embedding Diagnostic Test")
    print("=" * 50)
    
    # Test 1: Model loading
    embedder = test_model_loading()
    if embedder is None:
        print("\n❌ Cannot proceed without model. Check model files and dependencies.")
        return
    
    # Test 2: Face detection
    faces = test_face_detection()
    
    # Test 3: Embedding generation
    if faces:
        embedding = test_embedding_generation(embedder, faces)
        if embedding is not None:
            print("\n✅ All tests passed! Embedding system is working.")
        else:
            print("\n❌ Embedding generation failed.")
    else:
        print("\n⚠️  No faces detected for embedding test.")
        print("Try with a real image containing faces.")

if __name__ == "__main__":
    main()
