import os
import cv2
import torch
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1
import streamlit as st

TRUSTED_FACES_DIR = "trusted_faces"
OUTPUT_DIR = "detected_faces_facenet"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def init_face_models():
    detector = MTCNN()
    facenet = InceptionResnetV1(pretrained='vggface2').eval()
    return detector, facenet

def get_face_embedding(face_image, facenet_model):
    face_image = Image.fromarray(face_image).resize((160, 160))
    face_tensor = torch.tensor(np.array(face_image)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        embedding = facenet_model(face_tensor)
    return embedding

def load_trusted_faces(detector, facenet):
    trusted_face_embeddings = []
    if not os.path.exists(TRUSTED_FACES_DIR):
        os.makedirs(TRUSTED_FACES_DIR)
        return trusted_face_embeddings

    for filename in os.listdir(TRUSTED_FACES_DIR):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(TRUSTED_FACES_DIR, filename)
            image = cv2.imread(image_path)
            if image is None: continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(image_rgb)
            
            if len(faces) > 0:
                x, y, width, height = faces[0]['box']
                face_image = image_rgb[y:y+height, x:x+width]
                embedding = get_face_embedding(face_image, facenet)
                trusted_face_embeddings.append((filename, embedding))
    
    return trusted_face_embeddings

def view_detected_faces_ui():
    st.header("Detected Faces")
    if not os.path.exists(OUTPUT_DIR) or len(os.listdir(OUTPUT_DIR)) == 0:
        st.warning("No detected faces found.")
        return

    detected_faces = os.listdir(OUTPUT_DIR)
    for face_image in detected_faces:
        face_path = os.path.join(OUTPUT_DIR, face_image)
        st.image(face_path, caption=face_image, width=300)
