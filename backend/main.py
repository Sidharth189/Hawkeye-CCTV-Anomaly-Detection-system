import cv2
import torch
import numpy as np
import shutil
import os
import time
import threading
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from collections import defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import sys

# Ensure we can import detection_ops from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import detection_ops
from backend.alert_utils import AlertManager
from backend import api
from pydantic import BaseModel

class LoginRequest(BaseModel):
    username: str
    password: str

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startups ensure directories exist
if not os.path.exists("captured_faces"):
    os.makedirs("captured_faces")
if not os.path.exists("trusted_faces"):
    os.makedirs("trusted_faces")
if not os.path.exists("uploads"):
    os.makedirs("uploads")

app.mount("/captured_faces", StaticFiles(directory="captured_faces"), name="captured_faces")
app.mount("/trusted_faces", StaticFiles(directory="trusted_faces"), name="trusted_faces")

app.include_router(api.router)

# Optimization for Windows Stability
cv2.setNumThreads(0)

# Global State
class VideoState:
    def __init__(self):
        self.video_source = "test_video.mp4" # Default
        self.using_webcam = True
        self.cap = None
        self.reload_cap = True
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        
        # Models
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        print(f"Loading YOLO on {self.device}...")
        self.yolo_model = YOLO("yolov8s.pt")
        print("Loading DeepSort...")
        self.deepsort_tracker = DeepSort(
            max_age=30, n_init=1, nms_max_overlap=1.0, embedder_gpu=self.use_cuda
        )
        
        # Face Recognition
        try:
            from facenet_pytorch import MTCNN, InceptionResnetV1
            from backend import database
            print("Loading Face Recognition Models...")
            self.mtcnn = MTCNN(keep_all=True, device=self.device)
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            
            database.init_db()
            self.known_faces = database.get_trusted_faces()
            print(f"Loaded {len(self.known_faces)} trusted faces.")
        except Exception as e:
            print(f"Face Recognition Init Error: {e}")
            self.mtcnn = None
            self.resnet = None
            self.known_faces = []

        # State trackers
        self.track_history = defaultdict(list)
        self.loitering_saved = defaultdict(lambda: False)
        self.saved_untrusted_session = set()
        self.frame_count = 0
        
        # Statistics
        self.current_occupancy = 0
        self.peak_occupancy = 0
        self.total_alerts = 0
        
        # Settings
        self.settings = {
            'loitering_threshold': 10,
            'crowd_threshold': 60,     
            'confidence_threshold': 0.15, 
            'trespassing_zone': [200, 300, 300, 350],
            'trespassing_enabled': True,
            'loitering_enabled': True,
            'crowd_enabled': True
        }
        
        # Alert Manager
        self.alert_manager = AlertManager()

    def get_capture(self):
        if self.reload_cap or self.cap is None or not self.cap.isOpened():
            if self.cap:
                try:
                    self.cap.release()
                except:
                    pass
                
            source = 0 if self.using_webcam else self.video_source
            print(f"[DEBUG] Opening video source: {source}")
            
            if self.using_webcam:
                # Use DSHOW on Windows for better stability with webcams
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(source)
                
            if not self.cap.isOpened():
                print(f"[ERROR] Failed to open source: {source}")
                self.reload_cap = True
                return None
            
            print(f"[DEBUG] Source opened successfully: {source}")
            self.reload_cap = False
            self.frame_count = 0
            self.track_history.clear()
            self.loitering_saved.clear()
            self.saved_untrusted_session.clear()
            self.deepsort_tracker.delete_all_tracks()
            
            if self.mtcnn:
                from backend import database
                self.known_faces = database.get_trusted_faces()
            
        return self.cap

    def process_video(self):
        print("[DEBUG] Background video processing loop started.")
        while self.running:
            cap = self.get_capture()
            if not cap:
                time.sleep(1)
                continue
                
            ret, frame = cap.read()
            if not ret:
                if not self.using_webcam:
                    self.frame_count = 0
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("[DEBUG] Capture read failed, retrying in 1s...")
                    self.reload_cap = True
                    time.sleep(1)
                    continue
                    
            self.frame_count += 1
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            current_time = self.frame_count / fps
            
            # Run Detection
            detections_list = []
            results = self.yolo_model(
                frame, stream=True, conf=self.settings['confidence_threshold'], 
                device=self.device if self.device == 'cpu' else 0, imgsz=640, verbose=False
            )
            
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    cls_id = int(box.cls[0])
                    if cls_id == 0: # Person
                        w = x2 - x1
                        h = y2 - y1
                        detections_list.append([[x1, y1, w, h], conf, 0])
            
            # Tracker
            tracks = self.deepsort_tracker.update_tracks(detections_list, frame=frame)
            
            # Annotate
            curr_settings = self.settings.copy()
            curr_settings['trespassing_zone'] = tuple(self.settings['trespassing_zone'])
            
            final_frame, alerts, self.saved_untrusted_session = detection_ops.process_frame_annotations(
                frame, tracks, current_time, 
                self.track_history, self.loitering_saved, curr_settings,
                mtcnn=self.mtcnn,
                resnet=self.resnet,
                known_faces=self.known_faces,
                device=self.device,
                saved_untrusted_session=self.saved_untrusted_session
            )
            
            # Process Alerts
            self.alert_manager.process_alerts(alerts)
            
            # Update Stats
            self.current_occupancy = alerts['count']
            if self.current_occupancy > self.peak_occupancy:
                self.peak_occupancy = self.current_occupancy
            
            # Encoding
            ret, buffer = cv2.imencode('.jpg', final_frame)
            if ret:
                with self.lock:
                    self.latest_frame = buffer.tobytes()
            
            # Small sleep to yield
            time.sleep(0.01)

state = VideoState()

@app.on_event("startup")
async def startup_event():
    threading.Thread(target=state.process_video, daemon=True).start()

def generate_frames():
    while True:
        if state.latest_frame is not None:
            with state.lock:
                frame_bytes = state.latest_frame
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Sleep to match typical camera high-end FPS or just yield
        time.sleep(0.03) # ~30 FPS output

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/settings")
def get_settings():
    return state.settings

@app.post("/settings")
def update_settings(new_settings: dict):
    state.settings.update(new_settings)
    return {"status": "updated", "settings": state.settings}

@app.get("/stats")
def get_stats():
    alerts = state.alert_manager.recent_alerts if hasattr(state.alert_manager, 'recent_alerts') else []
    return {
        "occupancy": state.current_occupancy,
        "peak_occupancy": state.peak_occupancy,
        "total_alerts": state.alert_manager.alert_count,
        "alerts": alerts
    }

@app.post("/set_source")
def set_source(source_type: str = Form(...)):
    if source_type == 'webcam':
        state.using_webcam = True
    else:
        state.using_webcam = False
    state.reload_cap = True
    return {"status": "source_changed", "type": source_type}

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    file_location = os.path.abspath(f"uploads/{file.filename}")
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    state.video_source = file_location
    state.using_webcam = False
    state.reload_cap = True
    return {"status": "file_uploaded", "filename": file.filename}

@app.post("/login")
def login(creds: LoginRequest):
    admin_user = os.getenv("ADMIN_USERNAME", "admin")
    admin_pass = os.getenv("ADMIN_PASSWORD", "admin")
    
    if creds.username == admin_user and creds.password == admin_pass:
        return {"token": "authenticated_session_token", "message": "Login successful"}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
