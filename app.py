import streamlit as st
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from datetime import datetime
import os
import torch
from PIL import Image
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1

# Streamlit UI setup
st.set_page_config(page_title="Loitering & Crowd Detection", layout="wide")
st.title("hawkeye")

# Sidebar for navigation
page = st.sidebar.radio("Navigation", ["Home", "View Detected Faces"])

# Initialize YOLO and DeepSORT
yolo_model = YOLO("yolov8n.pt")
deepsort_tracker = DeepSort(max_age=20, n_init=2, nms_max_overlap=1.0)

# Initialize MTCNN and FaceNet
detector = MTCNN()
facenet = InceptionResnetV1(pretrained='vggface2').eval()

# Load trusted faces
trusted_faces_dir = "trusted_faces"
trusted_face_embeddings = []
output_dir = "detected_faces_facenet"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def get_face_embedding(face_image):
    face_image = Image.fromarray(face_image).resize((160, 160))
    face_tensor = torch.tensor(np.array(face_image)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        embedding = facenet(face_tensor)
    return embedding

for filename in os.listdir(trusted_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(trusted_faces_dir, filename)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(image_rgb)
        if len(faces) > 0:
            x, y, width, height = faces[0]['box']
            face_image = image_rgb[y:y+height, x:x+width]
            embedding = get_face_embedding(face_image)
            trusted_face_embeddings.append((filename, embedding))

# Function to display detected faces
def view_detected_faces():
    st.header("Detected Faces")
    if not os.path.exists(output_dir) or len(os.listdir(output_dir)) == 0:
        st.warning("No detected faces found.")
        return

    # List all detected face images
    detected_faces = os.listdir(output_dir)
    for face_image in detected_faces:
        face_path = os.path.join(output_dir, face_image)
        st.image(face_path, caption=face_image, width=300)

# Home page for video processing
if page == "Home":
    # Sidebar for user inputs
    st.sidebar.header("Settings")
    LOITERING_THRESHOLD = st.sidebar.slider("Loitering Threshold (seconds)", 5, 30, 10)
    CROWD_THRESHOLD = st.sidebar.slider("Crowd Threshold (number of people)", 2, 10, 5)
    CONFIDENCE_THRESHOLD = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4)

    # Define trespassing zone using two sliders for top-left and bottom-right coordinates
    st.sidebar.header("Trespassing Zone Settings")
    trespassing_x1 = st.sidebar.slider("Trespassing Zone X1", 0, 640, 100)
    trespassing_y1 = st.sidebar.slider("Trespassing Zone Y1", 0, 480, 100)
    trespassing_x2 = st.sidebar.slider("Trespassing Zone X2", 0, 640, 500)
    trespassing_y2 = st.sidebar.slider("Trespassing Zone Y2", 0, 480, 400)

    # Combine into a trespassing zone tuple
    TRESPASSING_ZONE = (trespassing_x1, trespassing_y1, trespassing_x2, trespassing_y2)

    # Toggle button for trespassing detection (in the sidebar)
    trespassing_button = st.sidebar.toggle(
        "Turn Trespassing Detection On" if not st.session_state.get("trespassing_enabled", False) else "Turn Trespassing Detection Off",
        value=st.session_state.get("trespassing_enabled", False),
        key="trespassing_toggle"
    )

    # Update state variable when button is clicked
    st.session_state.trespassing_enabled = trespassing_button

    # Toggle buttons for loitering and crowd detection (in the sidebar)
    loitering_enabled = st.sidebar.toggle(
        "Turn Loitering Detection On" if not st.session_state.get("loitering_enabled", False) else "Turn Loitering Detection Off",
        value=st.session_state.get("loitering_enabled", False),
        key="loitering_toggle"
    )

    crowd_enabled = st.sidebar.toggle(
        "Turn Crowd Detection On" if not st.session_state.get("crowd_enabled", False) else "Turn Crowd Detection Off",
        value=st.session_state.get("crowd_enabled", False),
        key="crowd_toggle"
    )

    # Update state variables when toggles are clicked
    st.session_state.loitering_enabled = loitering_enabled
    st.session_state.crowd_enabled = crowd_enabled

    # Open the camera stream
    cap = cv2.VideoCapture(1)  # Use 0 for default camera, or replace with IP camera URL

    if not cap.isOpened():
        st.error("Error: Could not open camera stream.")
        st.stop()

    # Frame dimensions and FPS
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Track history and loitering status
    track_history = defaultdict(lambda: [])
    loitering_saved = defaultdict(lambda: False)

    # Function to check loitering
    def is_loitering(track_id, current_time):
        if track_id in track_history:
            time_in_area = current_time - track_history[track_id][0]
            return time_in_area >= LOITERING_THRESHOLD
        return False

    # Function to draw bounding boxes and alerts
    def draw_boxes(frame, outputs, current_time):
        num_people = 0
        face_count = 0
        for output in outputs:
            x1, y1, x2, y2 = output.to_tlbr()
            track_id = output.track_id

            # Update track history
            if track_id not in track_history:
                track_history[track_id] = [current_time]
            else:
                track_history[track_id].append(current_time)

            # Initialize loitering_text with a default value
            loitering_text = "Normal"
            color = (255, 0, 0)  # Default color (blue for normal)

            # Check for trespassing (if enabled)
            if st.session_state.get("trespassing_enabled", False):
                if (x1 < TRESPASSING_ZONE[2] and x2 > TRESPASSING_ZONE[0] and
                    y1 < TRESPASSING_ZONE[3] and y2 > TRESPASSING_ZONE[1]):
                    trespassing = True
                    color = (0, 255, 0)  # Green for trespassing
                    loitering_text = "Trespassing"
                    
                    # Save frame if trespassing is detected
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"trespassing_{track_id}_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    st.warning(f"Trespassing detected! Saved frame as {filename}")

            # Check for loitering (if enabled)
            elif st.session_state.loitering_enabled and is_loitering(track_id, current_time):
                color = (0, 0, 255)  # Red for loitering
                loitering_text = f"Loitering {current_time - track_history[track_id][0]:.1f}s"

                # Save frame if loitering is detected for the first time
                if not loitering_saved[track_id]:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"loitering_{track_id}_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    faces = detector.detect_faces(rgb_frame)
                    for face in faces:
                        x, y, width, height = face['box']
                        confidence = face['confidence']
                        if confidence < 0.9:
                            continue
                        face_image = rgb_frame[y:y+height, x:x+width]
                        embedding = get_face_embedding(face_image)
                        is_trusted = False
                        for trusted_name, trusted_embedding in trusted_face_embeddings:
                            distance = torch.dist(embedding, trusted_embedding).item()
                            if distance < 1.0:
                                is_trusted = True
                                break
                        if not is_trusted:
                            # Save face with timestamp as filename
                            face_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # Include milliseconds
                            face_filename = os.path.join(output_dir, f"face_{face_timestamp}.jpg")
                            cv2.imwrite(face_filename, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
                            face_count += 1
                    st.warning(f"Loitering detected! Saved frame as {filename}")
                    loitering_saved[track_id] = True

            # Draw bounding box and text
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"ID {track_id} {loitering_text}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Draw trespassing zone
            if st.session_state.get("trespassing_enabled", False):
                cv2.rectangle(frame, (TRESPASSING_ZONE[0], TRESPASSING_ZONE[1]),
                              (TRESPASSING_ZONE[2], TRESPASSING_ZONE[3]), (0, 255, 0), 2)
                cv2.putText(frame, "Trespassing Zone", (TRESPASSING_ZONE[0], TRESPASSING_ZONE[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            num_people += 1

        # Display the number of people detected
        cv2.putText(frame, f"People: {num_people}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Check for crowd detection (if enabled)
        if st.session_state.crowd_enabled and num_people > CROWD_THRESHOLD:
            cv2.putText(frame, "Crowd Detected", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            st.warning("Crowd detected!")

        return frame

    # Main Streamlit app
    st_frame = st.empty()  # Placeholder for the video frame
    frame_count = 0
    outputs = []

    # Centered Start and Stop buttons
    col1, col2, col3 = st.columns([1, 2, 1])  # Create 3 columns, with the middle column wider
    with col2:  # Place buttons in the middle column
        start_button = st.button("Start Video Processing")
        stop_button = st.button("Stop Video Processing")

    # Variable to control video processing
    processing = False

    if start_button:
        processing = True
        st.success("Video processing started.")

    if stop_button:
        processing = False
        st.warning("Video processing stopped.")

    # Create a 4-camera layout
    col11, col12 = st.columns(2)
    col13, col14 = st.columns(2)

    # Placeholders for each camera feed
    cam1_placeholder = col11.empty()
    cam2_placeholder = col12.empty()
    cam3_placeholder = col13.empty()
    cam4_placeholder = col14.empty()

    # Local image for disconnected cameras
    local_image_path = "ss.jfif"  # Replace with your local image path

    # Video processing loop
    while processing:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Failed to read frame from camera.")
            break

        frame_count += 1
        current_time = frame_count / fps

        # Resize the frame for faster processing
        resized_frame = cv2.resize(frame, (640, 480))

        # Run YOLOv8 object detection every 5th frame
        if frame_count % 5 == 0:
            results = yolo_model(resized_frame, stream=True, conf=CONFIDENCE_THRESHOLD)
            detections = []
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    confidence = box.conf[0]
                    class_id = box.cls[0]
                    if class_id == 0:  # Filter for person class
                        detections.append([x1, y1, x2, y2, confidence])

            # Prepare detections for DeepSORT
            if len(detections) > 0:
                detections_list = []
                for detection in detections:
                    x1, y1, x2, y2, confidence = detection
                    detections_list.append([[x1, y1, x2 - x1, y2 - y1], confidence, 0])
            else:
                detections_list = []

            # Update tracker with detected objects
            outputs = deepsort_tracker.update_tracks(detections_list, frame=resized_frame)

        # Draw tracked objects and loitering status on the frame
        frame = draw_boxes(frame, outputs, current_time)

        # Convert the frame to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the input video feed in Cam 1
        cam1_placeholder.caption("Cam 1: Real-Time Video Feed")
        cam1_placeholder.image(frame_rgb, channels="RGB", use_container_width=False, width=400)

        # Simulate disconnected cameras for Cam 2, Cam 3, and Cam 4
        cam2_placeholder.caption("Cam 2: Camera Disconnected")
        cam2_placeholder.image(local_image_path, use_container_width=False, width=400)

        cam3_placeholder.caption("Cam 3: Camera Disconnected")
        cam3_placeholder.image(local_image_path, use_container_width=False, width=400)

        cam4_placeholder.caption("Cam 4: Camera Disconnected")
        cam4_placeholder.image(local_image_path, use_container_width=False, width=400)

        # Break the loop if the user stops the app
        if not processing:
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# View Detected Faces page
elif page == "View Detected Faces":
    view_detected_faces()
