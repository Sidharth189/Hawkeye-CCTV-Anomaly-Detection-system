import cv2
import numpy as np
import torch
from PIL import Image
import os
import uuid
from backend import database

# --- Colors ---
MAROON = (0, 0, 128)      
RED_ALERT = (0, 0, 255)  
ZONE_COLOR = (0, 0, 255)  
GREEN_SAFE = (0, 255, 0)

def check_trespassing(bbox, zone_coords):
    x1, y1, x2, y2 = bbox
    # Check if the "feet" are in the zone
    foot_x, foot_y = int((x1 + x2) / 2), int(y2)
    zx1, zy1, zx2, zy2 = zone_coords
    return zx1 < foot_x < zx2 and zy1 < foot_y < zy2

def check_loitering(track_id, center_point, track_history, current_time, threshold):
    if track_id not in track_history:
        track_history[track_id].append((current_time, center_point))
        return False
    
    first_time = track_history[track_id][0][0]
    duration = current_time - first_time
    track_history[track_id].append((current_time, center_point))

    if duration > threshold:
        return True
    return False

def recognize_frame_faces(frame, tracks, mtcnn, resnet, known_faces, device, saved_untrusted=None):
    """
    frame: cv2 image (BGR)
    tracks: deepsort tracks
    known_faces: list of {name, embedding}
    """
    if saved_untrusted is None:
        saved_untrusted = set()

    if mtcnn is None or resnet is None:
        return [], saved_untrusted
        
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    frame_h, frame_w = frame.shape[:2]
    
    results = []

    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
            
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
        
        # Clamp coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_w, x2), min(frame_h, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue

        person_crop = pil_img.crop((x1, y1, x2, y2))
        
        # Detect face in person crop
        # MTCNN expects PIL image
        try:
            boxes, _ = mtcnn.detect(person_crop)
        except (ValueError, RuntimeError, IndexError) as e:
            # Common facenet-pytorch error: "ValueError: torch.cat(): expected a non-empty list of Tensors"
            # Logic: If it sees "something" but filters it all out, it might crash on concatenation.
            # safe to ignore and assume no face found.
            continue
        except Exception as e:
            print(f"Unexpected error in face detection: {e}")
            continue
        
        if boxes is not None:
            # Sort by largest face if multiple (unlikely in a person crop, but possible)
            # Take the largest face
            # mtcnn.detect returns boxes relative to crop
            
            # For simplicity, we process the first valid face
            for box in boxes:
                fx1, fy1, fx2, fy2 = box
                
                # Check face resolution
                if (fx2-fx1) < 20 or (fy2-fy1) < 20: 
                    continue
                
                # Get the face crop again from the person crop for embedding
                # (Or use mtcnn implicit cropping with forward pass, but we want visual boxes)
                # Let's align and crop using MTCNN's functionality is standard but 
                # we want to control the flow.
                
                # Manual crop for embedding
                face_crop_pil = person_crop.crop((fx1, fy1, fx2, fy2))
                
                # Preprocess for ResNet
                # Standard InceptionResnetV1 transforms: resize close to 160x160 usually, 
                # but let's check what facenet_pytorch expects (whiten=True by default on forward?)
                # We can use mtcnn to return tensors
                
                # To be efficient, let's just use the crop we just made.
                # Only downside is alignment. MTCNN provides alignment.
                # Let's re-run mtcnn on crop to get tensor if we want alignment, 
                # but that's expensive.
                # Let's just standard resize.
                
                try:
                    face_tensor = torch.from_numpy(np.array(face_crop_pil.resize((160, 160)))).permute(2, 0, 1).float()
                    face_tensor = (face_tensor - 127.5) / 128.0 # Standard normalization for facenet
                    face_tensor = face_tensor.unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        embedding = resnet(face_tensor).detach().cpu().numpy()[0]
                    
                    # Compare with known faces
                    name = "Unknown"
                    is_trusted = False
                    min_dist = 0.8 # Threshold usually around 0.6-1.0 depending on use case
                    
                    for kf in known_faces:
                        known_emb = np.array(kf['embedding'])
                        dist = np.linalg.norm(embedding - known_emb)
                        if dist < min_dist:
                            min_dist = dist
                            name = kf['name']
                            is_trusted = True
                    
                    # Store result relative to full frame
                    abs_fx1 = int(x1 + fx1)
                    abs_fy1 = int(y1 + fy1)
                    abs_fx2 = int(x1 + fx2)
                    abs_fy2 = int(y1 + fy2)
                    
                    results.append({
                        "box": (abs_fx1, abs_fy1, abs_fx2, abs_fy2),
                        "name": name,
                        "trusted": is_trusted
                    })

                    # Handle Untrusted Capture
                    if not is_trusted:
                        # Avoid saving same person repeatedly too fast?
                        # Using track_id to throttle could work if passed down.
                        # For now, just save.
                        
                        # Generate unique filename
                        # Use timestamp or uuid
                        if track.track_id not in saved_untrusted: # Simple session throttle per track
                            filename = f"capture_{uuid.uuid4().hex}.jpg"
                            fpath = os.path.join("backend/captured_faces", filename)
                            # Save face crop or full person or full frame?
                            # Request says "captured ... as untrusted"
                            # Usually saving the face crop + maybe context
                            # Let's save the face crop (resized/orig)
                            
                            # Convert face_crop_pil back to BGR for cv2 save
                            save_img = cv2.cvtColor(np.array(face_crop_pil), cv2.COLOR_RGB2BGR)
                            cv2.imwrite(fpath, save_img)
                            database.log_untrusted_face(filename)
                            saved_untrusted.add(track.track_id)

                except Exception as e:
                    print(f"Face processing error: {e}")
                    pass

    return results, saved_untrusted

def process_frame_annotations(frame, tracks, current_time, track_history, loitering_saved, settings, 
                              mtcnn=None, resnet=None, known_faces=None, device='cpu', saved_untrusted_session=None):
    
    annotated_frame = frame.copy()
    
    # 1. Draw Restricted Zone
    if settings['trespassing_enabled']:
        tz = settings['trespassing_zone']
        
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (tz[0], tz[1]), (tz[2], tz[3]), ZONE_COLOR, -1)
        
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
        
        cv2.rectangle(annotated_frame, (tz[0], tz[1]), (tz[2], tz[3]), MAROON, 2)
        cv2.putText(annotated_frame, "Restricted Zone", (tz[0], tz[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, MAROON, 2)

    frame_alerts = {
        'count': 0,
        'trespassing': False,
        'loitering': False,
        'crowd': False
    }

    # Run Face Recognition if models provided
    face_results = []
    if mtcnn and resnet:
        face_results, saved_untrusted_session = recognize_frame_faces(
            frame, tracks, mtcnn, resnet, known_faces, device, saved_untrusted_session
        )

    for track in tracks:
        if not track.is_confirmed() and track.time_since_update > 1:
            continue
        
        frame_alerts['count'] += 1
        track_id = track.track_id
        
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
        bbox = (x1, y1, x2, y2)
        center = (int((x1+x2)/2), int((y1+y2)/2))

        # Check Trespassing
        if settings['trespassing_enabled'] and check_trespassing(bbox, settings['trespassing_zone']):
            frame_alerts['trespassing'] = True

        # Check Loitering
        if settings['loitering_enabled']:
            if check_loitering(track_id, center, track_history, current_time, settings['loitering_threshold']):
                frame_alerts['loitering'] = True
                loitering_saved[track_id] = True
    
    # Check crowd
    if settings['crowd_enabled'] and frame_alerts['count'] > settings['crowd_threshold']:
        frame_alerts['crowd'] = True


    # Draw Stats
    y_pos = 20
    cv2.putText(annotated_frame, f"People Count: {frame_alerts['count']}", (10, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, MAROON, 2)
    
    if frame_alerts['crowd']:
        y_pos += 20
        cv2.putText(annotated_frame, "Crowd Alert!", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED_ALERT, 2)

    if frame_alerts['loitering']:
        y_pos += 20
        cv2.putText(annotated_frame, "Loitering Alert!", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED_ALERT, 2)

    if frame_alerts['trespassing']:
        y_pos += 20
        cv2.putText(annotated_frame, "Trespassing Alert!", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED_ALERT, 2)
    
    # Draw Faces
    for res in face_results:
        fx1, fy1, fx2, fy2 = res['box']
        color = GREEN_SAFE if res['trusted'] else RED_ALERT
        label = res['name']
        cv2.rectangle(annotated_frame, (fx1, fy1), (fx2, fy2), color, 2)
        cv2.putText(annotated_frame, label, (fx1, fy1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return annotated_frame, frame_alerts, saved_untrusted_session