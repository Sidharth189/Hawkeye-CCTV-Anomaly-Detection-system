import cv2
import streamlit as st
from crowd_ops import handle_crowd_detection
from trespassing_ops import is_in_zone, handle_trespassing_alert, draw_trespassing_zone
from loitering_ops import is_loitering_active, handle_loitering_alert

def process_frame_annotations(frame, outputs, current_time, track_history, loitering_saved, settings, face_models, trusted_embeddings):
    """
    Orchestrates detection logic by calling specialized modules.
    """
    num_people = 0
    
    for output in outputs:
        # Get DeepSORT output
        x1, y1, x2, y2 = output.to_tlbr()
        track_id = output.track_id
        bbox = (x1, y1, x2, y2)
        
        # Track history update
        if track_id not in track_history:
            track_history[track_id] = [current_time]
        else:
            track_history[track_id].append(current_time)

        # Default Label
        loitering_text = "Normal"
        color = (255, 0, 0) # Blue for normal

        # --- Priority 1: Check Trespassing ---
        if is_in_zone(bbox, settings):
            color = (0, 255, 0) # Green for trespassing
            loitering_text = "Trespassing"
            handle_trespassing_alert(frame, track_id)

        # --- Priority 2: Check Loitering ---
        elif settings['loitering_enabled'] and is_loitering_active(track_id, current_time, track_history, settings['loitering_threshold']):
            color = (0, 0, 255) # Red for loitering
            loitering_text = f"Loitering {current_time - track_history[track_id][0]:.1f}s"
            
            # Delegate complex alert logic to the loitering module
            handle_loitering_alert(
                frame, 
                track_id, 
                loitering_saved, 
                face_models, 
                trusted_embeddings
            )

        # Draw Bounding Box & Label
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"ID {track_id} {loitering_text}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        num_people += 1

    # --- Draw Overlays ---
    draw_trespassing_zone(frame, settings)
    
    # --- Handle Crowd Detection ---
    frame = handle_crowd_detection(frame, num_people, settings)

    return frame
