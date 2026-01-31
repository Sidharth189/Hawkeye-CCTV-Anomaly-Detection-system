import cv2
from datetime import datetime
import streamlit as st

def draw_trespassing_zone(frame, settings):
    """Draws the visual boundary of the trespassing zone on the frame."""
    if settings.get('trespassing_enabled'):
        zone = settings['trespassing_zone']
        cv2.rectangle(frame, (zone[0], zone[1]), (zone[2], zone[3]), (0, 255, 0), 2)
        cv2.putText(frame, "Trespassing Zone", (zone[0], zone[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def is_in_zone(bbox, settings):
    """Checks if the bounding box overlaps with the trespassing zone."""
    if not settings.get('trespassing_enabled'):
        return False
    
    x1, y1, x2, y2 = bbox
    tx1, ty1, tx2, ty2 = settings['trespassing_zone']
    
    # Check for overlap
    return (x1 < tx2 and x2 > tx1 and y1 < ty2 and y2 > ty1)

def handle_trespassing_alert(frame, track_id):
    """Saves the frame and triggers a UI alert for trespassing."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"trespassing_{track_id}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    st.toast(f"Trespassing detected! ID: {track_id}", icon="⚠️")
