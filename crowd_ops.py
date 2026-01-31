import cv2
import streamlit as st

def handle_crowd_detection(frame, num_people, settings):
    # Always display the people count
    cv2.putText(frame, f"People: {num_people}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Check against threshold
    if settings['crowd_enabled'] and num_people > settings['crowd_threshold']:
        # Draw visual warning on frame
        cv2.putText(frame, "Crowd Detected", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Trigger Streamlit Toast (Logic to prevent spamming every frame)
        if 'crowd_warned' not in st.session_state:
            st.toast(f"Crowd Limit Exceeded! ({num_people} people)", icon="👥")
            st.session_state['crowd_warned'] = True
            
    else:
        # Reset the warning flag if the crowd drops below threshold
        if 'crowd_warned' in st.session_state:
            del st.session_state['crowd_warned']

    return frame
