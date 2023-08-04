import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import pickle

from utils.Face_Recognition import recognize_face,match

st.title("My first Streamlit app")
st.write("Hello, world")


def callback(frame):
    attended_id = []
    frame = frame.to_ndarray(format="bgr24")
    with open('2025_data (4).pkl', 'rb') as f:
        dictionary = pickle.load(f)
    # Process:
    fetures, faces = recognize_face(frame)
    for face, feature in zip(faces, fetures):
        result, user = match(feature,dictionary)
        box = list(map(int, face[:4]))
        color = (0, 255, 0) if result else (0, 0, 255)
        thickness = 2
        cv2.rectangle(frame, box, color, thickness, cv2.LINE_AA)

        id_name, score = user if result else ("unknown", 0.0)
        text = "{0} ({1:.2f})".format(id_name, score)
        position = (box[0], box[1] - 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        cv2.putText(frame, text, position, font, scale, color, thickness, cv2.LINE_AA)
        if id_name not in attended_id:
            attended_id.append(id_name)

    return av.VideoFrame.from_ndarray(frame, format="bgr24")


webrtc_streamer(
    key="example",
    video_frame_callback=callback,
    rtc_configuration={  # Add this line
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

