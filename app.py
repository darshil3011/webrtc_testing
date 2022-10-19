import asyncio
import logging
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple, Optional
import object_detection as detect
import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydub
import streamlit as st
from aiortc.contrib.media import MediaPlayer
import object_detection as detect
import base64

from streamlit_webrtc import (
    RTCConfiguration,
    WebRtcMode,
    WebRtcStreamerContext,
    webrtc_streamer,
)

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def add_bg_from_url(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
    
add_bg_from_url('white.jpg') 

def main():
    st.header("Think Mudra - A voice for dumb/deaf")

    pages = {
        "Real time object detection": app_object_detection,
        
    }
    page_titles = pages.keys()

    page_title = st.sidebar.selectbox(
        "Choose the app mode",
        page_titles,
    )
    
    st.subheader(page_title)
    st.markdown(
        "A Think In Bytes tool that connects a differently-abled person with the world by detecting American Sign Language (ASN) using AI"
        "This tool is still in its native stage and is just a proof of concept of our computer vision capabilities"
    )

    page_func = pages[page_title]
    page_func()

    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f"  {thread.name} ({thread.ident})")


def app_object_detection():
    
    def _annotate_image(image, detections):
        # loop over the detections
        (h, w) = image.shape[:2]
        result: List[Detection] = []
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > confidence_threshold:
                # extract the index of the class label from the `detections`,
                # then compute the (x, y)-coordinates of the bounding box for
                # the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                name = CLASSES[idx]
                result.append(Detection(name=name, prob=float(confidence)))

                # display the prediction
                label = f"{name}: {round(confidence * 100, 2)}%"
                cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(
                    image,
                    label,
                    (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    COLORS[idx],
                    2,
                )
        return image, result

    result_queue = (
        queue.Queue()
    )  # TODO: A general-purpose shared state object may be more useful.

    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        labels, colors, height, width, interpreter = detect.define_tf_lite_model()
        image = frame.to_ndarray(format="rgb24")
        object_detection = detect.display_results(labels, 
                                                      colors, 
                                                      height, 
                                                      width,
                                                      image, 
                                                      interpreter, 
                                                      threshold=0.25)

        return av.VideoFrame.from_ndarray(object_detection, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )





def app_media_constraints():
    """A sample to configure MediaStreamConstraints object"""
    frame_rate = 5
    webrtc_streamer(
        key="media-constraints",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {"frameRate": {"ideal": frame_rate}},
        },
        video_html_attrs={
            "style": {"width": "50%", "margin": "0 auto", "border": "5px yellow solid"},
            "controls": False,
            "autoPlay": True,
        },
    )
    st.write(f"The frame rate is set as {frame_rate}. Video style is changed.")



if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()
    
    st.markdown("<h2 style='text-align: center; color: black;'>Object Detection Applications</h2>", unsafe_allow_html=True)
    image = Image.open('screen3.png')
    st.image(image)
