import streamlit as st
from PIL import Image
import tempfile
import cv2
import os

# Title of the Streamlit app
st.title("Video/Photo/File Input App")

# Section for uploading a file
st.header("Upload a File (Video/Image)")
file_input = st.file_uploader("Choose a file (video or image)", type=["jpg", "jpeg", "png", "mp4", "mov"])

# Section for taking a photo
st.header("Take a Photo")
photo_input = st.camera_input("Capture an image")

# Section for uploading a video using camera or file upload
st.header("Record a Video (using file upload)")
video_input = st.file_uploader("Choose a video file", type=["mp4", "mov"])

# Handling file input
if file_input is not None:
    # Check if the uploaded file is an image
    if file_input.type.startswith('image'):
        # Display the uploaded image
        image = Image.open(file_input)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    elif file_input.type.startswith('video'):
        # Display the uploaded video
        st.video(file_input)
    else:
        st.write("Unsupported file format!")

# Handling photo input from the camera
if photo_input is not None:
    # Display the captured photo
    image = Image.open(photo_input)
    st.image(image, caption="Captured Image", use_column_width=True)

# Handling video input from file upload
if video_input is not None:
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(video_input.read())
        temp_video_path = temp_file.name

    # Display the uploaded video
    st.video(temp_video_path)

    # Process the video with OpenCV
    st.write("Video saved to temporary location: ", temp_video_path)

    # Optionally, process the video using OpenCV (example code below)
    cap = cv2.VideoCapture(temp_video_path)

    # Read and display video frames
    st.write("Reading video frames...")
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Convert OpenCV frame to RGB (Streamlit displays RGB images)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Show every 20th frame
        if frame_count % 20 == 0:
            st.image(frame_rgb, caption=f"Frame {frame_count}", use_column_width=True)

    cap.release()

# Streamlit running instructions: save this script and run `streamlit run script_name.py` from the command line
