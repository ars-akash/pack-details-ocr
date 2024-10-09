'''import cv2
import os

def save_frames_from_video(video_path, output_folder, interval=1):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    print(fps)
    frame_interval = int(fps * interval)
    frame_count = 0
    saved_frame_count = 0
    while True:
        success, frame = video.read()
        if not success:
            break  # End of video
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
        frame_count += 1
    video.release()
    print(f"Saved {saved_frame_count} frames.")

video_path = '/Users/Akash Sahoo/AI/Research/TestData/testvideo1.mp4'  # Path to the video file
output_folder = '/Users/Akash Sahoo/AI/Research/TestData/output_frames'  # Folder to save the frames

save_frames_from_video(video_path, output_folder, interval=1)
'''
'''
import cv2
import os
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

local_model_dir = '/Users/Akash Sahoo/AI/Research/llama-model2'
model = MllamaForConditionalGeneration.from_pretrained(
    local_model_dir,
    torch_dtype = torch.bfloat16,
    device_map="auto")
processor = AutoProcessor.from_pretrained(local_model_dir)

def save_frames_from_video(video_path, output_folder, interval=1):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # video = cv2.VideoCapture(video_path) #upload video
    video = cv2.VideoCapture(0) #from camera
    fps = video.get(cv2.CAP_PROP_FPS)
    print(fps)
    frame_interval = int(fps * interval)
    frame_count = 0
    saved_frame_count = 0
    while True:
        success, frame = video.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        if not success:
            break  # End of video
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

            image = Image.open(frame_filename)

            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Only write, what do you see"}
                ]}
            ]

            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device)
            output = model.generate(**inputs, max_new_tokens=10000)
            print(processor.decode(output[0]))

        frame_count += 1
    video.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved_frame_count} frames.")


video_path = '/Users/Akash Sahoo/AI/Research/TestData/testvideo1.mp4'  # Path to the video file
output_folder = '/Users/Akash Sahoo/AI/Research/TestData/output_frames'  # Folder to save the frames

save_frames_from_video(video_path, output_folder, interval=0.5)
'''

import os
import cv2
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import streamlit as st
import tempfile
from io import BytesIO

# Load your model and processor
local_model_dir = '/Users/AkashSahoo/AI/Research/llama-model2'
model = MllamaForConditionalGeneration.from_pretrained(
    local_model_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(local_model_dir)

# Streamlit UI components
st.title("Video Frame Analysis and Captioning with LLaMA")

# Video file uploader or camera input
video_source = st.radio("Choose video source:", ("Upload video", "Use webcam"))

if video_source == "Upload video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])
else:
    st.write("Using webcam for video capture.")

# Set the frame extraction interval
interval = st.slider("Frame extraction interval (in seconds):", 0.1, 5.0, 1.0)

# Process button
if st.button("Start Processing"):
    if video_source == "Upload video" and uploaded_file is not None:
        # Save the uploaded video temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
    elif video_source == "Use webcam":
        video_path = 0  # 0 is for webcam input
    else:
        st.warning("Please upload a video file or select webcam input.")
        st.stop()

    output_folder = tempfile.mkdtemp()  # Temporary directory to save frames

    def save_frames_from_video(video_path, output_folder, interval=1):
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval)
        frame_count = 0
        saved_frame_count = 0

        # Display the FPS info
        st.write(f"Video FPS: {fps}")

        while True:
            success, frame = video.read()
            if not success:
                break  # End of video

            if frame_count % frame_interval == 0:
                # Save frame as image file
                frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_frame_count += 1

                # Open frame as PIL image for processing
                image = Image.open(frame_filename)

                # Create a chat message with image input
                messages = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Only write, what do you see"}
                    ]}
                ]

                input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

                # Prepare inputs for the model
                inputs = processor(
                    image,
                    input_text,
                    add_special_tokens=False,
                    return_tensors="pt"
                ).to(model.device)

                # Generate the output using the model
                output = model.generate(**inputs, max_new_tokens=10000)
                caption = processor.decode(output[0])

                # Display the extracted frame and caption in the Streamlit app
                st.image(image, caption=f"Frame {saved_frame_count} - Caption: {caption}", use_column_width=True)

            frame_count += 1

        video.release()
        st.write(f"Saved {saved_frame_count} frames.")

    # Call the function to process the video and generate captions
    save_frames_from_video(video_path, output_folder, interval)

