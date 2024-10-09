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
