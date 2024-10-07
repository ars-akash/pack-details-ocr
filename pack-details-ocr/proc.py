import os
from PIL import Image
import cv2 
import time

def input_frames(interval = 5):
    frames = []
    try:         
        cam = cv2.VideoCapture(0)
        start_time = time.time()
        if not cam.isOpened():
                print("Error: Could not open video stream from camera.")
                return None
        while True:
            ret, frame = cam.read()

            cv2.imshow('Camera Feed', frame)
            while (time.time() - start_time) == 5:
                frames.append(frame)

            if cv2.waitKey(1) == ord('q'):
                break

        # Release the capture and writer objects
        cam.release()
        cv2.destroyAllWindows()
    except:
        video_path = input('upload the video: ')
        video = cv2.VideoCapture(video_path)
        suceess, frme = video.read()
        if suceess:
            start_time = time.time()
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            while (time.time() - start_time) == 5:
                frames.append(frame)
    return frames

