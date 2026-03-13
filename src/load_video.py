import cv2
import os

video_folder = "data/shanghaitech/training/videos"

videos = os.listdir(video_folder)

print("Total videos:", len(videos))

# pick the first video
video_path = os.path.join(video_folder, videos[0])

cap = cv2.VideoCapture(video_path)

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

print("Total frames in first video:", frame_count)

cap.release()