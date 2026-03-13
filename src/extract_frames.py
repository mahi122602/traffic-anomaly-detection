import cv2
import os

video_folder = "data/shanghaitech/training/videos"
output_folder = "data/frames"

print("Video folder:", video_folder)
print("Folder exists:", os.path.exists(video_folder))

os.makedirs(output_folder, exist_ok=True)

videos = os.listdir(video_folder)
print("Total videos found:", len(videos))

for video in videos:

    video_path = os.path.join(video_folder, video)
    print("Processing:", video_path)

    video_name = os.path.splitext(video)[0]
    frame_folder = os.path.join(output_folder, video_name)

    os.makedirs(frame_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_path = os.path.join(frame_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)

        frame_count += 1

    cap.release()

    print(f"{video} → {frame_count} frames extracted")

print("Frame extraction complete")