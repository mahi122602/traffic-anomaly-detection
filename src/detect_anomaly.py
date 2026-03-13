import os
import cv2
import torch
import numpy as np
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt

from model import Autoencoder

print("Starting anomaly detection...")

# Load model
model = Autoencoder()
model.load_state_dict(torch.load("models/anomaly_model.pth", map_location=torch.device("cpu")))
model.eval()

print("Model loaded successfully")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

frame_folder = "data/shanghaitech/testing/frames"

videos = os.listdir(frame_folder)

print("Testing videos found:", videos[:5])

criterion = nn.MSELoss()

threshold = 0.00145

# Create result folders
os.makedirs("results/anomalies", exist_ok=True)
os.makedirs("results/graphs", exist_ok=True)
os.makedirs("results/videos", exist_ok=True)

# Lists for evaluation
all_scores = []
all_labels = []

for video in videos:

    video_path = os.path.join(frame_folder, video)

    frames = sorted(os.listdir(video_path))

    if len(frames) == 0:
        continue

    # Initialize video writer
    first_frame_path = os.path.join(video_path, frames[0])
    first_frame = cv2.imread(first_frame_path)

    if first_frame is None:
        continue

    height, width, _ = first_frame.shape

    video_writer = cv2.VideoWriter(
        f"results/videos/{video}_output.avi",
        cv2.VideoWriter_fourcc(*'XVID'),
        20,
        (width, height)
    )

    print(f"\nProcessing video: {video}")

    loss_values = []

    for i, frame_name in enumerate(frames):

        frame_path = os.path.join(video_path, frame_name)

        frame = cv2.imread(frame_path)

        if frame is None:
            continue

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = transform(img)
        img = img.unsqueeze(0)

        with torch.no_grad():
            output = model(img)

        loss = criterion(output, img).item()

        loss_values.append(loss)

        # Save score for evaluation
        all_scores.append(loss)

        print(f"{video} Frame {i} | Loss: {loss:.6f}")

        if loss > threshold:

            all_labels.append(1)

            print(f"Anomaly detected in {video} frame {i} | Loss: {loss:.6f}")

            cv2.putText(frame,
                        "ANOMALY DETECTED",
                        (30,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        3)

            cv2.putText(frame,
                        f"Loss: {loss:.5f}",
                        (30,90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,0,255),
                        2)

            save_path = f"results/anomalies/{video}_frame_{i}.jpg"
            cv2.imwrite(save_path, frame)

        else:
            all_labels.append(0)

        # Write frame to video
        video_writer.write(frame)

    # Save anomaly score graph
    plt.figure()
    plt.plot(loss_values)
    plt.axhline(y=threshold, linestyle="--")
    plt.title(f"Anomaly Scores for {video}")
    plt.xlabel("Frame Number")
    plt.ylabel("Reconstruction Loss")

    graph_path = f"results/graphs/{video}_scores.png"
    plt.savefig(graph_path)
    plt.close()

    print(f"Graph saved: {graph_path}")

    video_writer.release()

# Save scores for evaluation
np.save("results/anomaly_scores.npy", np.array(all_scores))
np.save("results/labels.npy", np.array(all_labels))

print("Anomaly scores saved for evaluation")

print("\nDetection complete")