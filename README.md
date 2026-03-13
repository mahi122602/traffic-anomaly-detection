# Traffic Anomaly Detection Using Deep Learning

## Overview

This project implements a deep learning–based approach for detecting abnormal events in traffic surveillance videos. The system is designed to learn patterns of normal traffic behavior and identify frames that deviate from those patterns.

The model is trained using an unsupervised learning approach based on a convolutional autoencoder. During training, the network learns to reconstruct frames representing normal traffic conditions. When the trained model processes unseen data, frames containing unusual activity produce higher reconstruction errors, which are used as indicators of anomalies.

The objective of this project is to demonstrate a complete machine learning pipeline for anomaly detection in video data, including data preprocessing, model training, inference, and result visualization.

---

## Methodology

The system follows a reconstruction-based anomaly detection strategy.

1. Video frames are extracted from surveillance footage.
2. An autoencoder is trained on frames representing normal traffic activity.
3. The model learns a compressed representation of normal visual patterns.
4. During inference, each frame is reconstructed by the trained model.
5. The reconstruction error is calculated for every frame.
6. Frames with reconstruction errors exceeding a predefined threshold are flagged as anomalies.

This approach enables the detection of abnormal events without requiring labeled anomaly data.

---

## System Pipeline

Video Input
→ Frame Extraction
→ Autoencoder Training
→ Frame Reconstruction
→ Reconstruction Error Calculation
→ Anomaly Detection
→ Visualization and Output Generation

---

## Project Structure

```
traffic-anomaly-detection
│
├── data
│
├── models
│   └── anomaly_model.pth
│
├── results
│   ├── anomalies
│   │   └── detected anomaly frames
│   │
│   ├── graphs
│   │   └── anomaly score plots
│   │
│   └── videos
│       └── annotated detection videos
│
├── src
│   ├── model.py
│   ├── train.py
│   ├── detect_anomaly.py
│   ├── extract_frames.py
│   └── load_video.py
│
├── requirements.txt
└── README.md
```

---

## Dataset

The model is trained and evaluated using the ShanghaiTech Campus dataset, a widely used benchmark for video anomaly detection in surveillance environments. The dataset contains multiple surveillance scenes recorded under different conditions and includes both normal and anomalous activities.

---

## Installation

Clone the repository:

```
git clone https://github.com/yourusername/traffic-anomaly-detection.git
```

Navigate to the project directory:

```
cd traffic-anomaly-detection
```

Install the required dependencies:

```
pip install -r requirements.txt
```

---

## Usage

### Step 1: Extract Frames from Videos

```
python src/extract_frames.py
```

This step converts video files into individual frames for training and testing.

---

### Step 2: Train the Autoencoder

```
python src/train.py
```

The model is trained using frames representing normal traffic conditions.

---

### Step 3: Run Anomaly Detection

```
python src/detect_anomaly.py
```

This step processes the test frames, computes reconstruction errors, and identifies anomalous frames.

---

## Output

After running the detection script, the system generates the following outputs:

Detected anomaly frames stored in:

```
results/anomalies
```

Anomaly score plots showing reconstruction error trends stored in:

```
results/graphs
```

Annotated output videos with detected anomalies stored in:

```
results/videos
```

These outputs provide both quantitative and visual insights into the detection results.

---

## Technologies Used

Python
PyTorch
OpenCV
NumPy
Matplotlib

---

## Potential Extensions

Several improvements can be explored to enhance the system:

* Incorporating object detection models to focus on specific entities such as vehicles or pedestrians
* Implementing real-time anomaly detection from live video streams
* Deploying the model as a web service for smart surveillance applications
* Experimenting with advanced architectures such as convolutional LSTMs or graph neural networks for spatio-temporal modeling

---

## Author

This project was developed as part of graduate-level work in data science, focusing on the application of deep learning techniques for computer vision and anomaly detection in real-world environments.
