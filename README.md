# Getting Started with CUSPv1.0

CUSPv1.0 is a computer vision system designed for real-time vehicle detection, tracking, and counting. This guide provides instructions to get the project running locally using the enhanced Flask application for debugging purposes.

---

## Prerequisites

Before you begin, make sure you have the following installed on your system:

- Python 3.8 or higher  
- Git for cloning the repository  
- CUDA-compatible GPU (Recommended for optimal YOLO inference performance)  
- A Webcam or an RTSP camera stream to use as a video source  

---

## Installation

Follow these steps to set up the project environment.

### 1. Clone the Repository
First, open your terminal, navigate to your desired directory, and clone the project repository from GitHub.

```bash
git clone https://github.com/Himank-Khatri/CUSPv1.0.git
cd CUSPv1.0
````

### 2. Install Dependencies

The project uses several Python libraries, including the SORT tracking algorithm. Install all required dependencies using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 3. Configure the Video Source

You need to tell the system which video feed to use. Edit the configuration file and update the `video_path` setting to point to your:

* Video file
* Webcam index (e.g., `0`)
* RTSP stream URL

---

## Running the Application

This project includes several web application options.
As requested, this guide focuses on the **enhanced Flask app**, which is ideal for debugging as it provides additional endpoints and improved logging.

To start the server, run the following command in your terminal:

```bash
python newapp.py
```

The application will start on your local machine, typically on **port 5000**.

---

## Accessing the System

Once the application is running, open your web browser and navigate to:

```
http://localhost:5000
```

You should see the web interface displaying:

* Live video feed with bounding boxes drawn around detected vehicles
* Running vehicle count

---

## Key Features

* **Real-time Vehicle Detection**: Utilizes YOLO models to identify vehicles in the video feed.
* **Multi-Object Tracking**: Implements the SORT (Simple Online and Realtime Tracking) algorithm to track multiple vehicles simultaneously.
* **Vehicle Counting**: Tracks vehicles entering and exiting defined zones to maintain an accurate count.
* **Web-Based Monitoring**: Provides a simple and accessible web interface for viewing the video feed and data.
* **RESTful API Endpoints**: Offers API endpoints for programmatic integration and data retrieval.

---

## Notes

* The system integrates the **SORT tracking algorithm**, which is licensed under **GPL v3.0**.
  This means any derivative work must also comply with the GPL licensing terms.
* While effective for real-time tracking, the SORT algorithm has known limitations with:

  * Handling long-term occlusions
  * Re-identifying objects that leave and re-enter the frame

---

## Wiki Pages

For more details, explore:

* [Web Applications (Himank-Khatri/CUSPv1.0)](https://github.com/Himank-Khatri/CUSPv1.0/wiki)
