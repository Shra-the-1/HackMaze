🚀 HackMaze 2025 – Crack The Crack
This repository contains the code and implementation details for HackMaze 2025, a hackathon conducted by IIIT Dharwad in March 2025. The challenge was to develop a real-time crack detection system for bottles moving on a conveyor belt using a Raspberry Pi 3.

🏆 Challenge Overview
Objective: Detect cracks on bottles moving through a conveyor belt using a machine learning model.
Platform: Raspberry Pi 3
Duration: 2 days
📸 How It Works
1. Video Input and Frame Extraction
Captured a continuous video feed of the conveyor belt using a Raspberry Pi camera.
Extracted individual frames from the video stream in real-time for processing.
2. Angle Detection with ML Model 1
Developed and trained an ML model (Model 1) to predict the angle at which the camera is positioned relative to the conveyor belt.
Model predicted one of three possible angles.
3. Frame Cropping and Preprocessing
Based on the predicted angle, cropped the frame to focus on the relevant bottle region.
Cropped the frame into a square shape since Model 2 was trained on square inputs only.
4. Crack Detection with ML Model 2
Processed the cropped frame through a second ML model (Model 2) trained to detect cracks.
Predicted if the bottle is:
✅ Perfect
❌ Cracked
🏗️ Technical Stack
Hardware: Raspberry Pi 3, Raspberry Pi Camera
Language: Python
ML Framework: TensorFlow Lite
Image Processing: OpenCV, PIL
Deployment: Edge-based processing on Raspberry Pi
🌟 Challenges and Solutions
✅ Limited compute power → Optimized model size using TensorFlow Lite.
✅ Limited training time → Focused on high-quality datasets and faster training.
✅ Cropping complexity → Predefined cropping parameters for each angle.

🤝 Team
Shrawan Tibarewal (ME)
Aditya Vikram Singh
Sourav Jha (Lead)
Kushagara Kumar Arora
🏅 Outcome
Successfully implemented a real-time crack detection system using Raspberry Pi within 2 days! 🎉
