# AI-Powered Termite Tracking System

## Overview
A Pi Camera Module (ribbon cable into the CSI port)?


The AI-Powered Termite Tracking System is designed to automate the process of monitoring termite behavior in a petri dish using real-time computer vision and machine learning. The system utilizes a high-resolution camera connected to a Raspberry Pi, which processes video input and sends the video to a larger computer which performs inference using a pre-trained deep learning model (e.g., SLEAP) to track individual termites and their components in near real time. The processed data is then transmitted and configured for further analysis and visualization.

This project aims to improve the efficiency and accuracy of termite behavioral studies compared to traditional manual observation by providing automated tracking and significantly quicker workflow.

## Project Goals

- Capture high-resolution video frames of termite movement.
- Preprocess the video for analysis (resize, normalize, etc.).
- Run inference with a deep-learning model on connected computer (SLEAP or similar).
- Extract and overlay keypoints (e.g., termite body parts) on each video frame.
- Display and export processed data, including tracked variables (speed, direction, interactions).
- Provide a user-friendly dashboard for monitoring termite behavior.

## External Hardware Components

- **Preprocessing Unit:** Raspberry Pi 4 Model B (8GB) (https://www.sparkfun.com/raspberry-pi-4-model-b-8-gb.html?src=raspberrypi)
- **High-Resolution Camera:** https://www.sparkfun.com/raspberry-pi-ai-camera.html?src=raspberrypi.
- **Lighting System:** Controlled lighting for better visibility and tracking accuracy. (Yet to determine)
- **Petri Dish & 3d printed case Environment:** Standardized environment to ensure consistent data collection. (Yet to design)

### Preprocessing Unit Software

- **Automated Recording:** Develop a script to start recording once the petri dish is placed and a button is pressed.
- **Preprocessing Pipeline:** Includes steps like cropping, adjusting FPS, and reducing noise.
- **Transmission:** Compression and frame transmission to larger computer to run inference on proccessed frame.
