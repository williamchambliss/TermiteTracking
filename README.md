# AI-Powered Termite Tracking System

## Overview


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
Step	Next Steps
Research hardware (Camera and pi) 	DONE
Develop a plan on connection between pi and camera and computer	DONE
Get the right hardware	DONE
Connect the camera to the raspberry pi	DONE
implement the transmit data from camera to pi, do any sort of proccessing and then to the computer	DONE
Power/Wake button	1. Wire a momentary pushbutton between Pi RUN (pin 5/GPIO3) and GND (pin 6)
“Start streamer” button	1. Wire a second button to GPIO17 (pin 11) → GND (pin 9) and Run Script
PC GUI with video preview, Run button, and path fields and inference preview	
Test with personal laptop entire process	
develop correct case	
find correct LED lights for case	
3D print case	
connect everything to case	
Record a good amount of termites with this set up, train models based off of these 	
I want to have some sort of button or switch, so when I flip it, Pi will turn on and turn on that file, I also want to have a gui, so when I start the file on my computer from the cmd line, it will pop up a gui that shows a preview, a run button, and real time inference if that is possible, and a textbox that I tell where to store the inference file, where to store the video file, and where the model to run the inference is.	
