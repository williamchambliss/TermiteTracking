# 🧰 Physical Build: Hardware, Connections, and Enclosure

This document details the physical assembly and environmental setup of the AI-Powered Termite Tracking System, including camera wiring, lighting, 3D-printed parts, and environmental considerations used for successful tracking.

---

## 🔌 Hardware Connections

### Raspberry Pi
- **Model:** Raspberry Pi 4 Model B (8GB)
- **Camera Interface:** CSI (Camera Serial Interface)
- **Power Supply:** Official Raspberry Pi 5V 3A USB-C

### Camera Module
- **Model:** Raspberry Pi HQ Camera or AI Camera
- **Connection:** Connected to CSI port on Pi with ribbon cable (firmly locked in with blue tab facing Ethernet port)
- **Mounting:** Positioned vertically over Petri dish using custom mount

### Networking
- **Transmission Protocol:** MJPEG over TCP
- **Requirement:** Ensure Raspberry Pi and host PC are on the same subnet (e.g., `192.168.4.X`)

---

## 💡 Lighting System

Termites are small, and pose estimation requires high-contrast imagery. Here’s the lighting setup used:

- **Lighting Type:** Soft white LED ring light (USB-powered)
- **Positioning:** Evenly placed around the camera lens or angled from side at 45°
- **Mounting:** Attached to the 3D-printed case with zip ties or recessed slots
- **Power Source:** External USB power bank or Pi-powered GPIO header (careful with draw)

> ✅ *Goal:* Eliminate shadows and reflections while increasing contrast between termites and background.

---

## 🧪 Petri Dish Environment

- **Dish Type:** Standard 90mm clear plastic Petri dish
- **Background Material:** Matte white paper insert to reduce reflection and improve contrast
- **Mounting:** Centered and slightly elevated on platform inside case

---

## 🧱 3D Printed Case

> 🧩 STL File: [`hardware/termite_case_v1.stl`](hardware/termite_case_v1.stl)

### Case Components

- **Base Plate:** Holds the Raspberry Pi and Petri dish platform
- **Camera Mount:** Vertical arm that holds the Pi camera in a fixed position
- **Light Ring Mount:** Optional attachment ring or pegs for LED mounting
- **Ventilation:** Passive airflow slits on sides to reduce heat

### Printing Instructions

- **Material:** PLA or PETG
- **Layer Height:** 0.2mm
- **Infill:** 20–40% (higher around Pi mount area)
- **Supports:** Yes (especially for arm or overhangs)

---

## 🛠️ Assembly Instructions

1. Mount Raspberry Pi to base using M2 screws or zip ties
2. Attach camera via CSI and secure ribbon
3. Position Petri dish platform and secure with glue or mounting putty
4. Insert LED ring and secure to light mount
5. Place camera at fixed focal distance (~10–15 cm above Petri dish)
6. Power via USB-C, connect Pi to network, and run streamer script

---

## 📸 Final Setup Tips

- Perform a **focus test**: Manually focus the Pi HQ camera for sharpest termite edges.
- Run `libcamera-hello` to test lighting, shadows, and visibility.
- Adjust LED brightness using a dimmer if termites scatter under strong light.
- Ensure **consistent environmental conditions** for reproducibility across recordings.

---

## 📎 To-Do / Future Plans

- [ ] Design v2 of the case with integrated light diffuser
- [ ] Add ventilation fan or heat sink if Pi runs too hot
- [ ] Modular mount for different Petri dish sizes
- [ ] Integrated toggle switch for power control

---

## 📷 Photos (Optional)

> Add real-world setup photos here to improve documentation and make the repo stand out to reviewers or hiring managers.

---

## 🧵 STL + CAD Files

| File                             | Description                      |
|----------------------------------|----------------------------------|
| `termite_case_v1.stl`            | Main enclosure for Pi + dish     |
| `camera_mount_arm.stl`           | Vertical camera support          |
| `light_ring_bracket.stl`         | Optional LED mounting adapter    |

---

