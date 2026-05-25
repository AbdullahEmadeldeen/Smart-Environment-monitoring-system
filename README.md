# Smart Adaptive Environment Monitoring System

## Overview

This repository contains a complete end‑to‑end smart environmental monitoring solution that integrates **IoT hardware**, **real‑time data streaming**, a **multi‑stage AI pipeline**, and a **simulation environment** for rapid development and testing.

- **Hardware layer** – Simulated (Wokwi) and real ESP‑32‑S3 boards with sensors (DHT22, MQ‑135, LDR, Ultrasonic, Servo, Buzzer, Relay/Fan, RGB‑LED).
- **Communication** – MQTT broker (`mqtt.eclipse.org` by default) connects the ESP‑32 firmware to the laptop‑side inference server.
- **Data pipeline** – Raw sensor readings → preprocessing → feature engineering → AI inference → actuator decisions.
- **AI pipeline** – PCA → Self‑Organising Map (SOM) → Radial Basis Function (RBF) network → Fuzzy Logic → Reinforcement Learning (Q‑learning) → Adaptive Resonance Theory (ART‑1 & ART‑2).
- **Optimization** – Genetic Algorithm optimises RBF weights/centres and fuzzy rule parameters.
- **Visualization** – Web‑based dashboard (`webapp/`) shows live sensor data, model predictions, and actuator states.

The project is modularised for easy extension, replacement of models, or migration to other edge devices.

## Repository Structure

```
.
├─ __pycache__/                # Python byte‑code cache
├─ dashboard.py                # Flask/Dash dashboard server
├─ data/                       # Sample datasets & logs
├─ esp/                        # ESP‑32 firmware source (C++)
├─ firmware/                   # Compiled binaries for hardware
├─ iot_enhanced_generator.html# Simulation UI (Wokwi)   
├─ main.cpp                    # C++ entry point for ESP‑32
├─ main.py                     # Central orchestration script
├─ models/                     # Trained AI models (ONNX/TFLite)
├─ modules/                    # Re‑usable Python modules (pre‑processing, AI, GA, etc.)
├─ mqtt_subscriber.py          # MQTT client for data ingestion
├─ results/                    # Output artefacts (plots, logs)
├─ server_iot.py               # IoT server handling MQTT & inference
├─ test_models.py              # Unit tests for AI models
├─ txt/                        # Documentation & notes
├─ webapp/                     # Front‑end assets (HTML/CSS/JS)
└─ README.md                   # 📖 This file
```

## Data Flow Pipeline

1. **Sensor Capture (ESP‑32)** – Sensors sample at 1 Hz and publish JSON payloads via MQTT.
2. **MQTT Ingestion (`mqtt_subscriber.py`)** – Subscribes to the topic, buffers recent frames.
3. **Pre‑processing (`modules/preprocess.py`)** – Normalises, filters outliers, and extracts statistical features.
4. **Dimensionality Reduction (`modules/pca.py`)** – Reduces feature space while preserving variance.
5. **Clustering (`modules/som.py`)** – Organises patterns into interpretable clusters.
6. **Prediction (`modules/rbf.py`)** – Multi‑output RBF predicts:
   - Overall system state (normal / warning / dangerous)
   - Temperature trend (rising / stable / falling)
   - Gas danger level (low / medium / high)
7. **Fuzzy Logic (`modules/fuzzy.py`)** – Applies calibrated fuzzy rules to produce crisp actuator commands.
8. **Reinforcement Learning (`modules/rl_qlearning.py`)** – Refines actions over time based on reward feedback.
9. **ART Networks (`modules/art1.py`, `modules/art2.py`)** – Continual learning for new sensor patterns.
10. **Genetic Optimisation (`modules/genetic_algo.py`)** – Periodic optimisation of RBF centres/weights and fuzzy rule thresholds.
11. **Actuator Control (`esp/main.cpp`)** – Commands hardware (servo, buzzer, fan, LED) based on the final decision.
12. **Visualization (`dashboard.py` & `webapp/`)** – Live charts, model confidence, and hardware state.

## AI Models & Training

- **PCA** – Trained on the `iot_sensor_dataset.csv` (features: temperature, humidity, CO₂, light, distance).
- **SOM** – 10 × 10 grid, labels manually assigned for interpretability.
- **RBF** – Multivariate output, Gaussian kernel, centres initialised from K‑means.
- **Fuzzy Logic** – Rule base derived from domain expert thresholds, fine‑tuned by GA.
- **Q‑Learning** – Discrete action space (ON/OFF fan, open/close vent, LED colour) with reward = negative environmental risk.
- **ART‑1 / ART‑2** – Online clustering for novel sensor patterns.

All models are stored in `models/` in ONNX/TFLite format for cross‑platform inference.

## Hardware Simulation

The repository includes a **Wokwi** simulation (`iot_enhanced_generator.html`) that mimics the ESP‑32‑S3 board and attached sensors. To run the simulation:

1. Open the HTML file in a modern browser.
2. Click **Start** – the virtual sensors begin publishing MQTT messages.
3. The Python server (`server_iot.py`) receives the data exactly as the real hardware would.
4. Adjust sensor values via the UI sliders to test edge‑case scenarios (e.g., high CO₂, rapid temperature rise).

This simulated environment enables rapid iteration without flashing firmware.

## Setup & Installation

### Prerequisites
- Python 3.10+ (`pip install -r requirements.txt`)
- MQTT broker (public broker works for testing, or run Mosquitto locally)
- ESP‑32‑S3 toolchain (for flashing real hardware) – see `esp/README.md`

### Installation Steps
```bash
# Clone the repository
git clone <repo‑url>
cd "Smart adaptive environment monitoring system last"

# Python environment
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start the MQTT subscriber and inference server
python server_iot.py

# (Optional) Run the dashboard
python dashboard.py
```

### Flashing Real Hardware
1. Connect ESP‑32‑S3 via USB.
2. Build the firmware:
   ```bash
   cd esp
   idf.py build
   ```
3. Flash:
   ```bash
   idf.py -p COM3 flash
   ```
4. Verify the device publishes to the MQTT topic.

## Usage

- **Live monitoring** – Open `http://localhost:8050` to view the dashboard.
- **Control** – The dashboard lets you manually override actuator states for testing.
- **Training** – Run `python modules/train_all.py` to retrain models on a new dataset.
- **Optimization** – Execute `python modules/genetic_algo.py` to fine‑tune the RBF and fuzzy parameters.

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/awesome‑feature`).
3. Ensure code passes existing tests (`pytest`).
4. Open a Pull Request with a clear description and screenshots (if UI changes).

## License

Distributed under the **MIT License**. See `LICENSE` for details.
