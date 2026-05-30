# 🌿 Smart Adaptive Environment Monitoring System (SAEMS)

<p align="center">
  <b>An end-to-end IoT + AI system that monitors environmental conditions in real time<br>
  and autonomously controls actuators using a multi-model intelligence pipeline.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Board-ESP32--S3-blue?logo=espressif" alt="ESP32-S3"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Protocol-MQTT-660066?logo=mqtt" alt="MQTT"/>
  <img src="https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Firmware-Arduino IDE-00979D?logo=arduino&logoColor=white" alt="Arduino"/>
</p>

---

## 📖 Overview

This project implements a **closed-loop smart environment monitoring and control system** that fuses real ESP32 hardware with a laptop-side multi-stage AI inference pipeline. Sensor data flows from the microcontroller over MQTT, passes through six AI models running on the host machine, and actionable commands are sent back to drive physical actuators — all in real time.

### Key Components

| Layer | Description |
|-------|-------------|
| **Hardware** | ESP32-S3 with DHT22, MQ-135, LDR, SG90 Servo, Passive Buzzer, Relay/Fan, RGB LED |
| **Communication** | MQTT via `broker.hivemq.com:1883` (public broker) |
| **AI Pipeline** | PCA → SOM → RBF Network → Fuzzy Logic → Q-Learning RL → ART1/ART2 |
| **Optimization** | Genetic Algorithm evolves RBF weights and fuzzy rule parameters |
| **Dashboards** | Real-time WebSocket dashboard (Flask + Socket.IO) & Streamlit analytics dashboard |
| **Data Generation** | Browser-based enhanced dataset generator with anti-overfitting features |

---

## 📁 Repository Structure

```
.
│
├── main.py                         # Training pipeline — trains all AI models sequentially
├── server_iot.py                   # Flask + Socket.IO server — serves webapp & starts MQTT inference
├── mqtt_subscriber.py              # MQTT inference pipeline — loads models, processes live sensor data
├── dashboard.py                    # Streamlit analytics dashboard (run with: streamlit run dashboard.py)
├── test_models.py                  # Interactive CLI tester — manually test all models with custom inputs
├── iot_enhanced_generator.html     # Browser-based dataset generator with anti-overfitting features
│
├── modules/                        # Python AI & preprocessing modules
│   ├── __init__.py
│   ├── data_preprocessing.py       # DataPreprocessor — cleaning, feature engineering, normalization
│   ├── pca_analysis.py             # PCAAnalyzer — dimensionality reduction (visualization only)
│   ├── rbf_network.py              # RBFNetwork — multi-output classifier (state, trend, gas danger)
│   ├── som_clustering.py           # SOMCluster — self-organizing map for environment topology mapping
│   ├── art_network.py              # ART1Network & ART2Network — adaptive resonance / novelty detection
│   ├── fuzzy_logic.py              # FuzzyDecisionMaker — rule-based crisp actuator output
│   ├── genetic_optimizer.py        # GeneticOptimizer — evolutionary tuning of RBF weights
│   └── rl_agent.py                 # QLearningAgent — reinforcement learning actuator policy
│
├── esp/                            # ESP32 firmware (Arduino IDE)
│   └── sketch_may17a.ino           # Main firmware — sensor reading, MQTT pub/sub, actuator control
│
├── main.cpp                        # Standalone C++ firmware variant (same logic as sketch_may17a.ino)
│
├── firmware/                       # Wokwi simulation configuration
│   ├── diagram.json                # Wokwi circuit wiring diagram
│   └── libraries.txt               # Required Arduino libraries for Wokwi
│
├── webapp/                         # Real-time web dashboard (served by server_iot.py)
│   └── index.html                  # Single-page dashboard — Socket.IO + Chart.js live UI
│
├── data/                           # Datasets & processed outputs
│   ├── iot_sensor_dataset.csv      # Original sensor dataset
│   ├── iot_enhanced_dataset.csv    # Enhanced dataset (generated with anti-overfitting features)
│   ├── processed_data.csv          # Normalized & feature-engineered output from training
│   └── pca_features.csv            # PCA-reduced feature matrix
│
├── models/                         # Trained model artifacts (Python pickle .pkl)
│   ├── rbf_model.pkl               # Base RBF network
│   ├── rbf_model_optimized.pkl     # GA-optimized RBF network
│   ├── som_model.pkl               # Trained SOM grid (20×20)
│   ├── art1_model.pkl              # ART1 binary pattern categories
│   ├── art2_model.pkl              # ART2 continuous pattern categories
│   └── rl_model.pkl                # Q-learning agent Q-table
│
├── results/                        # Training output plots
│   ├── pca_plot.png                # PCA 2D projection scatter plot
│   ├── som_clusters.png            # SOM cluster visualization
│   ├── ga_convergence.png          # Genetic algorithm fitness over generations
│   ├── correlation_matrix.png      # Feature correlation heatmap
│   ├── boxplots.png                # Feature distribution boxplots
│   ├── pairplot.png                # Feature pairplot by label
│   ├── label_distribution.png      # Class balance bar chart
│   ├── temporal_trends.png         # Time-series feature trends
│   ├── event_distribution.png      # Event type distribution
│   └── anomaly_detection.png       # Anomaly detection results
│
├── txt/                            # Documentation & reference
│   ├── saems_workflow.md           # Detailed A-to-Z technical workflow document
│   └── components&wiring.txt       # Hardware wiring diagram (pin-by-pin)
│
└── README.md                       # This file
```

---

## 🔄 System Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FULL SYSTEM PIPELINE                                │
│                                                                             │
│  ┌──────────────┐       MQTT        ┌──────────────────────────────────┐   │
│  │  ESP32-S3    │  ──────────────►  │  Laptop (server_iot.py)          │   │
│  │              │  sensors topic    │                                  │   │
│  │  DHT22       │                   │  mqtt_subscriber.py              │   │
│  │  MQ-135      │                   │    ├─ Preprocess (StandardScaler)│   │
│  │  LDR         │                   │    ├─ RBF Network → 3 predictions│   │
│  │              │                   │    ├─ SOM → environment mapping  │   │
│  │  Servo       │                   │    ├─ ART1/ART2 → novelty check │   │
│  │  Buzzer      │                   │    ├─ Fuzzy Logic → crisp output │   │
│  │  Relay/Fan   │  ◄──────────────  │    └─ RL Agent → optimal action  │   │
│  │  RGB LED     │  commands topic   │                                  │   │
│  └──────────────┘                   │  webapp/ → real-time dashboard   │   │
│                                     └──────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Pipeline

| # | Stage | File(s) | Description |
|---|-------|---------|-------------|
| 1 | **Sensor Capture** | `esp/sketch_may17a.ino` | ESP32 reads DHT22, MQ-135, LDR every 5 seconds. Publishes JSON to `esp32/smart_env/sensors` via MQTT. |
| 2 | **MQTT Ingestion** | `mqtt_subscriber.py` | `MQTTInferencePipeline` subscribes to the sensor topic, receives JSON payloads, buffers recent readings. |
| 3 | **Preprocessing** | `modules/data_preprocessing.py` | `StandardScaler` normalizes raw values. Gas feature weighted ×4.5 for SOM separation. |
| 4 | **RBF Prediction** | `modules/rbf_network.py` | Multi-output RBF predicts **system state** (normal/warning/dangerous), **temperature trend** (stable/rising/falling), and **gas danger level** (safe/medium/danger). |
| 5 | **SOM Mapping** | `modules/som_clustering.py` | Maps the current reading onto a 20×20 neuron grid. Returns the winning neuron position and its labeled environment state. |
| 6 | **ART Novelty** | `modules/art_network.py` | ART1 (binary) and ART2 (continuous) classify the pattern. If no existing category matches the vigilance threshold, flags `is_novel: True`. |
| 7 | **Fuzzy Logic** | `modules/fuzzy_logic.py` | Triangular/trapezoidal membership functions produce crisp actuator outputs: fan speed (%), vent angle (°), alarm level, and risk score. |
| 8 | **RL Action** | `modules/rl_agent.py` | Q-learning agent queries its Q-table using (state, trend, gas_danger) to select the most energy-efficient action. |
| 9 | **Conflict Resolution** | `mqtt_subscriber.py` | **Safety override**: if Fuzzy Logic detects "dangerous" AND the RL agent's alarm is low → Fuzzy takes priority (max fan, 180° vent, full alarm). Otherwise RL action is used. |
| 10 | **Actuator Command** | `mqtt_subscriber.py` | Final decision JSON published to `esp32/smart_env/commands`. ESP32 immediately executes `applyActuators()`. |
| 11 | **Dashboard** | `server_iot.py` + `webapp/index.html` | Flask + Socket.IO pushes every inference result to the browser via WebSocket. Live charts, AI model cards, actuator status panel. |

---

## 🧠 AI Models & Training

The training pipeline (`main.py`) runs all models sequentially in 9 steps:

| Model | Module | Key Parameters | Output |
|-------|--------|----------------|--------|
| **PCA** | `pca_analysis.py` | 4 components | 2D scatter plot for visualization only (not used in inference) |
| **RBF Network** | `rbf_network.py` | 40 K-Means centers, Gaussian kernel | 3-head classifier: system state, temp trend, gas danger |
| **SOM** | `som_clustering.py` | 20×20 grid, 10,000 iterations | Topology-preserving environment map with labeled neurons |
| **ART1** | `art_network.py` | Vigilance = 0.7 | Binary pattern category formation |
| **ART2** | `art_network.py` | Vigilance = 0.85 | Continuous pattern categories + novelty detection |
| **Genetic Algorithm** | `genetic_optimizer.py` | Pop = 20, Generations = 20 | Optimizes all 3 RBF weight heads (state, trend, gas) |
| **Fuzzy Logic** | `fuzzy_logic.py` | Calibrated membership functions | Rule-based crisp outputs (fan, vent, alarm, risk) |
| **Q-Learning** | `rl_agent.py` | α=0.1, γ=0.9, ε=0.1, 10K pretrain iterations | Discrete action policy from Q-table |

All trained models are saved as **Python pickle files** (`.pkl`) in the `models/` directory.

---

## ⚡ Hardware

### ESP32-S3 Pin Map

| Component | GPIO | Type | Notes |
|-----------|------|------|-------|
| DHT22 (Temp + Humidity) | GPIO 12 | Digital Input | 10kΩ pull-up to 3.3V |
| MQ-135 (Gas / Air Quality) | GPIO 4 | Analog Input (ADC) | Voltage divider 10kΩ+10kΩ for 3.3V safety |
| LDR Module | GPIO 8 | Digital Input (DO) | HIGH = Bright, LOW = Dark |
| Passive Buzzer (HYLD1205A) | GPIO 5 | PWM Output (LEDC) | Non-blocking state machine |
| Relay Module (Fan) | GPIO 7 | Digital Output | Active HIGH — HIGH = Fan ON |
| SG90 Servo (Vent) | GPIO 9 | PWM Output | 0°–180° range (500–2400 µs) |
| RGB LED — Red | GPIO 20 | Digital Output | 330Ω resistor, common cathode |
| RGB LED — Green | GPIO 18 | Digital Output | 330Ω resistor |
| RGB LED — Blue | GPIO 17 | Digital Output | 330Ω resistor |
| Onboard Status LED | GPIO 2 | Digital Output | Blinks on successful MQTT publish |

### Actuator Behavior by State

| State | RGB LED | Fan | Buzzer | Servo |
|-------|---------|-----|--------|-------|
| 🟢 **Normal** | Green (solid) | OFF | Silent | 0° (closed) |
| 🔵 **Warning** | Blue (solid) | ON (continuous) | Blinking beep (1 kHz) | 90° (half open) |
| 🔴 **Dangerous** | Red (solid) | ON (full power) | Continuous alarm (2.5 kHz) | 180° (fully open) |

---

## 🛠️ Setup & Installation

### Prerequisites

- **Python 3.10+**
- **MQTT broker** — the system uses the public `broker.hivemq.com:1883` by default
- **Arduino IDE** with ESP32-S3 board support (for flashing real hardware)
- Arduino libraries: `ESP32Servo`, `DHT sensor library`, `ArduinoJson (v7+)`, `PubSubClient`

### 1. Clone & Install Python Dependencies

```bash
# Clone the repository
git clone <repo-url>
cd "Smart adaptive environment monitoring system last"

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn paho-mqtt flask flask-socketio streamlit scikit-fuzzy
```

### 2. Train All AI Models

```bash
python main.py
```

This runs the full 9-step training pipeline and saves all models to `models/`.

### 3. Start the Live Inference Server

```bash
python server_iot.py
```

This starts:
- The **MQTT inference pipeline** (subscribes to `esp32/smart_env/sensors`)
- A **Flask + Socket.IO web server** at `http://localhost:5000`
- The **real-time dashboard** served from `webapp/index.html`

### 4. (Optional) Start the Streamlit Analytics Dashboard

```bash
streamlit run dashboard.py
```

Opens at `http://localhost:8501` — provides dataset exploration, fuzzy logic playground, and training result visualizations.

### 5. Flash the ESP32

1. Open `esp/sketch_may17a.ino` in Arduino IDE.
2. Update `WIFI_SSID` and `WIFI_PASSWORD` in the sketch.
3. Select board: **ESP32-S3 Dev Module**.
4. Upload to the ESP32 via USB.
5. Open Serial Monitor (115200 baud) to verify sensor readings and MQTT connection.

---

## 🖥️ Usage

### Real-Time Dashboard (`server_iot.py`)
Open `http://localhost:5000` after starting the server. Shows:
- Live sensor readings (temperature, humidity, gas, light)
- AI model predictions (RBF, SOM, ART, Fuzzy Logic)
- Actuator status (RGB LED, servo angle, fan, buzzer)
- Sensor history chart (last 50 readings)
- Risk level assessment banner

### REST API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/latest` | Latest sensor reading + AI predictions |
| `GET /api/history?n=50` | Last N readings with full inference results |
| `GET /api/status` | MQTT connection status and message count |
| `GET /api/predictions/som` | Latest SOM cluster prediction |
| `GET /api/predictions/rbf` | Latest RBF state/trend/gas predictions |
| `GET /api/predictions/all` | All AI model predictions combined |

### WebSocket Events

| Event | Direction | Data |
|-------|-----------|------|
| `new_reading` | Server → Client | Full sensor + action + predictions payload |
| `som_prediction` | Server → Client | SOM cluster and state predictions |
| `rbf_prediction` | Server → Client | System state, temp trend, gas danger |
| `all_predictions` | Server → Client | All AI model predictions together |
| `request_predictions` | Client → Server | Request specific prediction type (`som`, `rbf`, `all`) |

### MQTT Topics

| Topic | Direction | Payload |
|-------|-----------|---------|
| `esp32/smart_env/sensors` | ESP32 → Laptop | `{"temperature", "humidity", "gas_level", "light", "gas_alarm", "timestamp"}` |
| `esp32/smart_env/commands` | Laptop → ESP32 | `{"fan_speed", "vent_angle", "alarm_level", "risk_level", "rbf_label", "som_state"}` |
| `esp32/smart_env/ai_insights` | Laptop → (any) | Full AI insights JSON (RBF, SOM, ART, Fuzzy) |

### Testing Models Manually

```bash
python test_models.py
```

Interactive CLI tool — enter custom sensor values and see predictions from every model.

### Dataset Generation

Open `iot_enhanced_generator.html` in a browser to generate custom training datasets with configurable:
- Sensor ranges and safety thresholds
- Label distribution (normal/warning/dangerous %)
- Anti-overfitting features (temporal patterns, noise, correlations, drift, outliers)
- Environment presets (Home, Office, Warehouse, Server Room, Greenhouse)

---

## 📊 Training Results

Sample results generated by the training pipeline in `results/`:

| Plot | Description |
|------|-------------|
| `pca_plot.png` | 2D PCA projection colored by label — shows cluster separation |
| `som_clusters.png` | SOM neuron grid with mapped environment states |
| `ga_convergence.png` | Genetic algorithm fitness improvement over generations |
| `correlation_matrix.png` | Feature-to-feature Pearson correlation heatmap |
| `label_distribution.png` | Normal vs. Warning vs. Dangerous class balance |
| `temporal_trends.png` | Time-series view of all sensor features |
| `boxplots.png` | Per-feature distribution boxplots by class |
| `pairplot.png` | Pairwise feature scatter plots |

---

## 📂 MQTT Data Flow Diagram

```
ESP32-S3                           Laptop (Python)                    Browser
┌──────────┐    MQTT Publish       ┌──────────────────┐              ┌────────────┐
│ Sensors   │──────────────────►   │ mqtt_subscriber   │              │ webapp/    │
│ DHT22     │  sensors topic       │   .py             │   Socket.IO  │ index.html │
│ MQ-135    │                      │                   │─────────────►│            │
│ LDR       │                      │  ┌─ Preprocess    │              │ Chart.js   │
│           │                      │  ├─ RBF predict   │              │ Live UI    │
│ Actuators │    MQTT Subscribe    │  ├─ SOM map       │              └────────────┘
│ Servo     │◄──────────────────   │  ├─ ART novelty   │
│ Buzzer    │  commands topic      │  ├─ Fuzzy logic   │
│ Fan/Relay │                      │  ├─ RL agent      │
│ RGB LED   │                      │  └─ Conflict res. │
└──────────┘                       └──────────────────┘
                                          │
                                   server_iot.py
                                   (Flask + SocketIO)
                                   Port 5000
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Open a Pull Request with a clear description

---

## 📄 License

This project is for academic and educational purposes.
