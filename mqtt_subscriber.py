import os
import json
import time
import threading
import numpy as np
import paho.mqtt.client as mqtt

from modules.data_preprocessing import DataPreprocessor
from modules.rbf_network import RBFNetwork
from modules.som_clustering import SOMCluster
from modules.art_network import ART1Network, ART2Network
from modules.fuzzy_logic import FuzzyDecisionMaker
from modules.rl_agent import QLearningAgent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

TREND_LABELS = {0: "Stable", 1: "Rising", 2: "Falling"}
GAS_DNG_LABELS = {0: "Safe", 1: "Medium", 2: "Danger"}
STATE_LABELS = {0: "Normal", 1: "Warning", 2: "Dangerous"}

BROKER_HOST = "broker.hivemq.com"
BROKER_PORT = 1883
TOPIC_SENSOR = "esp32/smart_env/sensors"
TOPIC_COMMANDS = "esp32/smart_env/commands"
TOPIC_AI = "esp32/smart_env/ai_insights"
TOPIC_DATA = TOPIC_SENSOR

class MQTTInferencePipeline:
    def __init__(self, data_dir: str, model_dir: str, buffer_size: int = 100):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.buffer_size = buffer_size
        # Use a stable client id for the server-side subscriber/pipeline
        # so the broker can identify this receiver separately from the ESP32 publisher.
        self.client = mqtt.Client(client_id="server_pipeline_001")
        self.connected = False
        self.message_count = 0
        self._history: list[dict] = []
        self._latest: dict | None = None
        self.on_new_reading = None
        self.rbf = None
        self.som = None
        self.art1 = None
        self.art2 = None
        self.fuzzy = None
        self.rl = None
        self.dp = None

    def load_models(self):
        print("[INIT] Loading AI models...")
        self.dp = DataPreprocessor()

        data_file = os.path.join(self.data_dir, "iot_enhanced_dataset.csv")
        df_raw = self.dp.load_data(data_file)
        df_clean = self.dp.clean_data(df_raw)
        df_derived = self.dp.add_derived_targets(df_clean)
        self.dp.normalize_data(df_derived)

        try:
            self.rbf = RBFNetwork().load_model(os.path.join(self.model_dir, "rbf_model_optimized.pkl"))
        except Exception:
            self.rbf = RBFNetwork().load_model(os.path.join(self.model_dir, "rbf_model.pkl"))

        self.som = SOMCluster().load_model(os.path.join(self.model_dir, "som_model.pkl"))
        self.art1 = ART1Network().load_model(os.path.join(self.model_dir, "art1_model.pkl"))
        self.art2 = ART2Network().load_model(os.path.join(self.model_dir, "art2_model.pkl"))

        self.fuzzy = FuzzyDecisionMaker()

        self.rl = QLearningAgent().load_model(os.path.join(self.model_dir, "rl_model.pkl"))
        print("[INIT] Models loaded successfully.")

    def start(self):
        self.load_models()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

        backoff = 1
        while True:
            try:
                print(f"[MQTT] Connecting to {BROKER_HOST}:{BROKER_PORT}...")
                self.client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
                # start network loop in background
                self.client.loop_start()

                # wait a short time for on_connect
                wait_until = time.time() + 10
                while not self.connected and time.time() < wait_until:
                    time.sleep(0.1)

                if self.connected:
                    print("[MQTT] Connection established.")
                    break
                else:
                    print("[ERROR] MQTT connection timed out waiting for CONNACK.")
                    self.client.loop_stop()
            except Exception as exc:
                print(f"[ERROR] MQTT connection failed: {exc}")

            print(f"[MQTT] Retrying in {backoff} seconds...")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)

    def _on_connect(self, client, userdata, flags, rc):
        self.connected = True
        print(f"[MQTT] Connected with result code {rc}")
        client.subscribe(TOPIC_SENSOR)
        print(f"[MQTT] Subscribed to {TOPIC_SENSOR}")

    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        print(f"[MQTT] Disconnected (rc={rc}), attempting background reconnect...")

        def _reconnect_loop():
            backoff = 1
            while not self.connected:
                try:
                    client.reconnect()
                    return
                except Exception as e:
                    print(f"[ERROR] Reconnect failed: {e}")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)

        threading.Thread(target=_reconnect_loop, daemon=True).start()

    def _on_message(self, client, userdata, msg):
        payload = msg.payload.decode("utf-8")
        try:
            sensors = json.loads(payload)
            self.process_sensor_data(client, sensors)
        except json.JSONDecodeError:
            print(f"[WARN] Invalid JSON received: {payload}")
        except Exception as e:
            print(f"[ERROR] Inference error: {e}")

    def _record_reading(self, data: dict):
        self._latest = data
        self._history.append(data)
        if len(self._history) > self.buffer_size:
            self._history.pop(0)
        if self.on_new_reading:
            self.on_new_reading(data)

    def get_latest(self) -> dict | None:
        return self._latest

    def get_history(self, n: int = 50) -> list[dict]:
        return list(self._history[-n:])

    def process_sensor_data(self, client, sensors):

        # ========================================================================
        # RAW SENSOR ARRAY
        # ========================================================================

        raw_array = np.array([[
            sensors.get("temperature", 25.0),
            sensors.get("humidity", 50.0),
            sensors.get("gas_level", 150),
            sensors.get("light", 1)
        ]], dtype=np.float32)

        # ========================================================================
        # PREPROCESSING
        # ========================================================================

        scaled_array = self.dp.scaler.transform(raw_array)

        # IMPORTANT:
        # Match SOM training preprocessing
        # Increase gas feature influence
        scaled_array[0][2] *= 4.5

        print("\n========== PIPELINE DEBUG ==========")
        print(f"Raw Input    : {raw_array}")
        print(f"Scaled Input : {scaled_array}")
        print("====================================")

        # ========================================================================
        # RBF PREDICTION
        # ========================================================================

        rbf_preds = self.rbf.predict_one(raw_array)

        sys_state = rbf_preds["system_state"]

        temp_trend = rbf_preds["temp_trend"]

        gas_danger = rbf_preds["gas_danger"]

        # ========================================================================
        # SOM PREDICTION
        # ========================================================================

        env_state = self.som.get_env_state(
            scaled_array
        )

        # ========================================================================
        # ART NETWORKS
        # ========================================================================

        art1_cat = int(
            self.art1.classify(scaled_array)[0]
        )

        art2_cat = int(
            self.art2.classify(scaled_array)[0]
        )

        novel_pattern = bool(
            self.art2.detect_new_patterns(
                scaled_array
            )[0]
        )

        # ========================================================================
        # RL AGENT
        # ========================================================================

        action_idx = self.rl.choose_action(
            sys_state,
            temp_trend,
            gas_danger,
            explore=False
        )

        rl_action = self.rl.get_action_dict(
            action_idx
        )

        # ========================================================================
        # FUZZY LOGIC
        # ========================================================================

        fuzzy_decision = self.fuzzy.decide(
            sensors
        )

        fuzzy_state = self.fuzzy.decide_state(
            sensors
        )

        # ========================================================================
        # FINAL DECISION
        # ========================================================================

        if (
            fuzzy_state == "dangerous" and
            rl_action["alarm"] < 50
        ):

            print(
                "[AI] Overriding RL with "
                "Fuzzy Logic (Danger Detected)"
            )

            final_action = {

                "fan_speed": int(
                    fuzzy_decision["fan_speed"]
                ),

                "vent_angle": int(
                    fuzzy_decision["vent_angle"]
                ),

                "alarm_level": int(
                    fuzzy_decision["alarm_level"]
                ),

                "risk_level": int(
                    fuzzy_decision["risk_level"]
                ),

                "rbf_label": fuzzy_state.lower(),

                "som_state": env_state.lower(),

                "source": "fuzzy"
            }

        else:

            final_action = {

                "fan_speed": int(
                    rl_action["fan"]
                ),

                "vent_angle": int(
                    rl_action["vent"]
                ),

                "alarm_level": int(
                    rl_action["alarm"]
                ),

                "risk_level": round(
                    fuzzy_decision["risk_level"],
                    1
                ),

                "rbf_label": STATE_LABELS.get(
                    sys_state,
                    "Unknown"
                ),

                "som_state": env_state.lower(),

                "source": "rl_agent"
            }

        # ========================================================================
        # SEND COMMANDS
        # ========================================================================

        client.publish(
            TOPIC_COMMANDS,
            json.dumps(final_action)
        )

        # ========================================================================
        # SOM DETAILS
        # ========================================================================

        row, col = self.som.winner(
            scaled_array[0]
        )

        cluster_id = int(
            row * self.som.grid_size[1] + col
        )

        # ========================================================================
        # AI INSIGHTS
        # ========================================================================

        ai_insights = {

            "rbf": {

                "label": STATE_LABELS.get(
                    sys_state,
                    "Unknown"
                ).lower(),

                "state": STATE_LABELS.get(
                    sys_state,
                    "Unknown"
                ),

                "temp_trend": TREND_LABELS.get(
                    temp_trend,
                    "Unknown"
                ),

                "gas_danger": GAS_DNG_LABELS.get(
                    gas_danger,
                    "Unknown"
                )
            },

            "som": {

                "mapped_state": env_state,

                "cluster_id": cluster_id,

                "position": [
                    int(row),
                    int(col)
                ]
            },

            "art": {

                "binary_category": art1_cat,

                "continuous_category": art2_cat,

                "is_novel": novel_pattern
            },

            "fuzzy": {

                "fan_speed": float(
                    fuzzy_decision["fan_speed"]
                ),

                "alarm_level": float(
                    fuzzy_decision["alarm_level"]
                ),

                "vent_angle": float(
                    fuzzy_decision["vent_angle"]
                ),

                "risk_level": round(
                    fuzzy_decision["risk_level"],
                    1
                ),

                "assessed_state": fuzzy_state
            }
        }

        # ========================================================================
        # PUBLISH AI INSIGHTS
        # ========================================================================

        client.publish(
            TOPIC_AI,
            json.dumps(ai_insights)
        )

        # ========================================================================
        # STORE HISTORY
        # ========================================================================

        self.message_count += 1

        self._record_reading({

            "timestamp": sensors.get(
                "timestamp"
            ),

            "sensor": sensors,

            "action": final_action,

            "insights": ai_insights
        })

        # ========================================================================
        # TERMINAL OUTPUT
        # ========================================================================

        print(
            f"[DATA] "
            f"T:{raw_array[0][0]:.1f}C, "
            f"H:{raw_array[0][1]:.1f}%, "
            f"G:{raw_array[0][2]:.0f}, "
            f"L:{raw_array[0][3]}"
        )

        print(
            f"  -> Action : "
            f"Fan={final_action['fan_speed']}%, "
            f"Vent={final_action['vent_angle']}°, "
            f"Alarm={final_action['alarm_level']} "
            f"({final_action['source']})"
        )

        print(
            f"  -> AI     : "
            f"State={ai_insights['rbf']['state']}, "
            f"Novel={novel_pattern}, "
            f"Env={env_state}"
        )

if (__name__ == "__main__"):
    pipeline = MQTTInferencePipeline(data_dir=os.path.join(BASE_DIR, "data"), model_dir=MODELS_DIR)
    pipeline.start()
