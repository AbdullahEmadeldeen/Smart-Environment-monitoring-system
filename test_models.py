"""
test_models.py

FINAL FIXED MODEL TESTER
for Smart Adaptive Environment Monitoring System (SAEMS)

Features:
- Correct SOM preprocessing consistency
- Proper gas weighting during inference
- Fixed indentation issues
- Stable testing pipeline
- Clean debugging
"""

import os
from typing import Any

import numpy as np

from modules.data_preprocessing import (
    DataPreprocessor,
    FEATURE_COLS,
    LABEL_NAMES,
    TREND_COL,
    GAS_DNG_COL
)

from modules.fuzzy_logic import FuzzyDecisionMaker

from modules.art_network import (
    ART1Network,
    ART2Network
)

from modules.rbf_network import (
    RBFNetwork,
    TREND_NAMES,
    GAS_DNG_NAMES
)

from modules.som_clustering import SOMCluster

from modules.rl_agent import QLearningAgent


# ============================================================================
# PATHS
# ============================================================================

ROOT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    ROOT_DIR,
    "models"
)

DATA_CANDIDATES = [

    os.path.join(
        ROOT_DIR,
        "data",
        "iot_enhanced_dataset.csv"
    ),

]


# ============================================================================
# DATASET SEARCH
# ============================================================================

def find_dataset_file() -> str | None:

    for path in DATA_CANDIDATES:

        if os.path.exists(path):

            return path

    return None


# ============================================================================
# LOAD SCALER
# ============================================================================

def load_data_scaler():

    dataset_path = find_dataset_file()

    if not dataset_path:

        print(
            "[WARN] Dataset not found."
        )

        return None, None

    preprocessor = DataPreprocessor()

    raw_df = preprocessor.load_data(
        dataset_path
    )

    clean_df = preprocessor.clean_data(
        raw_df
    )

    derived_df = preprocessor.add_derived_targets(
        clean_df
    )

    scaled_df = preprocessor.normalize_data(
        derived_df
    )

    return preprocessor, scaled_df


# ============================================================================
# FLOAT INPUT
# ============================================================================

def prompt_float(
    name: str,
    default: float,
    min_value: float | None = None,
    max_value: float | None = None
) -> float:

    while True:

        prompt = f"{name} [{default}]: "

        value = input(prompt).strip()

        if value == "":

            return default

        try:

            result = float(value)

            if (
                min_value is not None and
                result < min_value
            ):

                print(
                    f"  Value must be >= {min_value}"
                )

                continue

            if (
                max_value is not None and
                result > max_value
            ):

                print(
                    f"  Value must be <= {max_value}"
                )

                continue

            return result

        except ValueError:

            print(
                "  Please enter a numeric value."
            )


# ============================================================================
# SENSOR INPUTS
# ============================================================================

def prompt_sensor_inputs():

    print(
        "\nEnter the raw sensor readings."
    )

    temperature = prompt_float(
        "Temperature (C)",
        default=25.0,
        min_value=0.0,
        max_value=100.0
    )

    humidity = prompt_float(
        "Humidity (%)",
        default=50.0,
        min_value=0.0,
        max_value=100.0
    )

    gas_level = prompt_float(
        "Gas level",
        default=200.0,
        min_value=0.0,
        max_value=2000.0
    )

    light = prompt_float(
        "Light (0=dark, 1=lit)",
        default=0.0,
        min_value=0.0,
        max_value=1.0
    )

    return {
        "temperature": temperature,
        "humidity": humidity,
        "gas_level": gas_level,
        "light": light,
    }


# ============================================================================
# LOAD MODEL
# ============================================================================

def load_model_if_available(
    model_class: Any,
    filename: str
):

    model_path = os.path.join(
        MODEL_DIR,
        filename
    )

    if not os.path.exists(model_path):

        print(
            f"[INFO] Model not found: {model_path}"
        )

        return None

    model = model_class()

    try:

        model.load_model(model_path)

        return model

    except Exception as exc:

        print(
            f"[WARN] Could not load "
            f"'{filename}': {exc}"
        )

        return None


# ============================================================================
# MAIN
# ============================================================================

def main():

    print(
        "=== Smart Environment Model Tester ==="
    )

    # ------------------------------------------------------------------------
    # LOAD PREPROCESSING
    # ------------------------------------------------------------------------

    preprocessor, scaled_df = load_data_scaler()

    if (
        preprocessor is None or
        scaled_df is None
    ):

        print(
            "[WARN] Scaler unavailable."
        )

    # ------------------------------------------------------------------------
    # SENSOR INPUTS
    # ------------------------------------------------------------------------

    sensors = prompt_sensor_inputs()

    raw_input = np.array([[
        sensors[FEATURE_COLS[0]],
        sensors[FEATURE_COLS[1]],
        sensors[FEATURE_COLS[2]],
        sensors[FEATURE_COLS[3]],
    ]], dtype=np.float32)

    # ------------------------------------------------------------------------
    # SCALE INPUT
    # ------------------------------------------------------------------------

    scaled_input = None

    if preprocessor is not None:

        try:

            scaled_input = (
                preprocessor.scaler
                .transform(raw_input)
            )

            # IMPORTANT:
            # Match SOM training preprocessing
            scaled_input[0][2] *= 3.0

            print("\n[DEBUG] Scaled Input:")
            print(scaled_input)

        except Exception as exc:

            print(
                f"[WARN] Could not scale inputs: {exc}"
            )

            scaled_input = None

    print("\n--- Results ---")

    # =========================================================================
    # FUZZY LOGIC
    # =========================================================================

    try:

        fuzzy = FuzzyDecisionMaker()

        decision = fuzzy.decide(sensors)

        print("\n[Fuzzy Logic]")

        for key, value in decision.items():

            print(
                f"  {key}: {value:.2f}"
            )

    except Exception as exc:

        print(
            f"[ERROR] Fuzzy logic failed: {exc}"
        )

    # =========================================================================
    # RBF NETWORK
    # =========================================================================

    rbf = load_model_if_available(
        RBFNetwork,
        "rbf_model.pkl"
    )

    if rbf is not None:

        try:

            preds = rbf.predict(raw_input)

            state = int(
                preds["system_state"][0]
            )

            trend = int(
                preds["temp_trend"][0]
            )

            gas_danger = int(
                preds["gas_danger"][0]
            )

            print("\n[RBF Network]")

            print(
                f"  System State: {state} "
                f"({LABEL_NAMES.get(state, str(state))})"
            )

            print(
                f"  Temperature Trend: {trend} "
                f"({TREND_NAMES.get(trend, str(trend))})"
            )

            print(
                f"  Gas Danger Level: {gas_danger} "
                f"({GAS_DNG_NAMES.get(gas_danger, str(gas_danger))})"
            )

        except Exception as exc:

            print(
                f"[WARN] RBF prediction failed: {exc}"
            )

    # =========================================================================
    # ART2 NETWORK
    # =========================================================================

    art = load_model_if_available(
        ART2Network,
        "art2_model.pkl"
    )

    if art is not None:

        if scaled_input is None:

            print(
                "[SKIP] ART requires scaler."
            )

        else:

            try:

                category = int(
                    art.classify(scaled_input)[0]
                )

                print("\n[ART2 Network]")

                print(
                    f"  Category index: {category}"
                )

            except Exception as exc:

                print(
                    f"[WARN] ART failed: {exc}"
                )

    # =========================================================================
    # SOM NETWORK
    # =========================================================================

    som = load_model_if_available(
        SOMCluster,
        "som_model.pkl"
    )

    if som is not None:

        if scaled_input is None:

            print(
                "[SKIP] SOM requires scaler."
            )

        else:

            try:

                cluster_id = int(
                    som.get_cluster_id(
                        scaled_input
                    )[0]
                )

                row, col = som.winner(
                    scaled_input[0]
                )

                cluster_name = som.get_env_state(
                    scaled_input
                )

                print("\n[SOM Clustering]")

                print(
                    f"  Neuron position: ({row}, {col})"
                )

                print(
                    f"  Flat cluster id: {cluster_id}"
                )

                print(
                    f"  Cluster name: {cluster_name}"
                )

            except Exception as exc:

                print(
                    f"[WARN] SOM failed: {exc}"
                )

    # =========================================================================
    # RL AGENT
    # =========================================================================

    try:

        rl_agent = QLearningAgent(
            alpha=0.1,
            gamma=0.9,
            epsilon=0.1
        )

        action_idx = rl_agent.choose_action(
            1,
            1,
            1,
            explore=False
        )

        action = rl_agent.get_action_dict(
            action_idx
        )

        print("\n[RL Agent (Q-Learning)]")

        print(
            "  System State: warning | "
            "Temp Trend: rising | "
            "Gas Danger: medium"
        )

        print(
            f"  Recommended Action: {action}"
        )

    except Exception as exc:

        print(
            f"[WARN] RL agent failed: {exc}"
        )

    print("\n=== Test Complete ===")


# ============================================================================
# ENTRY
# ============================================================================

if __name__ == "__main__":

    main()