import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Import our new/updated modules
from modules.data_preprocessing import DataPreprocessor, LABEL_COL, TREND_COL, GAS_DNG_COL, FEATURE_COLS
from modules.rbf_network import RBFNetwork
from modules.som_clustering import SOMCluster
from modules.art_network import ART1Network, ART2Network
from modules.fuzzy_logic import FuzzyDecisionMaker
from modules.genetic_optimizer import GeneticOptimizer
from modules.rl_agent import QLearningAgent
from modules.pca_analysis import PCAAnalyzer
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "iot_enhanced_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_pipeline():
    print("="*60)
    print("  SMART ENVIRONMENT AI - TRAINING PIPELINE")
    print("="*60)
    
    # 1. Load & Preprocess Data
    print("\n[1/9] Data Preprocessing...")
    dp = DataPreprocessor()
    df_raw = dp.load_data(DATA_FILE)
    df_clean = dp.clean_data(df_raw)
    df_derived = dp.add_derived_targets(df_clean)
    df_scaled = dp.normalize_data(df_derived)
    dp.save_processed(df_scaled, os.path.join(BASE_DIR, "data", "processed_data.csv"))
    
    # 2. Split Data (80/20)
    print("\n[2/9] Splitting Data (80% train / 20% test)...")
    df_train, df_test = train_test_split(df_scaled, test_size=0.2, random_state=42, stratify=df_scaled[LABEL_COL])
    df_train_raw, df_test_raw = train_test_split(df_derived, test_size=0.2, random_state=42, stratify=df_derived[LABEL_COL])
    
    print(f"   [OK] Training samples: {len(df_train)}")
    print(f"   [OK] Test samples: {len(df_test)}")
    print(f"   [OK] Train label distribution:\n{df_train[LABEL_COL].value_counts().to_string()}")
    print(f"   [OK] Test label distribution:\n{df_test[LABEL_COL].value_counts().to_string()}")
    
    # Extract labels for training and testing
    y_train_state = df_train[LABEL_COL].values
    y_train_trend = df_train[TREND_COL].values
    y_train_gas   = df_train[GAS_DNG_COL].values
    
    y_test_state = df_test[LABEL_COL].values
    y_test_trend = df_test[TREND_COL].values
    y_test_gas   = df_test[GAS_DNG_COL].values

    # 3. PCA Analysis
    print("\n[3/9] PCA Dimensionality Reduction...")
    pca = PCAAnalyzer(n_components=4)
    X_train_pca = pca.fit_transform(df_train)
    pca.plot_results(X_train_pca, df_train[LABEL_COL], 
                     save_path=os.path.join(BASE_DIR, "results", "pca_plot.png"))
    pca.save_results(X_train_pca, filepath=os.path.join(BASE_DIR, "data", "pca_features.csv"))

    # 4. RBF Network (Multi-output)
    print("\n[4/9] Training Multi-Output RBF Network...")
    rbf = RBFNetwork(n_centers=40)
    rbf.train(df_train_raw)
    rbf.save_model(os.path.join(MODELS_DIR, "rbf_model.pkl"))
    
    # Evaluate RBF on Test Data
    print("   [Evaluating on Test Data]")
    preds_rbf = rbf.predict(df_test_raw)
    acc_state = accuracy_score(y_test_state, preds_rbf["system_state"])
    acc_trend = accuracy_score(y_test_trend, preds_rbf["temp_trend"])
    acc_gas = accuracy_score(y_test_gas, preds_rbf["gas_danger"])
    print(f"   Test Accuracy - System State: {acc_state:.2%}")
    print(f"   Test Accuracy - Temp Trend: {acc_trend:.2%}")
    print(f"   Test Accuracy - Gas Danger: {acc_gas:.2%}")

    # 5. SOM Clustering (Environment mapping)
    print("\n[5/9] Training SOM Cluster...")
    som = SOMCluster(grid_size=(20,20))
    som.train(df_train, num_iterations=10000)
    som.label_neurons(y_train_state)
    som.plot_clusters(df_train, y_train_state, save_path=os.path.join(BASE_DIR, "results", "som_clusters.png"))
    som.save_model(os.path.join(MODELS_DIR, "som_model.pkl"))

    # 6. ART Networks
    print("\n[6/9] Training ART Networks...")
    art1 = ART1Network(vigilance=0.7)
    art1.train(df_train)
    art1.save_model(os.path.join(MODELS_DIR, "art1_model.pkl"))
    
    art2 = ART2Network(vigilance=0.85)
    art2.train(df_train)
    art2.save_model(os.path.join(MODELS_DIR, "art2_model.pkl"))

    # 7. Genetic Algorithm Optimization
    print("\n[7/9] Genetic Algorithm Optimization...")
    ga = GeneticOptimizer(population_size=20, generations=20)
    
    print("  -> Optimizing RBF Weights (system_state head)...")
    ga.optimise(rbf, df_train_raw, y_train_state, y_train_trend, y_train_gas, target="rbf_weights_state")
    
    print("  -> Optimizing RBF Weights (temp_trend head)...")
    ga.optimise(rbf, df_train_raw, y_train_state, y_train_trend, y_train_gas, target="rbf_weights_trend")
    
    print("  -> Optimizing RBF Weights (gas_danger head)...")
    ga.optimise(rbf, df_train_raw, y_train_state, y_train_trend, y_train_gas, target="rbf_weights_gas")
    
    # Evaluate optimized RBF on test data
    print("   [Evaluating Optimized RBF on Test Data]")
    preds_rbf_opt = rbf.predict(df_test_raw)
    acc_state_opt = accuracy_score(y_test_state, preds_rbf_opt["system_state"])
    print(f"   Test Accuracy (Optimized): {acc_state_opt:.2%}")
    
    # Save optimized RBF
    rbf.save_model(os.path.join(MODELS_DIR, "rbf_model_optimized.pkl"))
    ga.plot_convergence(save_path=os.path.join(BASE_DIR, "results", "ga_convergence.png"))

    # 8. Fuzzy Logic Calibration
    print("\n[8/9] Fuzzy Logic Decision Maker Setup...")
    fuzzy = FuzzyDecisionMaker()

    # 9. Reinforcement Learning
    print("\n[9/9] Reinforcement Learning Agent...")
    rl = QLearningAgent()
    
    # Pre-train RL Agent offline using simulated rewards
    rl.pretrain(iterations=10000)
    
    # Fine-tune on training data patterns
    print("   [Fine-tuning RL Agent on training data]")
    for idx in range(min(1000, len(df_train))):
        row = df_train.iloc[idx]
        sys_state = int(row[LABEL_COL])
        trend = int(row[TREND_COL])
        gas = int(row[GAS_DNG_COL])
        
        state = (sys_state, trend, gas)
        action = rl.choose_action(*state, explore=True)
        reward = 1.0 if sys_state == 0 else (0.5 if sys_state == 1 else -1.0)
        rl.update(state, action, reward, state)
    
    rl.save_model(os.path.join(MODELS_DIR, "rl_model.pkl"))

    print("\n" + "="*60)
    print("  TRAINING COMPLETE.")
    print("  Models saved to /models. Ready for MQTT inference.")
    print("="*60)

if __name__ == "__main__":
    start_time = time.time()
    os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)
    train_pipeline()
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")