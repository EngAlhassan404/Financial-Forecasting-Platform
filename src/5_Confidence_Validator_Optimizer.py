# -*- coding: utf-8 -*-
# =====================================================================================
# # Confidence Validator Optimizer - V1.0
#
# ## Purpose:
# This script performs hyperparameter tuning for the RandomForestClassifier model,
# which serves as the "Confidence Validator." It systematically tests a range of
# `n_estimators` values to identify the optimal parameter that maximizes the
# precision of correctly identifying a reliable Markov forecast.
#
# ## Role in Pipeline:
# This is an essential optimization and research tool. It is executed after the
# data has been enriched with Markov features (by 4_Feature_Enrichment_Engine.py).
# The optimal `estimators` value discovered by this script is then manually
# transferred to the asset's configuration in the main forecasting engine to
# improve its real-time performance.
#
# ## Execution Order:
# 5th - Run this script to find the best parameters for the validator model.
#
# ## Key Features:
# - Object-Oriented Design: Encapsulates the logic for each asset in a clean class.
# - Hyperparameter Sweeping: Iterates through a defined range of `n_estimators`.
# - Precision-Focused Ranking: Ranks results based on the precision for the
#   'RIGHT' class, as the primary goal is to correctly identify winning signals.
# =====================================================================================

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
import time
from typing import List, Dict, Tuple, Optional

# --- 1. Core Settings ---
warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1200)

# --- 2. Centralized Settings & Control Panel ---

MAIN_SCRIPT_NAME = "Professional_Asset_Confidence_Analyzer_V5"

ASSETS_TO_PROCESS: List[Dict[str, str]] = [
    {"name": "USDJPY", "enriched_file": "USDJPY_Enriched.csv"},
    {"name": "Gold", "enriched_file": "Gold_Enriched.csv"},
    {"name": "Silver", "enriched_file": "Silver_Enriched.csv"},
    {"name": "WTI", "enriched_file": "WTI_Enriched.csv"},
    {"name": "DXY", "enriched_file": "DXY_Enriched.csv"}
]

ENRICHED_DATA_FOLDER = Path(r"D:/Machine Learning/Data/Forex/Enriched_Data")

# RandomForest Model Settings
N_ESTIMATORS_RANGE = range(50, 501, 50) # Search range for n_estimators
RANDOM_STATE = 42
TRAIN_TEST_SPLIT_DATE = '2025-01-01'

# New: Define how often to log progress during the search loop
LOG_UPDATE_FREQUENCY = 20

# --- 3. Main Asset Analyzer Class ---

class AssetConfidenceAnalyzer:
    """
    A professional class that manages the end-to-end confidence analysis
    process for a single financial asset.
    """
    def __init__(self, asset_config: Dict[str, str], asset_number: int, total_assets: int):
        self.asset_name = asset_config["name"]
        self.enriched_file = asset_config["enriched_file"]
        self.input_path = ENRICHED_DATA_FOLDER / self.asset_name / self.enriched_file
        self.results = []
        # Print a header for the asset being processed, including a counter
        header = f"🚀 PROCESSING ASSET {asset_number}/{total_assets}: {self.asset_name} 🚀"
        print("\n" + "="*len(header))
        print(header)
        print("="*len(header))

    def _load_and_prepare_data(self) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """
        Loads and prepares the data for the prediction model.
        """
        print(f"  [1/4] 🔄 Loading and preparing data from '{self.input_path.name}'...")
        try:
            df_full = pd.read_csv(self.input_path, parse_dates=['Date']).set_index('Date')
        except FileNotFoundError:
            print(f"  [ERROR] ❌ Data file not found. Skipping this asset.")
            return None

        try:
            pred_direction_col = next(c for c in df_full.columns if '_Pred_Direction' in c)
            prob_up_col = next(c for c in df_full.columns if '_Prob_UP' in c)
            prob_down_col = next(c for c in df_full.columns if '_Prob_DOWN' in c)
        except StopIteration:
            print(f"  [ERROR] ❌ Required Markov columns not found. Skipping this asset.")
            return None

        df_clean = df_full[['Direction', pred_direction_col, prob_up_col, prob_down_col]].dropna()
        df_clean['is_markov_correct'] = np.where(df_clean['Direction'] == df_clean[pred_direction_col], 1, 0)

        y = df_clean['is_markov_correct']
        X = df_clean[[pred_direction_col, prob_up_col, prob_down_col]].copy()
        X[pred_direction_col] = X[pred_direction_col].map({'UP': 1, 'DOWN': 0})
        X.columns = ['Feature_Pred_Direction', 'Feature_Prob_UP', 'Feature_Prob_DOWN']
        
        print(f"  [SUCCESS] ✅ Data loaded. Found {len(X)} usable rows.")
        return X, y

    def _display_results(self):
        """
        Displays the top 3 results in a formatted table.
        """
        print(f"  [4/4] 📈 Generating final report for {self.asset_name}...")
        
        if not self.results:
            print("  [WARNING] ⚠️ No results were generated to display.")
            return

        results_df = pd.DataFrame(self.results)
        results_df.sort_values(by="Precision_RIGHT", ascending=False, inplace=True)
        
        top_3_results = results_df.head(3).copy()
        top_3_results.insert(0, 'Rank', range(1, len(top_3_results) + 1))
        
        top_3_results.rename(columns={
            'N_Estimators': 'Estimators',
            'Precision_RIGHT': 'Precision (RIGHT)',
            'Recall_RIGHT': 'Recall (RIGHT)',
            'F1_Score_RIGHT': 'F1-Score (RIGHT)'
        }, inplace=True)
        
        for col in ['Accuracy', 'Precision (RIGHT)', 'Recall (RIGHT)', 'F1-Score (RIGHT)']:
            top_3_results[col] = top_3_results[col].map('{:.2%}'.format)

        print("\n" + "-"*80)
        print(f"    Top 3 Results for [{self.asset_name}] (Ranked by Precision)")
        print(top_3_results.to_string(index=False))
        print("-"*80)
        print(f"  [SUCCESS] ✅ Report for {self.asset_name} is complete.")

    def run_analysis(self):
        """
        The execution function that manages the full workflow for the asset.
        """
        data_tuple = self._load_and_prepare_data()
        if not data_tuple: return

        X, y = data_tuple
        
        print(f"  [2/4] 📊 Splitting data into training/testing sets...")
        train_mask = X.index < TRAIN_TEST_SPLIT_DATE
        test_mask = X.index >= TRAIN_TEST_SPLIT_DATE
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if X_train.empty or X_test.empty:
            print("  [ERROR] ❌ Not enough data for training/testing split. Skipping.")
            return
            
        print(f"  [INFO] Train set size: {len(X_train)} | Test set size: {len(X_test)}")
        print(f"  [3/4] ⚙️  Starting hyperparameter search...")
        
        num_iterations = len(N_ESTIMATORS_RANGE)
        
        # Main search loop
        for i, n_estimators in enumerate(N_ESTIMATORS_RANGE):
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=RANDOM_STATE, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            report_dict = classification_report(y_test, y_pred, target_names=['WRONG', 'RIGHT'], output_dict=True, zero_division=0)
            
            self.results.append({
                "N_Estimators": n_estimators, "Accuracy": accuracy,
                "Precision_RIGHT": report_dict.get('RIGHT', {}).get('precision', 0),
                "Recall_RIGHT": report_dict.get('RIGHT', {}).get('recall', 0),
                "F1_Score_RIGHT": report_dict.get('RIGHT', {}).get('f1-score', 0)
            })

            # Periodic progress update instead of a progress bar
            if (i + 1) % LOG_UPDATE_FREQUENCY == 0 and (i + 1) < num_iterations:
                print(f"    -> Still working... Processed {i + 1}/{num_iterations} parameter sets.")
        
        print(f"  [SUCCESS] ✅ Hyperparameter search completed after {num_iterations} iterations.")
        self._display_results()

# --- 4. Program Entry Point ---

def main():
    """
    The main function that orchestrates the entire program workflow.
    """
    print(f"--- Running: {MAIN_SCRIPT_NAME} ---")
    start_time = time.time()
    
    total_assets = len(ASSETS_TO_PROCESS)
    
    try:
        for i, asset_config in enumerate(ASSETS_TO_PROCESS):
            analyzer = AssetConfidenceAnalyzer(asset_config, asset_number=i + 1, total_assets=total_assets)
            analyzer.run_analysis()
    except Exception as e:
        print(f"\n[CRITICAL ERROR] ❌ An unexpected error halted the script: {e}")
    finally:
        end_time = time.time()
        total_time = end_time - start_time
        print("\n" + "="*50)
        print(f"🎉 All tasks completed. 🎉")
        print(f"Total execution time: {total_time:.2f} seconds.")
        print(f"--- {MAIN_SCRIPT_NAME}: Finished ---")

if __name__ == "__main__":
    main()