# -*- coding: utf-8 -*-
# =====================================================================================
# # Historical Forecast Generator w/ Integrated Validation - V3.3
#
# ## Purpose:
# This script generates (or regenerates) historical forecasts for a specific,
# user-defined date range across a portfolio of assets. For assets where
# validation is enabled, it also trains a "Confidence Validator" model on
# prior data and enriches each new forecast with a confidence score and a
# recommendation (Excellent, Good, Weak).
#
# ## Role in Pipeline:
# This is an advanced utility for back-filling missing data, correcting historical
# forecast logs, or creating rich, validated datasets for deep performance
# analysis. It is a core tool for research and ensuring data integrity.
#
# ## Execution Order:
# 6th - This script can be run on-demand as needed to generate historical data
#       for analysis by the Performance Analyzer.
#
# ## Key Features:
# - Walk-forward generation for a specified historical window.
# - Smart pathing to handle standard assets and custom-named sub-models.
# - Integrated Confidence Validation for designated assets.
# - Intelligent log updating: preserves forecasts outside the specified
#   generation range.
# =====================================================================================

import pandas as pd
import numpy as np
import os
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import warnings

# --- 1. Core Settings ---
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1200)
warnings.filterwarnings('ignore', category=FutureWarning)

SCRIPT_NAME = "Historical_Forecast_Generator_V3"
print(f"--- Running: {SCRIPT_NAME} ---")
print("="*80)

# --- 2. Unified Asset Configuration ---
ASSETS_CONFIG = [
    {"asset_name": "USDJPY", "markov_order": 4, "input_file_name": "USDJPY.csv", "is_enriched": False, "is_validated": False, "confluence_partner": None, "output_name": "USDJPY"},
    {"asset_name": "Gold", "markov_order": 5, "estimators": 50, "input_file_name": "Gold.csv", "is_enriched": True, "is_validated": True, "confluence_partner": None},
    {"asset_name": "Silver", "markov_order": 1, "input_file_name": "Silver.csv", "is_enriched": True, "is_validated": False, "confluence_partner": None},
    {"asset_name": "WTI", "markov_order": 5, "estimators": 135, "input_file_name": "WTI.csv", "is_enriched": True, "is_validated": True, "confluence_partner": None},
    {"asset_name": "DXY", "markov_order": 1, "input_file_name": "DXY.csv", "is_enriched": True, "is_validated": False, "confluence_partner": None},
]

# --- Generation Period Control ---
# Use 'YYYY-MM-DD' format
GENERATION_START_DATE_STR = "2025-07-01"
GENERATION_END_DATE_STR = "2025-07-03"
# --- End of Control Section ---

# --- 3. Base Path Configuration ---
BASE_PROJECT_PATH = Path(r"D:\Machine Learning\Projects\Markov Forecaster")
RAW_DATA_FOLDER = Path(r"D:\Machine Learning\Data\FOREX")
ENRICHED_DATA_FOLDER = Path(r"D:\Machine Learning\Data\Forex\Enriched_Data")
VALIDATOR_RANDOM_STATE = 42

# --- 4. Helper Functions ---
def build_markov_transition_matrix(data_frame, markov_order):
    transition_counts = defaultdict(Counter)
    if len(data_frame) < markov_order: return transition_counts
    for i in range(markov_order, len(data_frame)):
        previous_sequence = tuple(data_frame['Direction'].iloc[i-markov_order : i])
        current_direction = data_frame['Direction'].iloc[i]
        transition_counts[previous_sequence][current_direction] += 1
    return transition_counts

def predict_next_direction(df_data, transition_counts, markov_order):
    last_known_sequence = tuple(df_data['Direction'].iloc[-markov_order:])
    sequence_counts = transition_counts.get(last_known_sequence, Counter())
    up_count, down_count = sequence_counts.get('UP', 0), sequence_counts.get('DOWN', 0)
    total_frequency = up_count + down_count
    historical_frequency_str = f"F:{total_frequency} - U:{up_count} - D:{down_count}"
    prob_up = up_count / total_frequency if total_frequency > 0 else 0.5
    prob_down = down_count / total_frequency if total_frequency > 0 else 0.5
    if prob_up > prob_down:
        prediction, probability, reason = 'UP', prob_up, "Direct Prediction"
    elif prob_down > prob_up:
        prediction, probability, reason = 'DOWN', prob_down, "Direct Prediction"
    else:
        prediction, probability = df_data['Direction'].iloc[-1], 0.5
        reason = "Tie in probabilities" if total_frequency > 0 else "New Sequence (Used Last Direction)"
    return prediction, probability, prob_up, prob_down, historical_frequency_str, reason

def train_validator_model(config, training_data):
    asset_name = config["asset_name"]
    n_estimators = config.get("estimators", 100)
    print(f"    -> Training confidence model for {asset_name} (Estimators: {n_estimators})...")
    try:
        base_name = asset_name.split('_')[0]
        enriched_path = ENRICHED_DATA_FOLDER / base_name / config["input_file_name"].replace('.csv', '_Enriched.csv')
        if not enriched_path.exists():
            raise FileNotFoundError(f"Enriched file for validation not found at {enriched_path}")
        
        df_enriched = pd.read_csv(enriched_path, parse_dates=['Date'])
        df_validator_train = df_enriched[df_enriched['Date'].isin(training_data['Date'])].set_index('Date')
        
        pred_col = next((c for c in df_validator_train.columns if '_Pred_Direction' in c), None)
        up_col = next((c for c in df_validator_train.columns if '_Prob_UP' in c), None)
        down_col = next((c for c in df_validator_train.columns if '_Prob_DOWN' in c), None)
        
        if not all([pred_col, up_col, down_col]):
             raise ValueError("Required Markov prediction columns not found in enriched training file.")

        df_validator_train['is_markov_correct'] = np.where(df_validator_train['Direction'] == df_validator_train[pred_col], 1, 0)
        y_train = df_validator_train['is_markov_correct']
        X_train = pd.DataFrame({'Pred': df_validator_train[pred_col].map({'UP': 1, 'DOWN': 0}), 'Prob_UP': df_validator_train[up_col], 'Prob_DOWN': df_validator_train[down_col]}).dropna()
        y_train = y_train.reindex(X_train.index)
        
        if len(X_train) < 50:
            print(f"    -> WARNING: Not enough training data for validator ({len(X_train)} rows). Validation may be unreliable.")
            return None

        model = RandomForestClassifier(n_estimators=n_estimators, random_state=VALIDATOR_RANDOM_STATE, n_jobs=-1).fit(X_train, y_train)
        print(f"    -> Confidence model for {asset_name} trained successfully.")
        return model
    except Exception as e:
        print(f"    -> ERROR during validator training for {asset_name}: {e}")
        return None

def get_confidence_for_forecast(validator_model, forecast_row):
    if validator_model is None:
        return {'Validator Confirmation Percentage': np.nan, 'Recommendation': ''}
    
    features = pd.DataFrame([{'Pred': 1 if forecast_row['Predicted_Direction'] == 'UP' else 0,
                              'Prob_UP': forecast_row['Prob_UP'], 'Prob_DOWN': forecast_row['Prob_DOWN']}])
    score = validator_model.predict_proba(features)[:, 1][0]
    
    if score > 0.65: recommendation = "Excellent"
    elif score > 0.55: recommendation = "Good"
    else: recommendation = "Weak"
    
    return {'Validator Confirmation Percentage': score, 'Recommendation': recommendation}

# --- 5. Main Workflow ---
def generate_historical_forecasts():
    try:
        # Pandas is smart enough to infer 'YYYY-MM-DD' format automatically
        gen_start_date = pd.to_datetime(GENERATION_START_DATE_STR)
        gen_end_date = pd.to_datetime(GENERATION_END_DATE_STR)
    except Exception as e:
        print(f"CRITICAL ERROR: Invalid date format in settings. Please check. Details: {e}")
        return

    for config in ASSETS_CONFIG:
        model_name = config.get("output_name", config["asset_name"])
        markov_order = config["markov_order"]
        asset_name = config["asset_name"]
        
        print(f"\n{'#'*88}\n# Processing Model: {model_name} (Order: {markov_order})\n{'#'*88}")

        base_asset_name = asset_name.split('_')[0]
        forecast_log_folder = BASE_PROJECT_PATH / model_name / f"{model_name}_Order{markov_order}_Daily_Forecasts"
        forecast_log_path = forecast_log_folder / f"{model_name}_MarkovO{markov_order}_DailyForecasts.csv"
        raw_data_path = RAW_DATA_FOLDER / base_asset_name / f"{base_asset_name}.csv"
        
        print(f"  -> Reading raw data from: {raw_data_path}")
        print(f"  -> Target forecast log: {forecast_log_path}")

        try:
            df_full = pd.read_csv(raw_data_path, parse_dates=['Date'])
            df_full.sort_values(by='Date', inplace=True, ignore_index=True)
            df_analysis_data = df_full[df_full['Direction'].isin(['UP', 'DOWN'])].copy()
        except Exception as e:
            print(f"--> ERROR: Could not load raw data. Skipping. Details: {e}"); continue
        
        simulation_data = df_analysis_data[(df_analysis_data['Date'] >= gen_start_date) & (df_analysis_data['Date'] <= gen_end_date)].copy()
        if simulation_data.empty:
            print(f"INFO: No raw data in the specified generation period. Skipping."); continue
        
        first_sim_day = simulation_data['Date'].min()
        initial_history = df_analysis_data[df_analysis_data['Date'] < first_sim_day].copy()
        if len(initial_history) < markov_order:
            print(f"--> ERROR: Not enough historical data before {first_sim_day:%Y-%m-%d}. Skipping."); continue
        
        df_log_to_keep = pd.DataFrame()
        if os.path.exists(forecast_log_path):
            try:
                df_existing_log = pd.read_csv(forecast_log_path, parse_dates=['Forecast_Date'])
                # Keep only forecasts outside the range we are about to regenerate
                df_log_to_keep = df_existing_log[(df_existing_log['Forecast_Date'] < gen_start_date) | (df_existing_log['Forecast_Date'] > gen_end_date)].copy()
                print(f"INFO: Existing log found. Preserving {len(df_log_to_keep)} forecasts outside the target range.")
            except Exception as e:
                print(f"--> WARNING: Could not read existing log file. Details: {e}")

        validator_model = None
        if config.get("is_validated"):
            validator_model = train_validator_model(config, initial_history)

        print(f"Generating forecasts for the period: {simulation_data['Date'].min():%Y-%m-%d} to {simulation_data['Date'].max():%Y-%m-%d}.")
        generated_forecasts = []
        temp_history = initial_history.copy()

        for idx, current_day in simulation_data.iterrows():
            print(f"\r   Forecasting day {idx - simulation_data.index[0] + 1}/{len(simulation_data)}: {current_day['Date']:%Y-%m-%d}", end="", flush=True)
            transition_counts = build_markov_transition_matrix(temp_history, markov_order)
            pred, prob, p_up, p_down, freq_str, reason = predict_next_direction(temp_history, transition_counts, markov_order)
            
            forecast_entry = {
                'Forecast_Date': current_day['Date'], 'Forecast_Day_Name': current_day['Date'].strftime('%A'),
                'Data_Used_Up_To': temp_history['Date'].iloc[-1],
                'Last_N_Day_Actual_Sequence': str(tuple(temp_history['Direction'].iloc[-markov_order:])),
                'Predicted_Direction': pred, 'Historical_Frequency_FUD': freq_str,
                'Prediction_Probability': prob, 'Prob_UP': p_up, 'Prob_DOWN': p_down,
                'Fallback_Reason': reason, 'Forecast_Generated_At': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            if validator_model:
                validation_results = get_confidence_for_forecast(validator_model, forecast_entry)
                forecast_entry.update(validation_results)

            generated_forecasts.append(forecast_entry)
            temp_history = pd.concat([temp_history, pd.DataFrame([current_day])], ignore_index=True)
        
        print(f"\nGeneration complete. Created {len(generated_forecasts)} new forecasts.")

        df_new_forecasts = pd.DataFrame(generated_forecasts)
        # Combine the forecasts we kept with the newly generated ones
        df_final_log = pd.concat([df_log_to_keep, df_new_forecasts], ignore_index=True)
        df_final_log.sort_values(by='Forecast_Date', inplace=True, ignore_index=True)
        
        try:
            os.makedirs(forecast_log_folder, exist_ok=True)
            df_final_log.to_csv(forecast_log_path, index=False)
            print(f"✅ Forecast Log for {model_name} successfully written/overwritten.")
        except Exception as e:
            print(f"--> ERROR: Could not save the updated log file. Details: {e}")

if __name__ == "__main__":
    generate_historical_forecasts()
    print(f"\n{SCRIPT_NAME} - All configured asset logs generated successfully!")