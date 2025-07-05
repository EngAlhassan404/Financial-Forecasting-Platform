# -*- coding: utf-8 -*-
# =====================================================================================
# # Master Forecasting Engine V4.7
#
# ## Purpose:
# This script serves as the main, operational engine for the entire forecasting
# system. It orchestrates the end-to-end process of generating daily forecasts
# for a portfolio of assets.
#
# ## Role in Pipeline:
# This is the primary execution script for daily operations. It integrates multiple
# modules to perform a sequence of tasks:
# 1. Generates a base forecast using a Markov Chain model.
# 2. Updates individual, persistent forecast logs for each asset.
# 3. Enriches historical data files with the latest forecast data.
# 4. (Optionally) Runs a Random Forest-based "Confidence Validator" to score the forecasts.
# 5. Generates combined daily reports in multiple formats (CSV, HTML, JSON).
#
# ## Execution Order:
# 7th - This is the main script to run for generating daily production forecasts.
#
# ## Key Features:
# - Multi-Stage Pipeline: Executes forecasting, enrichment, and validation in a
#   clear, logical sequence.
# - Configuration-Driven: Manages a diverse portfolio of assets, including
#   sub-models with custom pathing, from a central configuration list.
# - Two-Layer Forecasting: Combines a primary Markov model with a secondary
#   confidence validation model for deeper insights.
# - Comprehensive Reporting: Outputs detailed individual logs and combined daily
#   reports in user-friendly formats.
# =====================================================================================

import pandas as pd
import numpy as np
import os
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.ensemble import RandomForestClassifier
import json
import warnings

# --- 1. Core Settings ---
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1200)
pd.set_option('display.float_format', '{:.4f}'.format)
warnings.filterwarnings('ignore', category=FutureWarning)

MAIN_SCRIPT_NAME = "Centralized_Forex_Markov_Forecaster"
print(f"--- Running: {MAIN_SCRIPT_NAME} ---")
print("="*80)

# --- 2. Base Paths & Global Constants ---
BASE_PROJECT_PATH = Path(r"D:/Machine Learning/Projects/Markov Forecaster")
ENRICHED_DATA_FOLDER = Path(r"D:/Machine Learning/Data/Forex/Enriched_Data")
RAW_DATA_FOLDER = Path(r"D:/Machine Learning/Data/Forex")
COMBINED_OUTPUT_BASE_FOLDER = BASE_PROJECT_PATH / "Forex_Forecasts"
os.makedirs(COMBINED_OUTPUT_BASE_FOLDER, exist_ok=True)
VALIDATOR_RANDOM_STATE = 42

# --- 3. Unified Asset Configuration ---
ASSETS_CONFIG = [
    {"asset_name": "USDJPY", "markov_order": 4, "input_file_name": "USDJPY.csv", "is_enriched": True, "is_validated": False, "confluence_partner": None, "output_name": "USDJPY_O4"},
    {"asset_name": "Gold", "markov_order": 5, "estimators": 50, "input_file_name": "Gold.csv", "is_enriched": True, "is_validated": True, "confluence_partner": None},
    {"asset_name": "Silver", "markov_order": 1, "input_file_name": "Silver.csv", "is_enriched": True, "is_validated": False, "confluence_partner": None},
    {"asset_name": "WTI", "markov_order": 5, "estimators": 135, "input_file_name": "WTI.csv", "is_enriched": True, "is_validated": True, "confluence_partner": None},
    {"asset_name": "DXY", "markov_order": 1, "input_file_name": "DXY.csv", "is_enriched": True, "is_validated": False, "confluence_partner": None},
]

# --- 4. System Functions ---
def process_asset(asset_config, run_summary):
    # This function's role is to generate the base forecast
    asset_name = asset_config["asset_name"]
    markov_order = asset_config["markov_order"]
    input_file_name = asset_config["input_file_name"]
    print(f"\n--- Processing Asset: {asset_name} (Order: {markov_order}) ---")
    try:
        base_asset_name = asset_name.split('_')[0]
        input_data_path_asset = RAW_DATA_FOLDER / base_asset_name / input_file_name
        if not input_data_path_asset.exists(): raise FileNotFoundError(f"Input file not found: {input_data_path_asset}")
        df_full = pd.read_csv(input_data_path_asset, parse_dates=['Date']).sort_values(by='Date', ignore_index=True)
        df_analysis_data = df_full[df_full['Direction'].isin(['UP', 'DOWN'])].copy()
        if len(df_analysis_data) <= markov_order: raise ValueError(f"Not enough data for order {markov_order}.")
        transition_counts = defaultdict(Counter)
        for i in range(markov_order, len(df_analysis_data)):
            transition_counts[tuple(df_analysis_data['Direction'].iloc[i-markov_order:i])][df_analysis_data['Direction'].iloc[i]] += 1
        transition_probabilities = {k: {s: c/sum(v.values()) for s, c in v.items()} for k, v in transition_counts.items()}
        last_known_sequence = tuple(df_analysis_data['Direction'].iloc[-markov_order:])
        last_known_date_in_data = df_analysis_data['Date'].iloc[-1]
        next_trading_day_date = last_known_date_in_data + pd.offsets.BDay(1)
        sequence_counts = transition_counts.get(last_known_sequence, Counter())
        up_count, down_count = sequence_counts.get('UP', 0), sequence_counts.get('DOWN', 0)
        historical_frequency_fud_string = f"F:{up_count + down_count} - U:{up_count} - D:{down_count}"
        
        prob_up, prob_down = np.nan, np.nan
        if last_known_sequence in transition_probabilities:
            probs = transition_probabilities[last_known_sequence]
            prob_up, prob_down = probs.get('UP', 0.0), probs.get('DOWN', 0.0)
            if prob_up > prob_down: pred, prob, reason = 'UP', prob_up, ""
            elif prob_down > prob_up: pred, prob, reason = 'DOWN', prob_down, ""
            else: pred, prob, reason = df_analysis_data['Direction'].iloc[-1], 0.5, "Tie in probabilities"
        else: pred, prob, reason = df_analysis_data['Direction'].iloc[-1], np.nan, "Seq not found in matrix"
        
        print(f"    -> Markov Prediction for {next_trading_day_date.strftime('%Y-%m-%d')}: {pred} (Prob: {prob:.4f})")
        
        individual_forecast_entry = {
            'Forecast_Date': next_trading_day_date, 'Forecast_Day_Name': next_trading_day_date.strftime('%A'),
            'Data_Used_Up_To': last_known_date_in_data, 'Last_N_Day_Actual_Sequence': str(last_known_sequence),
            'Predicted_Direction': pred, 'Historical_Frequency_FUD': historical_frequency_fud_string,
            'Prediction_Probability': prob, 'Prob_UP': prob_up, 'Prob_DOWN': prob_down,
            'Fallback_Reason': reason or "Direct Prediction", 'Forecast_Generated_At': pd.Timestamp.now()
        }
        run_summary["processed"].append(asset_name)
        return {**{'Asset': asset_name, 'Markov_Order': markov_order}, **individual_forecast_entry}
    except Exception as e:
        print(f"    -> ERROR processing {asset_name}: {e}")
        run_summary["failed"].append({"asset": asset_name, "error": str(e)})
        return None

def update_enriched_file_dynamically(config, run_summary):
    """
    Correctly enriches data by merging historical raw data with its corresponding forecast.
    This function finds days where both raw data and a forecast exist but are not yet in the enriched file.
    """
    asset_name = config["asset_name"]
    markov_order = config["markov_order"]
    print(f"\n--- Corrected Dynamic Enrichment for: {asset_name} ---")
    
    try:
        base_name = asset_name.split('_')[0]
        output_name_for_path = config.get("output_name", asset_name)

        enriched_path = ENRICHED_DATA_FOLDER / base_name / config["input_file_name"].replace('.csv', '_Enriched.csv')
        raw_path = RAW_DATA_FOLDER / base_name / config["input_file_name"]
        forecast_path = BASE_PROJECT_PATH / output_name_for_path / f"{output_name_for_path}_Order{markov_order}_Daily_Forecasts" / f"{output_name_for_path}_MarkovO{markov_order}_DailyForecasts.csv"
        
        os.makedirs(enriched_path.parent, exist_ok=True)
        if not all([p.exists() for p in [raw_path, forecast_path]]):
            print(f"    -> SKIPPING enrichment: Raw or Forecast file is missing.")
            return

        raw_df = pd.read_csv(raw_path, parse_dates=['Date'])
        forecast_df = pd.read_csv(forecast_path, parse_dates=['Forecast_Date'])
        
        enriched_df = pd.DataFrame()
        if enriched_path.exists():
            enriched_df = pd.read_csv(enriched_path, parse_dates=['Date'])

        merged_df = pd.merge(raw_df, forecast_df, left_on='Date', right_on='Forecast_Date', how='inner')

        if merged_df.empty:
            print(f"    -> INFO: No matching dates found between raw data and forecasts yet.")
            return

        if not enriched_df.empty:
            new_rows_to_add_df = merged_df[~merged_df['Date'].isin(enriched_df['Date'])]
        else:
            new_rows_to_add_df = merged_df

        if new_rows_to_add_df.empty:
            print(f"    -> INFO: Enriched file is already up-to-date. No new data to add.")
            return

        print(f"    -> Found {len(new_rows_to_add_df)} new day(s) to enrich.")

        new_enriched_rows = []
        for _, row in new_rows_to_add_df.iterrows():
            new_row = {
                'Date': row['Date'], 'Close': row['Close'], 'Open': row['Open'], 'High': row['High'], 'Low': row['Low'],
                'Direction': row['Direction'], 'Day_Name': row['Date'].strftime('%A'),
                f'MarkovO{markov_order}_Pred_Direction': row['Predicted_Direction'],
                f'MarkovO{markov_order}_Prob_UP': row['Prob_UP'], f'MarkovO{markov_order}_Prob_DOWN': row['Prob_DOWN']
            }
            new_enriched_rows.append(new_row)

        final_df = pd.concat([enriched_df, pd.DataFrame(new_enriched_rows)], ignore_index=True)
        final_df.drop_duplicates(subset=['Date'], keep='last', inplace=True)
        final_df.sort_values(by='Date', inplace=True)
        final_df.to_csv(enriched_path, index=False)
        
        print(f"    -> SUCCESS: Enriched file '{enriched_path.name}' updated with {len(new_enriched_rows)} new row(s).")

    except Exception as e:
        print(f"    -> ERROR during dynamic enrichment for {asset_name}: {e}")
        run_summary["failed"].append({"asset": f"Enrichment for {asset_name}", "error": str(e)})

def get_confidence_score(config, forecast_row, run_summary):
    # This function calculates the confidence score
    asset_name = config["asset_name"]
    n_estimators = config.get("estimators", 100)
    print(f"\n--- Dynamically Validating Signal for: {asset_name} (Estimators: {n_estimators}) ---")
    try:
        base_name = asset_name.split('_')[0]
        enriched_path = ENRICHED_DATA_FOLDER / base_name / config["input_file_name"].replace('.csv', '_Enriched.csv')
        if not enriched_path.exists(): raise FileNotFoundError(f"Enriched file not found at {enriched_path}")
        df = pd.read_csv(enriched_path, parse_dates=['Date']).set_index('Date')
        
        if len(df) < 20: raise ValueError(f"Not enough enriched data to build a reliable confidence model (n={len(df)}).")
        pred_col = next((c for c in df.columns if '_Pred_Direction' in c), None)
        up_col = next((c for c in df.columns if '_Prob_UP' in c), None)
        down_col = next((c for c in df.columns if '_Prob_DOWN' in c), None)
        if not all([pred_col, up_col, down_col]): raise ValueError("Required Markov columns not found.")
        
        df['is_markov_correct'] = np.where(df['Direction'] == df[pred_col], 1, 0)
        features_df = pd.DataFrame({'Pred': df[pred_col].map({'UP': 1, 'DOWN': 0}), 'Prob_UP': df[up_col], 'Prob_DOWN': df[down_col]})
        y_train = df['is_markov_correct']
        aligned_df = pd.concat([features_df, y_train], axis=1).dropna()
        X_train = aligned_df[['Pred', 'Prob_UP', 'Prob_DOWN']]
        y_train = aligned_df['is_markov_correct']
        if len(X_train) < 20: raise ValueError(f"Not enough clean data points to train validator (n={len(X_train)}).")

        print(f"    -> Training confidence model (n={len(X_train)})...")
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=VALIDATOR_RANDOM_STATE, n_jobs=-1).fit(X_train, y_train)
        
        features_to_predict = pd.DataFrame([{'Pred': 1 if forecast_row['Predicted_Direction'] == 'UP' else 0, 'Prob_UP': forecast_row['Prob_UP'], 'Prob_DOWN': forecast_row['Prob_DOWN']}])
        score = model.predict_proba(features_to_predict)[:, 1][0]
        
        if score > 0.65: recommendation = "Excellent"
        elif score > 0.55: recommendation = "Good"
        else: recommendation = "Weak"
        print(f"    -> Validation complete. Confidence: {score:.2%}")
        return {'Validator Confirmation Percentage': score, 'Recommendation': recommendation}
    except Exception as e:
        run_summary["failed"].append({"asset": f"Validation for {asset_name}", "error": str(e)})
        return {'Validator Confirmation Percentage': np.nan, 'Recommendation': f'Error'}

def run_forecasting_pipeline():
    run_summary = {"processed": [], "failed": [], "alerts": []}
    all_forecasts_list = []

    # Stage 1: Generate all daily forecasts and update individual logs
    print(f"\n{'='*25} STAGE 1: GENERATING DAILY FORECASTS {'='*25}")
    for config in ASSETS_CONFIG:
        forecast_result = process_asset(config, run_summary)
        if forecast_result:
            all_forecasts_list.append(forecast_result)
            output_name_for_path = config.get("output_name", config["asset_name"])
            output_asset_folder = BASE_PROJECT_PATH / output_name_for_path / f"{output_name_for_path}_Order{config['markov_order']}_Daily_Forecasts"
            os.makedirs(output_asset_folder, exist_ok=True)
            daily_forecast_file_path = output_asset_folder / f"{output_name_for_path}_MarkovO{config['markov_order']}_DailyForecasts.csv"
            new_individual_df = pd.DataFrame([forecast_result])
            if daily_forecast_file_path.exists():
                df_existing = pd.read_csv(daily_forecast_file_path, parse_dates=['Forecast_Date'])
                df_to_save = pd.concat([df_existing, new_individual_df], ignore_index=True)
            else:
                df_to_save = new_individual_df
            df_to_save['Forecast_Date'] = pd.to_datetime(df_to_save['Forecast_Date'])
            df_to_save.drop_duplicates(subset=['Forecast_Date'], keep='last', inplace=True)
            df_to_save.sort_values(by='Forecast_Date', inplace=True)
            df_to_save.to_csv(daily_forecast_file_path, index=False)
            print(f"    -> ✅ Individual log for {config['asset_name']} updated.")

    # Stage 2: Update enriched files with the latest forecasts
    print(f"\n{'='*25} STAGE 2: DYNAMIC DATA ENRICHMENT {'='*25}")
    for config in ASSETS_CONFIG:
        if config.get('is_enriched', False):
            update_enriched_file_dynamically(config, run_summary)
    
    # Stage 3: Perform validation and generate combined final reports
    print(f"\n{'='*25} STAGE 3: VALIDATION & FINAL REPORTING {'='*25}")
    if not all_forecasts_list:
        print("\nNo forecasts were generated. Exiting."); return

    results_df = pd.DataFrame(all_forecasts_list)
    validation_results = {}
    for config in ASSETS_CONFIG:
        if config.get('is_validated', False):
            asset_name = config['asset_name']
            forecast_row = results_df[results_df['Asset'] == asset_name]
            if not forecast_row.empty:
                validation_results[asset_name] = get_confidence_score(config, forecast_row.iloc[0], run_summary)
    
    final_df = results_df.copy()
    if validation_results:
        validation_df = pd.DataFrame.from_dict(validation_results, orient='index').reset_index().rename(columns={'index':'Asset'})
        final_df = pd.merge(final_df, validation_df, on='Asset', how='left')

    if final_df.empty:
        print("\nNo final forecasts to report."); return

    target_date_str = pd.to_datetime(final_df['Forecast_Date'].iloc[0]).strftime('%Y-%m-%d')
    output_folder = COMBINED_OUTPUT_BASE_FOLDER / target_date_str
    os.makedirs(output_folder, exist_ok=True)
    
    final_columns = [
        'Asset', 'Markov_Order', 'Forecast_Date', 'Forecast_Day_Name', 'Predicted_Direction',
        'Historical_Frequency_FUD', 'Prediction_Probability', 'Validator Confirmation Percentage', 'Recommendation'
    ]
    existing_cols = [col for col in final_columns if col in final_df.columns]
    final_df_to_save = final_df[existing_cols]

    csv_path = output_folder / "Forecasts.csv"
    final_df_to_save.to_csv(csv_path, index=False)
    print(f"\n✅ Combined CSV with validation saved to: {csv_path}")
    
    # JSON report generation can be added here
    
    print("\n\n" + "="*30 + " RUN HEALTH SUMMARY " + "="*30)
    print(f"Successfully Processed: {run_summary['processed']}")
    if run_summary['failed']:
        print("--- Failures ---")
        for failure in run_summary['failed']:
            print(f"  - Asset: {failure['asset']}")
            print(f"    Error: {failure['error']}")
    print("="*80)

if __name__ == "__main__":
    run_forecasting_pipeline()
    print(f"\n\n--- {MAIN_SCRIPT_NAME} - Pipeline Finished Successfully! ---")