# -*- coding: utf-8 -*-
# =====================================================================================
# # Feature Enrichment Engine - V1.0
#
# ## Purpose:
# This script enriches a clean time-series dataset by adding new, predictive
# features generated from a Markov Chain model. For each data point (day), it
# calculates what the Markov model would have predicted based on all prior data.
#
# ## Role in Pipeline:
# This is a critical data processing step that runs after the data has been
# cleaned (2_Data_Cleaning_Pipeline.py). The output of this script—the enriched
# file containing the base data plus the model's historical predictions—is the
# primary input for training the "Confidence Validator" model
# (in a subsequent script).
#
# ## Execution Order:
# 4th - This script should be run after the data is clean and before training
#       the confidence validator.
#
# ## Key Features:
# - Leak-Proof Design: Utilizes a walk-forward (expanding window) methodology to
#   generate features, ensuring that each prediction is made using only data
#   that was available at that point in time, preventing any data leakage.
# - Dynamic Feature Naming: Column names for new features are generated
#   dynamically based on the specified Markov Order (e.g., MarkovO4_Prob_UP).
# =====================================================================================

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import os
import time

# --- 1. User Settings & Configuration ---
SCRIPT_NAME = "Dynamic_Feature_Enrichment_Pipeline_Markov_Only"
print(f"--- Running: {SCRIPT_NAME} ---")

##-- Main Control: Asset Selection --##
# Change this variable only to run the script on any financial asset
ASSET_NAME = "USDJPY"
##-- End Control --##

# -- Path Settings --
BASE_DATA_PATH = Path("D:/Machine Learning/Data/Forex")

# -- Feature Settings --
MARKOV_ORDER = 4

# --- End of Settings ---

def add_markov_features(df_original, order):
    """
    # This function adds Markov features using a leak-proof method
    # (Rolling/Expanding Window Training). The model is trained and predicts
    # sequentially for each row.
    """
    print(f"Step 1: Adding Markov Features (Order={order}) - Leak-Proof Method...")

    df_markov_clean = df_original[df_original['Direction'].isin(['UP', 'DOWN'])].copy()
    df_markov_clean.reset_index(drop=True, inplace=True)
    
    pred_dir_col = f'MarkovO{order}_Pred_Direction'
    prob_up_col = f'MarkovO{order}_Prob_UP'
    prob_down_col = f'MarkovO{order}_Prob_DOWN'
    
    df_markov_clean[pred_dir_col] = pd.NA
    df_markov_clean[prob_up_col] = np.nan
    df_markov_clean[prob_down_col] = np.nan

    total_predictions_possible = len(df_markov_clean) - order
    if total_predictions_possible <= 0:
        print(f"  Not enough data ({len(df_markov_clean)} rows) to generate Markov features for order {order} (requires at least {order+1} rows). Skipping Markov features.")
        # Return the original dataframe with a standard set of columns
        return df_original[['Date', 'Close', 'Open', 'High', 'Low', 'Direction', 'Day_Name']].copy()

    start_time_markov = time.time()
    cases_with_pattern_found = 0 # Counter for cases where a known pattern was found

    for i in range(order, len(df_markov_clean)):
        current_training_data = df_markov_clean['Direction'].iloc[:i]

        transition_counts = defaultdict(Counter)
        if len(current_training_data) >= order:
            for j in range(order, len(current_training_data)):
                prev_sequence = tuple(current_training_data.iloc[j-order : j])
                current_state = current_training_data.iloc[j]
                transition_counts[prev_sequence][current_state] += 1

        transition_probabilities = defaultdict(lambda: defaultdict(float))
        for seq, counts in transition_counts.items():
            total = sum(counts.values())
            if total > 0:
                for state, count in counts.items():
                    transition_probabilities[seq][state] = count / total
        
        prev_sequence_for_prediction = tuple(df_markov_clean['Direction'].iloc[i-order : i])
        
        prediction = pd.NA
        prob_up = 0.5
        prob_down = 0.5

        if prev_sequence_for_prediction in transition_probabilities:
            probs = transition_probabilities[prev_sequence_for_prediction]
            prob_up = probs.get('UP', 0.0)
            prob_down = probs.get('DOWN', 0.0)
            
            if prob_up > prob_down:
                prediction = 'UP'
            elif prob_down > prob_up:
                prediction = 'DOWN'
            else:
                prediction = df_markov_clean['Direction'].iloc[i-1] # Fallback
            
            cases_with_pattern_found += 1
        else:
            prediction = df_markov_clean['Direction'].iloc[i-1] # Fallback
        
        df_markov_clean.loc[i, pred_dir_col] = prediction
        df_markov_clean.loc[i, prob_up_col] = prob_up
        df_markov_clean.loc[i, prob_down_col] = prob_down

        current_operation = i - order + 1
        # Update progress counter every 100 operations or at the very end
        if (current_operation % 100 == 0) or current_operation == total_predictions_possible:
            print(f"\r  Processing Markov features: {current_operation}/{total_predictions_possible} ({current_operation/total_predictions_possible:.1%})", end="", flush=True)

    end_time_markov = time.time()
    time_taken_markov = end_time_markov - start_time_markov
    print(f"\nStep 1: Done. Markov feature generation took {time_taken_markov:.2f} seconds.")
    print(f"  Total predictions attempted: {total_predictions_possible}")
    print(f"  Total cases with a specific pattern found (non-50/50 probabilities): {cases_with_pattern_found}")
    
    # Merge the new features back into the original dataframe
    df_final = df_original[['Date', 'Close', 'Open', 'High', 'Low', 'Direction', 'Day_Name']].copy()
    df_final = pd.merge(df_final, df_markov_clean[['Date', pred_dir_col, prob_up_col, prob_down_col]], on='Date', how='left')
    return df_final


def main():
    """
    # Main function that manages the full workflow
    """
    start_total_time = time.time()

    input_path = BASE_DATA_PATH / ASSET_NAME / f"{ASSET_NAME}.csv"
    output_folder = BASE_DATA_PATH / "Enriched_Data" / ASSET_NAME
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f"{ASSET_NAME}_Enriched.csv"

    try:
        print(f"Loading raw data from: {input_path}")
        df_raw = pd.read_csv(input_path)
        df_raw['Date'] = pd.to_datetime(df_raw['Date'])
        df_raw.sort_values(by='Date', inplace=True, ignore_index=True)
        df_raw['Day_Name'] = df_raw['Date'].dt.day_name()

        required_cols = ['Date', 'Close', 'Open', 'High', 'Low', 'Direction', 'Day_Name']
        if not all(col in df_raw.columns for col in required_cols):
            raise ValueError(f"Missing one or more required columns: {required_cols}. Found: {df_raw.columns.tolist()}")
        print(f"Successfully loaded {len(df_raw)} rows with essential columns.")
    except FileNotFoundError:
        print(f"FATAL ERROR: Input file not found at '{input_path}'. Please check ASSET_NAME and folder structure.")
        return
    except Exception as e:
        print(f"FATAL ERROR: Could not load or parse the input file. Error: {e}")
        return

    df_enriched = add_markov_features(df_raw, MARKOV_ORDER)

    try:
        print("\n--- Finalizing and Saving Enriched File ---")
        
        print("\nFinal file will contain the following columns:")
        print(df_enriched.columns.tolist())
        
        df_enriched.to_csv(output_path, index=False, float_format='%.4f')
        print(f"\n✅ Successfully created enriched data file at: {output_path}")
        
    except Exception as e:
        print(f"FATAL ERROR: Could not save the final enriched file. Error: {e}")

    end_total_time = time.time()
    total_time_taken = end_total_time - start_total_time
    print(f"\n--- {SCRIPT_NAME} Finished in {total_time_taken:.2f} seconds ---")

if __name__ == "__main__":
    main()