# -*- coding: utf-8 -*-
# ==============================================================================
# # Markov Order Optimizer (Backtester) - V1.0
#
# ## Purpose:
# This script performs a systematic, walk-forward backtest to find the optimal
# Markov Order for a given financial asset. It iterates through a specified
# range of orders, simulating predictions over a defined historical period
# and generating detailed performance reports for each order.
#
# ## Role in Pipeline:
# This is a critical research and optimization tool. It is used after the data
# has been cleaned (by 2_Data_Cleaning_Pipeline.py) to answer the question:
# "What is the most predictive 'memory length' (Markov Order) for this asset?"
# The results from this script inform which model should be deployed for live
# forecasting.
#
# ## Execution Order:
# 3rd - Run this script after cleaning the data for a specific asset.
#
# ## Key Features:
# - Walk-Forward Backtesting: Simulates realistic trading conditions where the
#   model only uses data available up to the point of prediction.
# - Parameter Sweeping: Automatically tests a range of Markov Orders.
# - Detailed Reporting: For each order, it generates a comprehensive report
#   with accuracy, confusion matrix, precision, recall, and F1-score.
# - Summary Comparison: Creates a final summary report comparing the
#   performance of all tested orders to easily identify the winner.
# ==============================================================================

import pandas as pd
import numpy as np
import os
from collections import Counter, defaultdict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# --- 1. Core Settings ---
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1200)

# --- 2. User Control Panel ---
# --- Specify the asset and parameters for the backtest ---

ASSET_NAME = "AUDUSD"

# Define the range of Markov Orders to search
ORDER_START = 1
ORDER_END = 10

# Define the backtesting period
BACKTEST_START_DATE_STR = "2025-01-01"
BACKTEST_END_DATE_STR = "2025-12-31"
# --- End of Control Panel ---

SCRIPT_NAME = f"{ASSET_NAME}_Markov_Order_Optimizer"
print(f"--- Running: {SCRIPT_NAME} ---")
print(f"Searching for the best Markov Order for {ASSET_NAME}")
print("="*80)


# --- 3. Path Configuration ---
BASE_DATA_PATH = r"D:\Machine Learning\Data\Forex"
BASE_PROJECT_PATH = r"D:\Machine Learning\Projects\Markov Forecaster"
INPUT_DATA_PATH = os.path.join(BASE_DATA_PATH, ASSET_NAME, f"{ASSET_NAME}.csv")
SEARCH_BASE_FOLDER = os.path.join(BASE_PROJECT_PATH, "Backtest", ASSET_NAME, f"Order_Search_{ORDER_START}-{ORDER_END}_{BACKTEST_START_DATE_STR}_to_{BACKTEST_END_DATE_STR}")
os.makedirs(SEARCH_BASE_FOLDER, exist_ok=True)

print(f"Input data file path: {INPUT_DATA_PATH}")
print(f"Output reports directory: {SEARCH_BASE_FOLDER}")


# --- 4. Data Loading and Preparation ---
print("\n--- Loading and Preparing Data ---")
try:
    df_full = pd.read_csv(INPUT_DATA_PATH)
    df_full['Date'] = pd.to_datetime(df_full['Date'], errors='coerce')
    df_full.dropna(subset=['Date'], inplace=True)
    df_full.sort_values(by='Date', inplace=True, ignore_index=True)
    df_analysis_data = df_full[df_full['Direction'].isin(['UP', 'DOWN'])].copy()
    print(f"Data loaded and prepared: {len(df_analysis_data)} UP/DOWN rows available.")
except FileNotFoundError:
    print(f"FATAL ERROR: Input file not found at {INPUT_DATA_PATH}. Please check ASSET_NAME and folder structure.")
    exit()
except Exception as e:
    print(f"FATAL ERROR loading data: {e}")
    exit()

# --- 5. Helper Functions ---
def build_markov_transition_matrix(data_frame, markov_order):
    transition_counts = defaultdict(Counter)
    transition_probabilities = defaultdict(lambda: defaultdict(float))
    if len(data_frame) < markov_order:
        return transition_counts, transition_probabilities
    for i in range(markov_order, len(data_frame)):
        previous_sequence = tuple(data_frame['Direction'].iloc[i-markov_order : i])
        current_direction = data_frame['Direction'].iloc[i]
        transition_counts[previous_sequence][current_direction] += 1
    for prev_seq, counts in transition_counts.items():
        total = sum(counts.values())
        if total > 0:
            for state, count in counts.items():
                transition_probabilities[prev_seq][state] = count / total
    return transition_counts, transition_probabilities

def predict_next_direction(df_data, transition_probs, markov_order):
    last_known_sequence = tuple(df_data['Direction'].iloc[-markov_order:])
    if last_known_sequence in transition_probs:
        probs = transition_probs[last_known_sequence]
        prob_up = probs.get('UP', 0.0)
        prob_down = probs.get('DOWN', 0.0)
        if prob_up > prob_down:
            return 'UP'
        elif prob_down > prob_up:
            return 'DOWN'
    # Fallback to the last day's direction in the sequence if tie or sequence not found
    return df_data['Direction'].iloc[-1] 

def generate_and_save_report(df_results, order, output_path, train_period_info, configured_test_period):
    y_true = df_results['Actual_Direction']
    y_pred = df_results['Predicted_Direction']
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=['UP', 'DOWN'])
    report_dict = classification_report(y_true, y_pred, labels=['UP', 'DOWN'], zero_division=0, output_dict=True)

    summary_text = [f"PERFORMANCE SUMMARY FOR MARKOV ORDER = {order}", "="*80]
    summary_text.append(f"Configured Test Period: {configured_test_period}")
    summary_text.append(f"Actual Test Period (Data Availability): {pd.to_datetime(df_results['Actual_Date']).min():%Y-%m-%d} to {pd.to_datetime(df_results['Actual_Date']).max():%Y-%m-%d}")
    summary_text.append(f"Training Period: {train_period_info}")
    summary_text.append(f"Total Predictions: {len(df_results)}")
    summary_text.append("="*80)
    summary_text.append("I. PREDICTION ACCURACY:")
    summary_text.append(f"   Overall Accuracy: {accuracy:.2%}")
    summary_text.append("\nII. CONFUSION MATRIX:")
    summary_text.append("           Predicted UP    Predicted DOWN")
    summary_text.append(f"   Actual UP        {cm[0, 0]:<14d} {cm[0, 1]:<14d}")
    summary_text.append(f"   Actual DOWN      {cm[1, 0]:<14d} {cm[1, 1]:<14d}")
    summary_text.append("\nIII. CLASSIFICATION REPORT:")
    summary_text.append("   Class          Precision       Recall          F1-Score        Support")
    for label in ['UP', 'DOWN']:
        metrics = report_dict[label]
        summary_text.append(f"   {label:<12s} {metrics['precision']:<15.2%} {metrics['recall']:<15.2%} {metrics['f1-score']:<15.2%} {int(metrics['support']):<15d}")
    
    final_report_str = "\n".join(summary_text)
    print("\n\n" + final_report_str)
    
    summary_filename = f"{ASSET_NAME}_Order_{order}_Report.txt"
    with open(os.path.join(output_path, summary_filename), 'w', encoding='utf-8') as f:
        f.write(final_report_str)
    print(f"\nIndividual report for Order {order} saved.")
    
    return {'Order': order, 'Accuracy': accuracy, 'F1_Score_UP': report_dict['UP']['f1-score'], 'F1_Score_DOWN': report_dict['DOWN']['f1-score']}


# --- 6. Main Workflow ---
def main():
    backtest_start = pd.to_datetime(BACKTEST_START_DATE_STR)
    backtest_end = pd.to_datetime(BACKTEST_END_DATE_STR)
    
    configured_test_period = f"{BACKTEST_START_DATE_STR} to {BACKTEST_END_DATE_STR}"

    # Split data into initial training and testing sets
    initial_training_data = df_analysis_data[df_analysis_data['Date'] < backtest_start].copy()
    test_data = df_analysis_data[(df_analysis_data['Date'] >= backtest_start) & (df_analysis_data['Date'] <= backtest_end)]
    
    if initial_training_data.empty or test_data.empty:
        print("ERROR: Training or Testing period is empty based on the specified dates. Please check.")
        return
    
    train_period_info = f"{initial_training_data['Date'].min():%Y-%m-%d} to {initial_training_data['Date'].max():%Y-%m-%d} ({len(initial_training_data)} days)"
    
    all_orders_summary = []

    for order_to_test in range(ORDER_START, ORDER_END + 1):
        try:
            order_output_path = os.path.join(SEARCH_BASE_FOLDER, f"Order_{order_to_test}")
            os.makedirs(order_output_path, exist_ok=True)
            
            print("\n" + "#"*80)
            print(f"# TESTING MARKOV ORDER = {order_to_test}")
            print("#"*80)

            if len(initial_training_data) < order_to_test:
                print(f"Skipping Order {order_to_test}: Not enough initial historical data ({len(initial_training_data)} days).")
                continue

            backtest_results = []
            # The training data grows with each step in the test set (Walk-Forward)
            temp_cumulative_data = initial_training_data.copy()
            
            for idx, current_day in test_data.iterrows():
                current_date = current_day['Date']
                print(f"\r   Processing Day {test_data.index.get_loc(idx) + 1}/{len(test_data)}: {current_date:%Y-%m-%d}", end="", flush=True)
                
                _, current_trans_probs = build_markov_transition_matrix(temp_cumulative_data, order_to_test)
                predicted_dir = predict_next_direction(temp_cumulative_data, current_trans_probs, order_to_test)
                
                backtest_results.append({
                    'Actual_Date': current_date.strftime('%Y-%m-%d'), # Corrected date format
                    'Actual_Direction': current_day['Direction'],
                    'Predicted_Direction': predicted_dir,
                })
                
                # Add the current day's actual data to the history for the next prediction
                temp_cumulative_data = pd.concat([temp_cumulative_data, pd.DataFrame([current_day])], ignore_index=True)

            print("\nBacktest loop finished for this order.")

            if backtest_results:
                df_backtest_results = pd.DataFrame(backtest_results)
                summary_for_order = generate_and_save_report(df_backtest_results, order_to_test, order_output_path, train_period_info, configured_test_period)
                all_orders_summary.append(summary_for_order)

        except Exception as e:
            print(f"\nERROR processing Order {order_to_test}: {e}"); continue
            
    if not all_orders_summary:
        print("\nNo backtests were completed."); return
        
    print("\n\n" + "="*80)
    print("--- OVERALL SUMMARY REPORT: COMPARING ALL ORDERS ---")
    print("="*80)
    df_summary = pd.DataFrame(all_orders_summary)
    df_summary.sort_values(by='Accuracy', ascending=False, inplace=True)
    
    # Format percentage columns for display
    for col in df_summary.columns:
        if col != 'Order':
            df_summary[col] = df_summary[col].apply(lambda x: f"{x:.2%}")
            
    print(df_summary.to_string(index=False))
    
    overall_summary_path = os.path.join(SEARCH_BASE_FOLDER, "Overall_Order_Comparison_Report.txt")
    with open(overall_summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Overall Performance Summary - Asset: {ASSET_NAME}\n")
        f.write(f"Test Period: {configured_test_period}\n")
        f.write("="*100 + "\n")
        f.write(df_summary.to_string(index=False))
    print(f"\nOverall summary report saved to: {overall_summary_path}")


if __name__ == "__main__":
    main()
    print(f"\n{SCRIPT_NAME} - Script Finished Successfully!")