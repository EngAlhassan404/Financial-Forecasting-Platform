# -*- coding: utf-8 -*-
# =====================================================================================
# # Unified Performance Analyzer V5.2
#
# ## Purpose:
# This script serves as the final analysis and reporting platform for the entire
# forecasting system. It reads generated forecast logs and actual historical
# data to produce comprehensive performance reports in multiple formats (HTML, JSON).
#
# ## Role in Pipeline:
# This is the capstone script in the workflow. It is used to evaluate the
# performance of any model (live, historical, H4, or daily) whose forecasts
# have been logged. It provides the ultimate "report card" for the system.
#
# ## Execution Order:
# 8th (Final) - Run this script whenever a performance evaluation is needed.
#
# ## Key Features:
# - Dual-Mode Analysis: Operates in "GENERAL" mode for standard periodic reports
#   or "CUSTOM_RANGE" mode for deep-dive analysis of specific timeframes.
# - Dynamic Pathing: Automatically locates log files for standard assets and
#   special sub-models (e.g., USDJPY_O4).
# - Validator Performance Analysis: Includes a dedicated module to assess the
#   effectiveness of the confidence validator's recommendations.
# - Multi-Format Reporting: Generates user-friendly visual HTML reports and
#   structured JSON data files for further machine processing.
# =====================================================================================

import pandas as pd
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64
import json

# --- 1. Core Settings ---
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1200)

SCRIPT_NAME = "Performance_Analyzer"
SCRIPT_VERSION = "v5.2_Final_Enhancements"
print(f"--- Running: {SCRIPT_NAME} - {SCRIPT_VERSION} ---")
print("="*80)

# --- 2. User Control Panel ---

# Select Mode: "GENERAL" (for standard periods) or "CUSTOM_RANGE" (for a specific range)
ANALYSIS_TYPE = "GENERAL"

# Custom range settings (only used if ANALYSIS_TYPE is "CUSTOM_RANGE")
CUSTOM_START_DATE_STR = "06-01-2025"  # Use "MM-DD-YYYY" format
CUSTOM_END_DATE_STR = "06-30-2025"    # Use "MM-DD-YYYY" format

# Unified list of assets to be analyzed
ASSETS_CONFIG = [
    {"asset_name": "USDJPY_O4", "markov_order": 4, "input_file_name": "USDJPY.csv", "market_type": "Forex"},
    {"asset_name": "Gold", "markov_order": 5, "input_file_name": "Gold.csv", "market_type": "Forex"},
    {"asset_name": "WTI", "markov_order": 5, "input_file_name": "WTI.csv", "market_type": "Forex"},
    {"asset_name": "Silver", "markov_order": 1, "input_file_name": "Silver.csv", "market_type": "Forex"},
    {"asset_name": "DXY", "markov_order": 1, "input_file_name": "DXY.csv", "market_type": "Forex"},
]
RECENT_TRADES_TO_SHOW = 100
# --- End of Control Panel ---


# --- 3. Path Configuration ---
BASE_PROJECT_PATH = r"D:\Machine Learning\Projects\Markov Forecaster"
MAIN_DATA_FOLDER = r"D:\Machine Learning\Data" 
REPORTS_OUTPUT_FOLDER = os.path.join(BASE_PROJECT_PATH, "Performance Reports")
os.makedirs(REPORTS_OUTPUT_FOLDER, exist_ok=True)

# --- 4. Helper Functions ---
def get_performance_stats(df, period_label):
    if df.empty: return {'Period': period_label, 'Total': 0, 'Correct': 0, 'Accuracy': 0.0}
    total, correct = len(df), df['Correct'].sum()
    accuracy = correct / total if total > 0 else 0.0
    return {'Period': period_label, 'Total': int(total), 'Correct': int(correct), 'Accuracy': accuracy}

def analyze_validator_performance(df):
    """Analyzes the performance of the validator's recommendations."""
    if 'Recommendation' not in df.columns:
        return pd.DataFrame()
        
    df_validated = df.dropna(subset=['Recommendation'])
    df_validated = df_validated[df_validated['Recommendation'] != '']
    if df_validated.empty:
        return pd.DataFrame()

    summary = df_validated.groupby('Recommendation')['Correct'].value_counts().unstack(fill_value=0)
    summary.rename(columns={True: 'Correct Cases', False: 'Incorrect Cases'}, inplace=True)
    
    if 'Correct Cases' not in summary.columns: summary['Correct Cases'] = 0
    if 'Incorrect Cases' not in summary.columns: summary['Incorrect Cases'] = 0
    
    summary['Total Cases'] = summary['Correct Cases'] + summary['Incorrect Cases']
    
    order = ['Excellent', 'Good', 'Weak']
    summary = summary.reindex(order).dropna(how='all').fillna(0).astype(int)
    
    return summary.reset_index()[['Recommendation', 'Total Cases', 'Correct Cases', 'Incorrect Cases']]

def analyze_asset_performance(asset_config, date_filter=None):
    asset_name = asset_config["asset_name"]
    markov_order = asset_config["markov_order"]
    print(f"\n--- Analyzing Performance for: {asset_name} (Order: {markov_order}) ---")
    
    model_name = asset_config.get("output_name", asset_name)
    forecasts_log_path = os.path.join(BASE_PROJECT_PATH, model_name, f"{model_name}_Order{markov_order}_Daily_Forecasts", f"{model_name}_MarkovO{markov_order}_DailyForecasts.csv")
    base_asset_name = asset_name.split('_')[0]
    actual_data_path = os.path.join(MAIN_DATA_FOLDER, "FOREX", base_asset_name, f"{base_asset_name}.csv")
    
    try:
        df_preds = pd.read_csv(forecasts_log_path, parse_dates=['Forecast_Date'])
        df_actuals = pd.read_csv(actual_data_path, parse_dates=['Date'])
    except FileNotFoundError as e:
        print(f"    ERROR: Could not find a required file. Skipping. Details: {e}"); return None
    
    df_preds.rename(columns={'Forecast_Date': 'Date'}, inplace=True)
    df_actuals.rename(columns={'Direction': 'Actual_Direction'}, inplace=True)
    df_merged = pd.merge(df_preds, df_actuals[['Date', 'Actual_Direction']], on='Date', how='inner')
    
    if df_merged.empty:
        print(f"    INFO: No matching dates found between forecasts and actuals for {asset_name}. Skipping."); return None
        
    df_merged['Correct'] = df_merged['Predicted_Direction'] == df_merged['Actual_Direction']
    df_merged.sort_values(by='Date', ascending=False, inplace=True)
    
    if date_filter:
        df_merged = df_merged[(df_merged['Date'] >= date_filter['start']) & (df_merged['Date'] <= date_filter['end'])].copy()
        if df_merged.empty:
            print(f"    INFO: No data found within the CUSTOM date range for {asset_name}. Skipping."); return None
    
    # Define analysis periods
    periods = {'Overall': df_merged}
    if not date_filter: # Standard periodic analysis only runs in GENERAL mode
        today = pd.to_datetime(datetime.now().date())
        periods.update({
            'Last 7 Trading Days': df_merged[df_merged['Date'] >= today - pd.tseries.offsets.BDay(7)],
            'Last 30 Days': df_merged[df_merged['Date'] >= today - timedelta(days=30)],
        })
    
    stats_list = [get_performance_stats(df, label) for label, df in periods.items()]
    return {"asset_name": asset_name, "order": markov_order, "stats_df": pd.DataFrame(stats_list), "full_data": df_merged}

# --- Reporting Functions ---
def generate_html_report(all_assets_data, output_path, report_title):
    html_sections = ""
    for asset_data in all_assets_data:
        asset_name, order = asset_data["asset_name"], asset_data["order"]
        df_stats = asset_data["stats_df"]
        df_full = asset_data["full_data"]
        
        validator_perf_df = analyze_validator_performance(df_full)
        validator_html = validator_perf_df.to_html(index=False, border=1, classes='dataframe', justify='center') if not validator_perf_df.empty else "<p>Validation is not enabled for this asset.</p>"
        stats_html = df_stats.to_html(index=False, border=1, classes='dataframe', justify='center', na_rep='N/A', float_format='{:.2%}'.format)
        
        report_cols = ['Date', 'Predicted_Direction', 'Actual_Direction', 'Recommendation', 'Validator Confirmation Percentage', 'Correct']
        existing_report_cols = [col for col in report_cols if col in df_full.columns]
        format_dict = {'Date': '{:%Y-%m-%d}', 'Validator Confirmation Percentage': '{:.2%}'}
        
        recent_html = df_full[existing_report_cols].head(RECENT_TRADES_TO_SHOW).style\
            .apply(lambda s: ['background-color: #d4edda' if v else 'background-color: #f8d7da' for v in s], subset=['Correct'])\
            .format(format_dict).hide(axis="index").to_html(classes='dataframe recent-table', na_rep='N/A')

        html_sections += f"""
        <div class="asset-section">
            <h2 class="forex-header">Performance: {asset_name} (Order: {order})</h2>
            <div class="container-grid">
                <div class="stats-container"><h3>Performance Summary</h3>{stats_html}</div>
                <div class="stats-container"><h3>Validator Performance</h3>{validator_html}</div>
            </div>
            <div class="recent-trades-container"><h3>Recent Forecasts vs Actuals</h3>{recent_html}</div>
        </div>
        """

    full_html = f"""
    <!DOCTYPE html><html><head><meta charset="UTF-8"><title>{report_title}</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; margin: 20px; background-color: #f9f9f9; }}
        h1, h2, h3 {{ text-align: center; color: #2c3e50; }} h1 {{ border-bottom: 3px solid #3498db; padding-bottom: 15px; }}
        .asset-section {{ border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin-bottom: 30px; background-color: #fff;}}
        .container-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 20px; align-items: flex-start; }}
        .stats-container, .recent-trades-container {{ width: 100%; margin-top: 20px; }}
        table.dataframe {{ border-collapse: collapse; margin: 15px 0; font-size: 0.9em; width: 100%; }}
        table.dataframe th, td {{ padding: 8px 12px; border: 1px solid #ddd; text-align: center;}}
        table.dataframe thead tr {{ background-color: #3498db; color: #ffffff; }}
    </style></head><body>
    <h1>{report_title}</h1>{html_sections}
    <p>Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </body></html>
    """
    with open(output_path, 'w', encoding='utf-8') as f: f.write(full_html)
    print(f"\n✅ Visual HTML report saved to: {output_path}")

def generate_json_report(all_assets_data, output_path, report_title):
    print(f"--- Generating JSON Report ---")
    asset_reports = []
    for asset_data in all_assets_data:
        df_full_copy = asset_data["full_data"].copy()
        for col in df_full_copy.columns:
            if pd.api.types.is_datetime64_any_dtype(df_full_copy[col]):
                df_full_copy[col] = df_full_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        asset_report = {
            "asset_name": asset_data["asset_name"],
            "order": asset_data["order"],
            "performance_summary": asset_data["stats_df"].to_dict(orient='records'),
            "validator_performance": analyze_validator_performance(asset_data["full_data"]).to_dict(orient='records'),
            "forecasts_for_period": df_full_copy.to_dict(orient='records')
        }
        asset_reports.append(asset_report)

    final_json = {
        "report_metadata": {"report_title": report_title, "report_generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
        "asset_performance_details": asset_reports
    }
    with open(output_path, 'w', encoding='utf-8') as f: json.dump(final_json, f, indent=4, ensure_ascii=False)
    print(f"✅ JSON data report saved to: {output_path}")

# --- 5. Main Pipeline Runner ---
def run_analysis_pipeline():
    date_filter = None
    output_dir = REPORTS_OUTPUT_FOLDER
    report_title = f"General Performance Report - {datetime.now():%Y-%m-%d}"

    if ANALYSIS_TYPE == "CUSTOM_RANGE":
        print("\n--- Running in CUSTOM RANGE mode ---")
        try:
            start_date = pd.to_datetime(CUSTOM_START_DATE_STR, format='%m-%d-%Y')
            end_date = pd.to_datetime(CUSTOM_END_DATE_STR, format='%m-%d-%Y').replace(hour=23, minute=59, second=59)
            date_filter = {'start': start_date, 'end': end_date}
            folder_name = f"Analysis_{start_date.strftime('%d-%b')}_to_{end_date.strftime('%d-%b-%Y')}"
            output_dir = os.path.join(REPORTS_OUTPUT_FOLDER, folder_name)
            os.makedirs(output_dir, exist_ok=True)
            report_title = f"Custom Period Analysis: {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}"
            print(f"Analyzing data from {start_date.date()} to {end_date.date()}")
            print(f"Custom reports will be saved in: {output_dir}")
        except ValueError:
            print(f"CRITICAL ERROR: Invalid date format in CUSTOM settings. Please use 'MM-DD-YYYY'.")
            return
    else:
        print("\n--- Running in GENERAL mode ---")
    
    all_results = [analyze_asset_performance(config, date_filter) for config in ASSETS_CONFIG]
    all_results = [res for res in all_results if res is not None]

    if not all_results:
        print("\nNo performance data could be generated for the selected criteria.")
        return
        
    report_name_base = "Performance_Report_Custom" if date_filter else f"Performance_Report_General"
    html_report_filepath = os.path.join(output_dir, f"{report_name_base}.html")
    json_report_filepath = os.path.join(output_dir, f"{report_name_base}.json")

    generate_html_report(all_results, html_report_filepath, report_title)
    generate_json_report(all_results, json_report_filepath, report_title)
    
if __name__ == "__main__":
    run_analysis_pipeline()
    print(f"\n{SCRIPT_NAME} - Analysis Finished!")