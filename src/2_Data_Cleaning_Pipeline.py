# -*- coding: utf-8 -*-
# ==============================================================================
# # Automated Data Cleaning Pipeline V2.1
#
# ## Purpose:
# This script serves as the second stage in the data processing workflow. It is
# designed to take raw, potentially messy financial data files and transform
# them into a clean, standardized, and analysis-ready format.
#
# ## Role in Pipeline:
# Executed after the data acquisition step (1_Data_Acquisition_Pipeline.py),
# this pipeline is crucial for ensuring data quality and integrity before any
# backtesting or forecasting is performed. It prepares the foundational dataset
# that all subsequent models will use.
#
# ## Execution Order:
# 2nd - This script should be run after fetching the latest raw data.
#
# ## Key Features:
# - Configuration-Driven: Processes multiple assets defined in a central list.
# - Robust Validation: Handles invalid dates, duplicates, and non-numeric values.
# - Data Standardization: Renames columns to a standard format (e.g., 'Price' to 'Close').
# - Feature Engineering: Creates the essential 'Direction' column (UP/DOWN/FLAT).
# - Detailed Logging: Prints a step-by-step report of its actions for each asset.
# ==============================================================================

import pandas as pd
import os

# --- 1. Core Settings ---
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1200)

SCRIPT_NAME = "Automated_Data_Cleaning_Pipeline"
print(f"--- Running: {SCRIPT_NAME} V2.1 ---")
print("="*80)

# --- 2. Central Control Panel ---
# To add or remove an asset, simply edit this list.
ASSETS_CONFIG = [
    {"asset_name": "USDJPY", "input_file": "USDJPY.csv", "output_file": "USDJPY.csv"},
    {"asset_name": "Gold",   "input_file": "Gold.csv",   "output_file": "Gold.csv"},
    {"asset_name": "Silver", "input_file": "Silver.csv", "output_file": "Silver.csv"},
    {"asset_name": "WTI",    "input_file": "WTI.csv",    "output_file": "WTI.csv"},
    {"asset_name": "DXY",    "input_file": "DXY.csv",    "output_file": "DXY.csv"},
]

# --- 3. Base Path Configuration ---
# The base path containing asset folders (e.g., EURUSD, Gold)
BASE_DATA_PATH = r"D:\Machine Learning\Data\Forex" 

# --- 4. Cleaning & Validation Function (The Core Engine) ---
def clean_and_validate_data(df, asset_name_for_report):
    """
    Performs all cleaning and validation steps on a given DataFrame.
    Prints a report to the console and returns the clean DataFrame.
    """
    print(f"\n--- Starting Cleaning & Validation for: {asset_name_for_report} ---")
    print(f"Initial rows to process: {len(df)}")
    
    # 1. Process Date Column
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    initial_rows = len(df)
    df.dropna(subset=['Date'], inplace=True)
    if len(df) < initial_rows:
        print(f"  - Removed {initial_rows - len(df)} rows due to invalid date format.")

    # 2. Remove Duplicates
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    if len(df) < initial_rows:
        print(f"  - Removed {initial_rows - len(df)} fully duplicate rows.")
        
    initial_rows = len(df)
    df.drop_duplicates(subset=['Date'], keep='first', inplace=True)
    if len(df) < initial_rows:
        print(f"  - Removed {initial_rows - len(df)} rows with duplicate timestamps.")

    # 3. Clean Numeric Columns
    # Add 'Close' to the list to handle files that already have this column
    price_cols_to_check = ['Price', 'Open', 'High', 'Low', 'Close']
    
    existing_price_cols = [col for col in price_cols_to_check if col in df.columns]
    for col in existing_price_cols:
        # Convert to numeric, replacing commas and coercing errors to NaN
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

    # 4. Drop rows with invalid (NaN or zero) price values
    initial_rows = len(df)
    # Use the list of columns that actually exist and were converted
    df.dropna(subset=existing_price_cols, inplace=True)
    df = df[(df[existing_price_cols] != 0).all(axis=1)]
    if len(df) < initial_rows:
        print(f"  - Removed {initial_rows - len(df)} rows with invalid (NaN or zero) price values.")
    
    # 5. Standardize Column Names
    if 'Price' in df.columns:
        df = df.rename(columns={'Price': 'Close'})
    
    # 6. Create 'Direction' Column
    if 'Open' in df.columns and 'Close' in df.columns:
        df['Direction'] = "FLAT"
        df.loc[df['Close'] > df['Open'], 'Direction'] = 'UP'
        df.loc[df['Close'] < df['Open'], 'Direction'] = 'DOWN'
        print("  - 'Direction' column created successfully.")
    else:
        print("  - WARNING: 'Open' or 'Close' column missing, cannot create 'Direction' column.")

    # 7. Select and order final columns
    final_columns = [col for col in ['Date', 'Open', 'High', 'Low', 'Close', 'Direction'] if col in df.columns]
    df_final = df[final_columns]
    
    return df_final.sort_values(by='Date').reset_index(drop=True)

# --- 5. Main Pipeline Runner ---
def run_cleaning_pipeline():
    """
    Iterates through each asset in the config list and runs the full cleaning process.
    """
    for config in ASSETS_CONFIG:
        asset_name = config["asset_name"]
        input_file = config["input_file"]
        output_file = config["output_file"]
        
        # Build paths dynamically
        asset_path = os.path.join(BASE_DATA_PATH, asset_name)
        input_filepath = os.path.join(asset_path, input_file)
        output_filepath = os.path.join(asset_path, output_file)

        print("\n" + "#"*60)
        print(f"Processing Asset: {asset_name}")
        print("#"*60)
        print(f"  - Input file: {input_filepath}")

        # Load raw data
        try:
            df_raw = pd.read_csv(input_filepath)
        except FileNotFoundError:
            print(f"  - ❌ ERROR: Input file not found. Skipping asset.")
            continue
        except Exception as e:
            print(f"  - ❌ ERROR: Could not read file. Details: {e}. Skipping asset.")
            continue
            
        # Apply the cleaning and validation process
        df_clean = clean_and_validate_data(df_raw, asset_name)

        # Save the final clean file
        if df_clean is not None and not df_clean.empty:
            try:
                # Ensure the output directory exists
                os.makedirs(asset_path, exist_ok=True)
                df_clean.to_csv(output_filepath, index=False)
                print("\n--- ✅ SUCCESS! ---")
                print(f"Clean file for {asset_name} saved successfully.")
                print(f"  - Final row count: {len(df_clean)}")
                print(f"  - Date range: {df_clean['Date'].min().date()} to {df_clean['Date'].max().date()}")
                print(f"  - Output location: {output_filepath}")
            except Exception as e:
                print(f"  - ❌ ERROR: Could not save the clean file. Details: {e}")
        else:
            print("  - ❌ ERROR: Processing resulted in an empty dataset. No file was saved.")
            
if __name__ == "__main__":
    run_cleaning_pipeline()
    print("\n\n" + "="*80)
    print("Data Cleaning Pipeline Finished.")
    print("="*80)