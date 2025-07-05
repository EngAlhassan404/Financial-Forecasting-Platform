# -*- coding: utf-8 -*-
# =====================================================================================
# # Data Acquisition Pipeline - V1.0
#
# ## Purpose:
# This script serves as the primary data ingestion engine for the entire forecasting
# system. Its core responsibility is to fetch the latest daily financial data
# from the Alpha Vantage API for a predefined list of Forex and Crypto assets.
#
# ## Role in Pipeline:
# This is the foundational first step in the workflow. It must be run to ensure
# all subsequent scripts (cleaning, analysis, forecasting) have the most
# recent data. It populates and maintains the raw data files in the
# `D:/Machine Learning/Data/` directory.
#
# ## Execution Order:
# 1st - This script should be executed before any other component in the system.
#
# ## Key Features:
# - Multi-Asset Support: Handles both Forex and Crypto assets from a central config.
# - Self-Healing Merge: Intelligently finds the last date in the local CSV and
#   appends only the new data, preventing duplicates and preserving history.
# - Automatic Backups: Creates timestamped backups of data files before updating.
# - API Rate Limiting: Includes a 20-second delay between API calls.
# - Resilient Fetching: Implements a retry mechanism for connection errors.
# =====================================================================================

import pandas as pd
import numpy as np
import requests
from pathlib import Path
import time
import shutil
import os

# --- 1. Centralized Settings & Run Control ---
SYSTEM_TYPE = "market"  # Options: "forex", "crypto", "market" (for both)

# --- Automatic Backup Settings ---
ENABLE_BACKUPS = True
MAX_BACKUPS_PER_ASSET = 7

# --- API and Path Configuration ---
ALPHA_VANTAGE_API_KEY = "YOUR_API_KEY_HERE"  # <-- PASTE YOUR API KEY HERE
BASE_DATA_FOLDER = Path(r"D:/Machine Learning/Data")
BACKUP_ROOT_FOLDER = BASE_DATA_FOLDER.parent / "_Backups"

# --- Asset Lists ---
FOREX_ASSETS = [
    {"user_name": "EURUSD", "market_type": "Forex", "av_from": "EUR", "av_to": "USD", "type": "FX"},
    {"user_name": "GBPUSD", "market_type": "Forex", "av_from": "GBP", "av_to": "USD", "type": "FX"},
    {"user_name": "NZDUSD", "market_type": "Forex", "av_from": "NZD", "av_to": "USD", "type": "FX"},
    {"user_name": "USDCAD", "market_type": "Forex", "av_from": "USD", "av_to": "CAD", "type": "FX"},
    {"user_name": "USDCHF", "market_type": "Forex", "av_from": "USD", "av_to": "CHF", "type": "FX"},
    {"user_name": "USDJPY", "market_type": "Forex", "av_from": "USD", "av_to": "JPY", "type": "FX"},
    {"user_name": "USDDKK", "market_type": "Forex", "av_from": "USD", "av_to": "DKK", "type": "FX"},
    {"user_name": "AUDUSD", "market_type": "Forex", "av_from": "AUD", "av_to": "USD", "type": "FX"},
]
CRYPTO_ASSETS = [
    {"user_name": "BTC", "market_type": "Crypto", "av_symbol": "BTC", "av_market": "USD", "type": "CRYPTO"},
    {"user_name": "ETH", "market_type": "Crypto", "av_symbol": "ETH", "av_market": "USD", "type": "CRYPTO"}
]

# ==============================================================================
# 2. Core Functions
# ==============================================================================

def manage_backups(source_file_path, asset_config):
    if not ENABLE_BACKUPS:
        return
    try:
        backup_dir = BACKUP_ROOT_FOLDER / asset_config['market_type'] / asset_config['user_name']
        backup_dir.mkdir(parents=True, exist_ok=True)
        if not source_file_path.exists():
            return
        
        # Manage backup rotation
        existing_backups = sorted(backup_dir.glob(f"{asset_config['user_name']}_backup_*.csv"), key=os.path.getmtime)
        while len(existing_backups) >= MAX_BACKUPS_PER_ASSET:
            oldest_backup = existing_backups.pop(0)
            oldest_backup.unlink()
            print(f"    -> 🗑️ Deleted oldest backup: {oldest_backup.name}")
            
        # Create new backup if one for today doesn't exist
        today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
        backup_file_path = backup_dir / f"{asset_config['user_name']}_backup_{today_str}.csv"
        if not backup_file_path.exists():
            shutil.copy2(source_file_path, backup_file_path)
            print(f"    -> 🛡️ Successfully created backup: {backup_file_path.name}")
    except Exception as e:
        print(f"    -> ❌ Error during backup process: {e}")

def fetch_and_update_asset(asset_config):
    asset_name = asset_config["user_name"]
    market_folder = asset_config["market_type"]
    print(f"\n--- 🔄 Processing: {asset_name} (Market: {market_folder}) ---")
    
    summary = {"asset": asset_name, "status": "❌ Failed", "latest_date": None, "details": ""}
    local_file_path = BASE_DATA_FOLDER / market_folder / asset_name / f"{asset_name}.csv"
    
    manage_backups(local_file_path, asset_config)
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            try:
                df_existing = pd.read_csv(local_file_path, parse_dates=['Date'])
                if attempt == 0:
                    print(f"    -> 📂 Found local file with {len(df_existing)} records.")
            except FileNotFoundError:
                df_existing = pd.DataFrame()
                if attempt == 0:
                    print("    -> 📄 No local file found, creating a new file.")

            if asset_config["type"] == "FX":
                url = (f'https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={asset_config["av_from"]}&to_symbol={asset_config["av_to"]}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}')
                data_key = 'Time Series FX (Daily)'
            elif asset_config["type"] == "CRYPTO":
                url = (f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={asset_config["av_symbol"]}&market={asset_config["av_market"]}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}')
                data_key = 'Time Series (Digital Currency Daily)'
            
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if data_key not in data:
                error_msg = data.get('Note') or data.get('Error Message', 'Unknown API error')
                print(f"    -> ❌ API Error for {asset_name}: {error_msg}")
                summary["details"] = error_msg
                return summary

            df_new = pd.DataFrame.from_dict(data[data_key], orient='index')
            df_new.index = pd.to_datetime(df_new.index)
            
            # Time Firewall: Only include data from before today (UTC)
            last_valid_date = (pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=1)).normalize().date()
            df_new = df_new[df_new.index.date <= last_valid_date]
            
            if asset_config["market_type"] == "Forex":
                df_new = df_new[df_new.index.dayofweek < 5] # Monday=0, Sunday=6

            if df_new.empty:
                print("    -> ℹ️ No new data matching the criteria to add.")
                summary["status"] = "ℹ️ No Change"
                if not df_existing.empty:
                    summary["latest_date"] = df_existing['Date'].max()
                return summary

            if asset_config["type"] == "CRYPTO":
                df_new.rename(columns=lambda x: x.split('. ')[1].split(' (')[0].capitalize(), inplace=True)
            else:
                df_new.rename(columns=lambda x: x.split('. ')[1].capitalize(), inplace=True)

            df_new = df_new[['Open', 'High', 'Low', 'Close']].astype(float)
            
            conditions = [(df_new['Close'] > df_new['Open']), (df_new['Close'] < df_new['Open'])]
            choices = ['UP', 'DOWN']
            df_new['Direction'] = np.select(conditions, choices, default='FLAT')
            
            df_new.reset_index(inplace=True)
            df_new.rename(columns={'index': 'Date'}, inplace=True)

            df_combined = pd.concat([df_existing, df_new], ignore_index=True) if not df_existing.empty else df_new
            
            df_combined.drop_duplicates(subset=['Date'], keep='last', inplace=True)
            df_combined.sort_values(by='Date', inplace=True, ignore_index=True)
            
            os.makedirs(local_file_path.parent, exist_ok=True)
            df_combined.to_csv(local_file_path, index=False)
            
            newly_added_count = len(df_combined) - len(df_existing)
            
            print(f"    -> ✅ Added {newly_added_count} new records. Historical data preserved.")
            
            summary["status"] = "✅ Succeeded"
            summary["latest_date"] = df_combined['Date'].max()
            return summary

        except requests.exceptions.ConnectionError as e:
            if "ConnectionResetError" in str(e) and attempt < max_retries - 1:
                print(f"    -> ⚠️ Connection failed (Attempt {attempt + 1}/{max_retries}). Retrying in 20 seconds...")
                time.sleep(20)
                continue
            else:
                print(f"    -> ❌ Unexpected connection error after multiple retries: {e}")
                summary["details"] = str(e)
                return summary
        except Exception as e:
            print(f"    -> ❌ Unexpected error while processing {asset_name}: {e}")
            summary["details"] = str(e)
            return summary
            
    return summary

def print_summary_table(summary_data):
    """Prints a formatted summary table of the run results."""
    print("\n\n" + "="*80)
    print("--- 🚀 FINAL RUN SUMMARY 🚀 ---")
    print("="*80)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df['latest_date'] = pd.to_datetime(summary_df['latest_date']).dt.strftime('%Y-%m-%d')
    summary_df.fillna('---', inplace=True)

    print(summary_df.to_string(index=False))
    print("="*80)

# ==============================================================================
# 3. Main Runner
# ==============================================================================
if __name__ == "__main__":
    
    assets_to_process = []
    if SYSTEM_TYPE == 'forex':
        assets_to_process = FOREX_ASSETS
    elif SYSTEM_TYPE == 'crypto':
        assets_to_process = CRYPTO_ASSETS
    elif SYSTEM_TYPE == 'market':
        assets_to_process = FOREX_ASSETS + CRYPTO_ASSETS
    else:
        print(f"\n[Error] Invalid SYSTEM_TYPE value: '{SYSTEM_TYPE}'. Please choose 'forex', 'crypto', or 'market'.")

    if assets_to_process:
        print(f"🚀 Starting Daily Data Ingestion Tool (Mode: {SYSTEM_TYPE.upper()}) 🚀")
        
        run_results = []
        for asset in assets_to_process:
            result = fetch_and_update_asset(asset)
            run_results.append(result)
            if asset != assets_to_process[-1]: # No need to wait after the last asset
                print("    -> ⏳ Waiting for 20 seconds to respect API rate limits...")
                time.sleep(20)
        
        print_summary_table(run_results)
        print("\n\n🎉 All specified assets have been updated successfully! 🎉")