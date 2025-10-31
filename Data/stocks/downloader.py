import yfinance as yf
import pandas as pd
import os
import time
from tqdm import tqdm

DATA_DIR = "data"
TICKERS_FILE = "tickers.txt"
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"
RETRIES = 5
WAIT = 10  # seconds between retries

os.makedirs(DATA_DIR, exist_ok=True)

def load_tickers(file_path):
    with open(file_path, "r") as f:
        return [line.strip().upper() for line in f if line.strip()]

def download_and_save(ticker):
    file_path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    if os.path.exists(file_path):
        print(f"{ticker} already downloaded. Skipping.")
        return

    for attempt in range(RETRIES):
        try:
            print(f"Downloading {ticker} (attempt {attempt+1})...")
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if not df.empty:
                df.reset_index(inplace=True)
                df.to_parquet(file_path, index=False)
                print(f"Saved {ticker} to {file_path}")
                return
            else:
                print(f"No data for {ticker}")
                return
        except Exception as e:
            print(f"Error downloading {ticker}: {e}. Retrying in {WAIT} seconds...")
            time.sleep(WAIT)

    print(f"Failed to download {ticker} after {RETRIES} attempts.")

def main():
    tickers = load_tickers(TICKERS_FILE)
    for ticker in tqdm(tickers):
        download_and_save(ticker)

if __name__ == "__main__":
    main()


