# data loading functions
import yfinance as yf
import pandas as pd
import numpy as np
import time
from config.config import CONFIG, START_DATE, END_DATE, stocks


# =========================
# DOWNLOAD FUNCTION (ROBUST)
# =========================
def download_price_data(retries=3, pause=1):

    all_data = []
    for ticker, name in stocks.items():
        print(f"📥 Downloading {ticker} --- {name}...")

        attempt = 0
        df = pd.DataFrame()
        # =========================
        # RETRY LOGIC (CRITICAL)
        # =========================
        while attempt < retries:
            try:
                df = yf.download(
                    ticker,
                    start=START_DATE,
                    end=END_DATE,
                    auto_adjust=True,
                    progress=False,
                    threads=False  # 🔥 safer for stability
                )

                if not df.empty:
                    break

            except Exception as e:
                print(f"⚠️ Error for {ticker}: {e}")

            attempt += 1
            time.sleep(pause)

        if df.empty:
            print(f"❌ Skipping {name} (no data)")
            continue

        # =========================
        # FIX MULTIINDEX
        # =========================
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # =========================
        # STANDARDIZE COLUMN NAMES
        # =========================
        df.columns = [str(col).capitalize() for col in df.columns]

        required_cols = ["Open", "High", "Low", "Close", "Volume"]

        for col in required_cols:
            if col not in df.columns:
                print(f"⚠️ Missing {col} in {name}, filling with NaN")
                df[col] = np.nan

        # =========================
        # ADD META
        # =========================
        df["Company"] = name
        df["Ticker"] = ticker

        # =========================
        # RESET INDEX
        # =========================
        df = df.reset_index()
        # =========================
        # CLEAN DATA
        # =========================
        df = df.drop_duplicates(subset=["Date"])
        df = df.sort_values("Date")

        # Remove extreme bad values
        df = df.replace([np.inf, -np.inf], np.nan)

        # Optional: forward fill small gaps
        # df = df.fillna(method="ffill").fillna(method="bfill") -- out dated
        df = df.ffill().bfill()
        all_data.append(df)

    # =========================
    # FINAL CONCAT
    # =========================
    if len(all_data) == 0:
        raise ValueError("❌ No data downloaded. Check tickers or internet.")

    final_df = pd.concat(all_data, ignore_index=True)

    # =========================
    # FINAL CLEANING
    # =========================
    final_df = final_df.drop_duplicates(subset=["Date", "Company"])
    final_df = final_df.sort_values(["Company", "Date"]).reset_index(drop=True)

    print("\n✅ Data download complete")
    print("Shape:", final_df.shape)

    return final_df



