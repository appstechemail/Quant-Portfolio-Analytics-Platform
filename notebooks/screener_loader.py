# ==========================================================
# FILE: src/fundamentals/screener_loader.py
# ==========================================================
# PURPOSE:
# Download quarterly + annual + ratios data
# from Screener.in for multiple companies
#
# OUTPUT:
# data/fundamentals/
# ├── quarterly/
# ├── annual/
# ├── ratios/
# └── merged/
#
# ==========================================================

# ==========================================================
# INSTALL REQUIREMENTS
# ==========================================================
# pip install selenium pandas webdriver-manager

# ==========================================================
# IMPORTS
# ==========================================================
import os
import time
import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service

from webdriver_manager.chrome import ChromeDriverManager


# ==========================================================
# CONFIG
# ==========================================================
BASE_PATH = "data/fundamentals"

os.makedirs(f"{BASE_PATH}/quarterly", exist_ok=True)
os.makedirs(f"{BASE_PATH}/annual", exist_ok=True)
os.makedirs(f"{BASE_PATH}/ratios", exist_ok=True)
os.makedirs(f"{BASE_PATH}/merged", exist_ok=True)


# ==========================================================
# COMPANIES
# ==========================================================
COMPANIES = {
    "HDBFS": {
        "screener": "HDB-Financial-Services",
        "name": "HDB Financial Services"
    },

    "PFC": {
        "screener": "PFC",
        "name": "Power Finance Corporation"
    },

    "TCS": {
        "screener": "TCS",
        "name": "Tata Consultancy Services"
    },

    "CRAMC": {
        "screener": "CRAMC",
        "name": "Canara Robeco Asset"
    },

    "SUZLON": {
        "screener": "SUZLON",
        "name": "Suzlon Energy"
    }
}


# ==========================================================
# CHROME DRIVER
# ==========================================================
def create_driver():

    options = webdriver.ChromeOptions()

    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

    return driver


# ==========================================================
# CLEAN COLUMN NAMES
# ==========================================================
def clean_columns(df):

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(" ", "_", regex=False)
        .str.replace("%", "_PERCENT", regex=False)
        .str.replace("+", "", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )

    return df


# ==========================================================
# CLEAN NUMERIC VALUES
# ==========================================================
def clean_numeric(df):

    for col in df.columns:

        if col in ["YEAR", "QUARTER", "TICKER", "COMPANY"]:
            continue

        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.strip()
        )

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    return df


# ==========================================================
# GENERIC TABLE SCRAPER
# ==========================================================
def scrape_table(driver, section_id):

    try:

        section = driver.find_element(
            By.ID,
            section_id
        )

        table = section.find_element(
            By.TAG_NAME,
            "table"
        )

        rows = table.find_elements(
            By.TAG_NAME,
            "tr"
        )

        raw_data = []

        for row in rows:

            cols = row.find_elements(
                By.TAG_NAME,
                "td"
            )

            row_data = [
                c.text.strip()
                for c in cols
            ]

            if len(row_data) > 0:
                raw_data.append(row_data)

        if len(raw_data) == 0:
            return pd.DataFrame()

        # --------------------------------------
        # TRANSPOSE STRUCTURE
        # --------------------------------------
        metrics = [r[0] for r in raw_data]

        values = [r[1:] for r in raw_data]

        periods = values[0]

        records = []

        for i, period in enumerate(periods):

            row_dict = {}

            row_dict["PERIOD"] = period

            for j, metric in enumerate(metrics):

                try:
                    row_dict[metric] = values[j][i]
                except:
                    row_dict[metric] = None

            records.append(row_dict)

        df = pd.DataFrame(records)

        df = clean_columns(df)

        df = clean_numeric(df)

        return df

    except Exception as e:

        print(f"❌ Section {section_id} failed: {e}")

        return pd.DataFrame()


# ==========================================================
# FETCH QUARTERLY
# ==========================================================
def fetch_quarterly_profit_loss(driver):

    return scrape_table(
        driver,
        "quarters"
    )


# ==========================================================
# FETCH ANNUAL
# ==========================================================
def fetch_annual_profit_loss(driver):

    return scrape_table(
        driver,
        "profit-loss"
    )


# ==========================================================
# FETCH RATIOS
# ==========================================================
def fetch_ratios(driver):

    return scrape_table(
        driver,
        "ratios"
    )


# ==========================================================
# MAIN PIPELINE
# ==========================================================
def run_pipeline():

    all_quarterly = []
    all_annual = []
    all_ratios = []

    driver = create_driver()

    for ticker, info in COMPANIES.items():

        screener_code = info["screener"]
        company_name = info["name"]

        print(f"\n📊 Processing {ticker}")

        try:

            # ==================================================
            # OPEN URL
            # ==================================================
            url = (
                f"https://www.screener.in/company/"
                f"{screener_code}/consolidated/"
            )

            driver.get(url)

            time.sleep(3)

            # ==================================================
            # QUARTERLY
            # ==================================================
            q_df = fetch_quarterly_profit_loss(driver)

            if not q_df.empty:

                q_df["TICKER"] = ticker
                q_df["COMPANY"] = company_name

                q_path = (
                    f"{BASE_PATH}/quarterly/"
                    f"{ticker}_quarterly.csv"
                )

                q_df.to_csv(
                    q_path,
                    index=False
                )

                all_quarterly.append(q_df)

                print("✅ Quarterly Saved")

            # ==================================================
            # ANNUAL
            # ==================================================
            a_df = fetch_annual_profit_loss(driver)

            if not a_df.empty:

                a_df["TICKER"] = ticker
                a_df["COMPANY"] = company_name

                a_path = (
                    f"{BASE_PATH}/annual/"
                    f"{ticker}_annual.csv"
                )

                a_df.to_csv(
                    a_path,
                    index=False
                )

                all_annual.append(a_df)

                print("✅ Annual Saved")

            # ==================================================
            # RATIOS
            # ==================================================
            r_df = fetch_ratios(driver)

            if not r_df.empty:

                r_df["TICKER"] = ticker
                r_df["COMPANY"] = company_name

                r_path = (
                    f"{BASE_PATH}/ratios/"
                    f"{ticker}_ratios.csv"
                )

                r_df.to_csv(
                    r_path,
                    index=False
                )

                all_ratios.append(r_df)

                print("✅ Ratios Saved")

        except Exception as e:

            print(f"❌ Failed {ticker}: {e}")

    driver.quit()

    # ======================================================
    # MERGED FILES
    # ======================================================
    if len(all_quarterly) > 0:

        merged_q = pd.concat(
            all_quarterly,
            ignore_index=True
        )

        merged_q.to_csv(
            f"{BASE_PATH}/merged/all_quarterly.csv",
            index=False
        )

    if len(all_annual) > 0:

        merged_a = pd.concat(
            all_annual,
            ignore_index=True
        )

        merged_a.to_csv(
            f"{BASE_PATH}/merged/all_annual.csv",
            index=False
        )

    if len(all_ratios) > 0:

        merged_r = pd.concat(
            all_ratios,
            ignore_index=True
        )

        merged_r.to_csv(
            f"{BASE_PATH}/merged/all_ratios.csv",
            index=False
        )

    print("\n✅ ALL FILES SAVED SUCCESSFULLY")


# ==========================================================
# ENTRY
# ==========================================================
if __name__ == "__main__":

    run_pipeline()
