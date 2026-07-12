import urllib.request
import os

urls = [
    "https://raw.githubusercontent.com/aditya-yadav-01/Sales-Forecasting/master/train.csv",
    "https://raw.githubusercontent.com/ajaygande/Sales-Forecasting-Using-Linear-Regression-Superstore-Dataset/master/train.csv",
    "https://raw.githubusercontent.com/ajaygande/Sales-Forecasting-Using-Linear-Regression-Superstore-Dataset/main/train.csv",
    "https://raw.githubusercontent.com/carlosalano/superstore-sales-forecasting/master/train.csv",
    "https://raw.githubusercontent.com/carlosalano/superstore-sales-forecasting/main/train.csv",
    "https://raw.githubusercontent.com/srinivasav22/Design-Thinking-and-Data-Science/master/train.csv",
    "https://raw.githubusercontent.com/srinivasav22/Design-Thinking-and-Data-Science/main/train.csv",
    "https://raw.githubusercontent.com/rajatawasthi0707/SUPERSTORE-SALES-DATASET-CASE-STUDY-DAY5-PYTHON-PANDAS-DATA-VISUALISATION/master/train.csv",
    "https://raw.githubusercontent.com/rajatawasthi0707/SUPERSTORE-SALES-DATASET-CASE-STUDY-DAY5-PYTHON-PANDAS-DATA-VISUALISATION/main/train.csv",
    "https://raw.githubusercontent.com/5umit-chandra/RFM_Analysis/main/train.csv",
    "https://raw.githubusercontent.com/5umit-chandra/RFM_Analysis/master/train.csv",
    "https://raw.githubusercontent.com/tushar2594/Superstore-Sales-Forecasting/main/train.csv",
    "https://raw.githubusercontent.com/tushar2594/Superstore-Sales-Forecasting/master/train.csv",
    "https://raw.githubusercontent.com/curran/data/gh-pages/superstoreSales/superstoreSales.csv"
]

downloaded = False
for url in urls:
    print(f"Testing URL: {url}")
    try:
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req) as resp:
            if resp.status == 200:
                print(f"Found active URL! Downloading from {url}...")
                urllib.request.urlretrieve(url, "train.csv")
                print("Download successful!")
                downloaded = True
                # Quick verification
                with open("train.csv", "r", encoding="utf-8", errors="ignore") as f:
                    header = f.readline()
                    print(f"Header: {header}")
                break
    except Exception as e:
        print(f"Failed: {e}")

if not downloaded:
    print("Could not download train.csv from any of the standard sources.")
