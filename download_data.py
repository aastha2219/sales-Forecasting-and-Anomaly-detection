import urllib.request
import os

def download_file(url, filename):
    print(f"Downloading {url} to {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Successfully downloaded {filename}")
        return True
    except Exception as e:
        print(f"Error downloading {filename} from {url}: {e}")
        return False

# Superstore sales dataset URLs to try (primary and fallback)
superstore_urls = [
    "https://raw.githubusercontent.com/aditya-yadav-01/Sales-Forecasting/main/train.csv",
    "https://raw.githubusercontent.com/5umit-chandra/RFM_Analysis/master/train.csv",
    "https://raw.githubusercontent.com/tushar2594/Superstore-Sales-Forecasting/master/train.csv"
]

# Video Games sales dataset URLs to try
vg_urls = [
    "https://raw.githubusercontent.com/raghav-19/Video-Games-Sales-Data-Analysis/master/vgsales.csv",
    "https://raw.githubusercontent.com/gregorut/videogamesales/master/vgsales.csv",
    "https://raw.githubusercontent.com/ricardocmuller/Video_Game_Sales_Analysis_and_Clustering/master/vgsales.csv"
]

# Download Superstore Sales
success = False
for url in superstore_urls:
    if download_file(url, "train.csv"):
        success = True
        break
if not success:
    print("Failed to download train.csv from all sources")

# Download Video Game Sales
success = False
for url in vg_urls:
    if download_file(url, "vgsales.csv"):
        success = True
        break
if not success:
    print("Failed to download vgsales.csv from all sources")
