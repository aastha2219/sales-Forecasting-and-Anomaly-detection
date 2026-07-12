# 📊 Retail Sales Forecasting & Anomaly Detection Intelligence Hub

A premium end-to-end Machine Learning and Business Intelligence solution for analyzing retail sales performance, detecting weekly revenue anomalies, clustering product demand segments, and forecasting out-of-sample demand. 

Built with **Streamlit**, **Pandas**, **Plotly**, **Scikit-Learn**, **Prophet**, and **XGBoost**.

---

## 🚀 Live Demo & Repository
- **GitHub Repository**: [sales-Forecasting-and-Anomaly-detection](https://github.com/aastha2219/sales-Forecasting-and-Anomaly-detection)
- **Live Streamlit App**: [sales-forecasting-and-anomaly-detection.streamlit.app](https://sales-forecasting-and-anomaly-detection-dfqzqqgzsvcnlkup9cqq4j.streamlit.app/)
- **Google Colab Notebook**: [Open in Google Colab](https://colab.research.google.com/github/aastha2219/sales-Forecasting-and-Anomaly-detection/blob/main/analysis.ipynb)
- **Deployment Platform**: Streamlit Cloud

---

## ✨ Key Features

### 1. 📊 Executive Sales Dashboard
- Real-time interactive analysis of key retail metrics: **Total Revenue**, **Total Orders**, **Average Order Value**, and **Average Shipping Time**.
- Drill down by **Region** and **Product Category**.
- High-fidelity visual trends: Annual Revenue Performance, Monthly Revenue Trends, and Category/Regional contributions.

### 2. 🔮 Forecast Explorer
- Interactive selection of forecast targets (specific Categories or Regions) with adjustable horizons (up to 3 months).
- Out-of-sample forecasting using the best-performing models (evaluating Prophet, XGBoost, etc.).
- Comprehensive evaluation metrics transparently displayed: **MAE**, **RMSE**, and **MAPE (%)** alongside a full model comparison matrix.

### 3. ⚠️ Weekly Sales Anomaly Audit
- Automated anomaly detection utilizing two complementary methodologies:
  - **Isolation Forest**: Multi-dimensional machine learning-based flagging.
  - **Rolling Z-Score**: Identifies deviations greater than 2 standard deviations from the 12-week moving average.
- Overlap indicator highlighting high-severity flags.
- Detailed audit trail with automated business contextual explanations (e.g., Holiday/Festive season spikes, Back-to-school promotional demand, post-holiday drops).

### 4. 🎯 Demand Segments (Product Clustering)
- Unsupervised learning (K-Means Clustering) applied to sub-categories based on volume, growth rate, volatility, and average order value.
- Interactive 2D visualization via Principal Component Analysis (PCA).
- Prescriptive inventory stocking strategies for each segment:
  - **Cluster 0**: High-Value, Growing & Volatile (Just-In-Time replenishment).
  - **Cluster 1**: High-Volume, Stable (High safety stock service level / Cash Cows).
  - **Cluster 2**: Growing Demand (Incremental stock capacity scaling).
  - **Cluster 3**: Low-Volume, Declining (Clearance, minimal holding).

---

## 🛠️ Project Structure

```bash
SalesForecasting_AasthaAnshu/
│
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Project dependencies for production & deployment
├── analysis.ipynb              # Full EDA, Model Training, & Clustering notebook
├── generate_notebook.py        # Automation script to generate the analysis notebook
├── generate_report.py          # Script to compile executive summary PDF reports
├── summary.pdf                 # Compiled executive PDF report
│
├── train.csv                   # Historical Superstore dataset
├── vgsales.csv                 # Video Game Sales dataset (cross-source analysis)
│
├── model_comparison.csv        # Pre-calculated model evaluation metrics
├── segment_forecasts.csv       # Pre-calculated 3-month out-of-sample forecasts
├── product_segments.csv        # Product clustering and PCA features
├── weekly_anomalies.csv        # Weekly anomaly flags and statistical deviations
│
└── charts/                     # Generated visualization assets
```

---

## 💻 Local Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/aastha2219/sales-Forecasting-and-Anomaly-detection.git
   cd sales-Forecasting-and-Anomaly-detection
   ```

2. **Set up a Virtual Environment** (Recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

---

## 🎛️ Technology Stack
- **Dashboard UI**: Streamlit (Premium dark glassmorphism styling)
- **Data Manipulation**: Pandas, NumPy, OpenPyXL
- **Visualization**: Plotly Express, Plotly Graph Objects, Matplotlib
- **Machine Learning & Modeling**: Scikit-Learn, Prophet, XGBoost, Statsmodels
- **Reporting**: ReportLab (PDF compiler), Jinja2
