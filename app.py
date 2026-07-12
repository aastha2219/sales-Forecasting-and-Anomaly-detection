import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# Set page config
st.set_page_config(
    page_title="Retail Sales Forecasting & Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling (Dark Glassmorphic UI)
st.markdown("""
<style>
    /* Main App Background */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #161a24 100%);
        color: #e2e8f0;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #2d3748;
    }
    
    /* Headings */
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Card design */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 4px 30px rgba(56, 189, 248, 0.1);
        transform: translateY(-2px);
    }
    
    /* Table headers */
    .stDataFrame {
        border: 1px solid #2d3748;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Data loading with caching
@st.cache_data
def load_superstore_data():
    df = pd.read_csv('train.csv', encoding='utf-8')
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='mixed', dayfirst=True)
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='mixed', dayfirst=True)
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['Ship_Duration'] = (df['Ship Date'] - df['Order Date']).dt.days
    return df

@st.cache_data
def load_comparison_data():
    if os.path.exists('model_comparison.csv'):
        return pd.read_csv('model_comparison.csv')
    return None

@st.cache_data
def load_segment_forecasts():
    if os.path.exists('segment_forecasts.csv'):
        df = pd.read_csv('segment_forecasts.csv', index_col=0)
        df.index = pd.to_datetime(df.index)
        return df
    return None

@st.cache_data
def load_anomaly_data():
    if os.path.exists('weekly_anomalies.csv'):
        df = pd.read_csv('weekly_anomalies.csv')
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        return df
    return None

@st.cache_data
def load_cluster_data():
    if os.path.exists('product_segments.csv'):
        return pd.read_csv('product_segments.csv')
    return None

# Attempt to load datasets
try:
    df = load_superstore_data()
    comp_df = load_comparison_data()
    seg_fc_df = load_segment_forecasts()
    anomaly_df = load_anomaly_data()
    cluster_df = load_cluster_data()
except Exception as e:
    st.error(f"Error loading data. Make sure to run the notebook first. Error: {e}")
    st.stop()

# Sidebar Navigation
st.sidebar.markdown("<h2 style='text-align: center;'>Intelligence Hub</h2>", unsafe_allow_html=True)
page = st.sidebar.radio(
    "Select Workspace",
    ["📊 Sales Overview", "🔮 Forecast Explorer", "⚠️ Anomaly Report", "🎯 Demand Segments"]
)

# Header Section
st.markdown("<h1 style='margin-bottom: 0px;'>Retail Sales Forecasting & Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748b; font-size: 14px; margin-top: 5px;'>Powered by Advanced Analytics, Machine Learning & Forecasting Models</p>", unsafe_allow_html=True)
st.markdown("---")

# ----------------- PAGE 1: SALES OVERVIEW -----------------
if page == "📊 Sales Overview":
    st.subheader("Executive Sales Dashboard")
    
    # Sidebar filters for page 1
    st.sidebar.markdown("### Filters")
    selected_regions = st.sidebar.multiselect("Select Region", options=df['Region'].unique(), default=df['Region'].unique())
    selected_categories = st.sidebar.multiselect("Select Category", options=df['Category'].unique(), default=df['Category'].unique())
    
    # Filter dataset
    filtered_df = df[df['Region'].isin(selected_regions) & df['Category'].isin(selected_categories)]
    
    if filtered_df.empty:
        st.warning("No data matches selected filters. Please select at least one region and category.")
        st.stop()
        
    # Key Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        total_sales = filtered_df['Sales'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <span style="color:#64748b; font-size: 13px; font-weight: 600; text-transform: uppercase;">Total Revenue</span>
            <h2 style="margin: 5px 0 0 0; background: linear-gradient(90deg, #38bdf8, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">${total_sales:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        total_orders = filtered_df['Order ID'].nunique()
        st.markdown(f"""
        <div class="metric-card">
            <span style="color:#64748b; font-size: 13px; font-weight: 600; text-transform: uppercase;">Total Orders</span>
            <h2 style="margin: 5px 0 0 0; background: linear-gradient(90deg, #34d399, #059669); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{total_orders:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        avg_order = filtered_df.groupby('Order ID')['Sales'].sum().mean()
        st.markdown(f"""
        <div class="metric-card">
            <span style="color:#64748b; font-size: 13px; font-weight: 600; text-transform: uppercase;">Avg Order Value</span>
            <h2 style="margin: 5px 0 0 0; background: linear-gradient(90deg, #fbbf24, #d97706); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">${avg_order:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        avg_ship = filtered_df['Ship_Duration'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <span style="color:#64748b; font-size: 13px; font-weight: 600; text-transform: uppercase;">Avg Shipping Time</span>
            <h2 style="margin: 5px 0 0 0; background: linear-gradient(90deg, #f87171, #dc2626); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{avg_ship:.2f} Days</h2>
        </div>
        """, unsafe_allow_html=True)
        
    # Charts Section
    col1, col2 = st.columns(2)
    
    with col1:
        # Total Sales by Year
        st.markdown("### Annual Revenue Performance")
        annual_sales = filtered_df.groupby('Year')['Sales'].sum().reset_index()
        fig_year = px.bar(
            annual_sales, x='Year', y='Sales',
            text_auto='.2s',
            labels={'Sales': 'Revenue ($)', 'Year': 'Fiscal Year'},
            color='Sales',
            color_continuous_scale=px.colors.sequential.Tealgrn
        )
        fig_year.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#cbd5e1',
            coloraxis_showscale=False,
            height=380
        )
        st.plotly_chart(fig_year, use_container_width=True)
        
    with col2:
        # Monthly Sales Trend
        st.markdown("### Monthly Revenue Trends")
        monthly_sales = filtered_df.set_index('Order Date').resample('MS')['Sales'].sum().reset_index()
        fig_monthly = px.line(
            monthly_sales, x='Order Date', y='Sales',
            labels={'Sales': 'Revenue ($)', 'Order Date': 'Date'},
            markers=True
        )
        fig_monthly.update_traces(line=dict(color='#38bdf8', width=3))
        fig_monthly.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#cbd5e1',
            height=380
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
        
    # Breakdown Section
    st.markdown("### Category and Regional Contributions")
    c1, c2 = st.columns(2)
    with c1:
        cat_sales = filtered_df.groupby('Category')['Sales'].sum().reset_index()
        fig_cat = px.pie(cat_sales, values='Sales', names='Category', hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_cat.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#cbd5e1', height=300)
        st.plotly_chart(fig_cat, use_container_width=True)
    with c2:
        reg_sales = filtered_df.groupby('Region')['Sales'].sum().reset_index()
        fig_reg = px.bar(reg_sales, x='Region', y='Sales', color='Region',
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig_reg.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#cbd5e1', height=300)
        st.plotly_chart(fig_reg, use_container_width=True)

# ----------------- PAGE 2: FORECAST EXPLORER -----------------
elif page == "🔮 Forecast Explorer":
    st.subheader("Interactive Sales Forecasting Hub")
    
    # Set selector
    st.markdown("### Segment Selector")
    seg_options = list(seg_fc_df.columns)
    selected_segment = st.selectbox("Select Category or Region to Forecast", options=seg_options)
    
    horizon_slider = st.slider("Select Forecast Horizon (Months)", min_value=1, max_value=3, value=3)
    
    # Retrieve forecast and actual historical data
    # Filter historical segment data
    if selected_segment in df['Category'].unique():
        hist_df = df[df['Category'] == selected_segment]
    elif selected_segment in df['Region'].unique() or selected_segment.replace(" Region", "") in df['Region'].unique():
        reg_name = selected_segment.replace(" Region", "")
        hist_df = df[df['Region'] == reg_name]
    else:
        # Default fallback is overall
        hist_df = df
        
    hist_monthly = hist_df.set_index('Order Date').resample('MS')['Sales'].sum()
    
    # Forecast dates
    forecast_values = seg_fc_df[selected_segment].iloc[:horizon_slider]
    
    # Plotly forecast chart
    st.markdown(f"### 3-Month Out-of-Sample Forecast for {selected_segment}")
    
    fig_fc = go.Figure()
    
    # Historical (last 12 months for context)
    fig_fc.add_trace(go.Scatter(
        x=hist_monthly.index[-12:],
        y=hist_monthly.values[-12:],
        name="Historical Sales (Last 12m)",
        line=dict(color="#cbd5e1", width=3),
        mode="lines+markers"
    ))
    
    # Forecast
    fig_fc.add_trace(go.Scatter(
        x=forecast_values.index,
        y=forecast_values.values,
        name="Best Model Forecast",
        line=dict(color="#a855f7", width=3, dash="dash"),
        mode="lines+markers"
    ))
    
    fig_fc.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#cbd5e1',
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        height=450
    )
    
    st.plotly_chart(fig_fc, use_container_width=True)
    
    # Model evaluation metrics section
    st.markdown("### Model Selection & Evaluation Metrics")
    
    # Show metrics of best model
    # We display the model comparison table
    if comp_df is not None:
        best_model_row = comp_df.iloc[comp_df['MAE'].idxmin()]
        st.markdown(f"**Recommended Production Model:** `{best_model_row['Model']}` (Lowest Error)")
        
        # Display columns for metrics
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Mean Absolute Error (MAE)", f"${best_model_row['MAE']:,.2f}")
        with c2:
            st.metric("Root Mean Squared Error (RMSE)", f"${best_model_row['RMSE']:,.2f}")
        with c3:
            st.metric("Mean Absolute Percentage Error (MAPE)", f"{best_model_row['MAPE (%)']:.2f}%")
            
        st.markdown("#### Full Model Comparison Table")
        st.dataframe(comp_df, use_container_width=True)

# ----------------- PAGE 3: ANOMALY REPORT -----------------
elif page == "⚠️ Anomaly Report":
    st.subheader("Weekly Sales Anomaly Audit")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Weekly Sales Trends & Flagged Anomalies")
        # Plotly anomaly chart
        fig_anom = go.Figure()
        
        # Weekly sales
        fig_anom.add_trace(go.Scatter(
            x=anomaly_df['Order Date'],
            y=anomaly_df['Sales'],
            name="Weekly Sales",
            line=dict(color="#38bdf8", width=2),
            opacity=0.6
        ))
        
        # Isolation Forest anomalies
        iforest_anom = anomaly_df[anomaly_df['IForest_Anomaly'] == 1]
        fig_anom.add_trace(go.Scatter(
            x=iforest_anom['Order Date'],
            y=iforest_anom['Sales'],
            name="IForest Anomaly Flag",
            mode="markers",
            marker=dict(color="red", size=10, symbol="x", line=dict(width=2))
        ))
        
        # Z-Score anomalies
        zscore_anom = anomaly_df[anomaly_df['ZScore_Anomaly'] == 1]
        fig_anom.add_trace(go.Scatter(
            x=zscore_anom['Order Date'],
            y=zscore_anom['Sales'],
            name="Z-Score Anomaly Flag (> 2 SD)",
            mode="markers",
            marker=dict(color="orange", size=12, symbol="circle-open", line=dict(width=2))
        ))
        
        fig_anom.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#cbd5e1',
            xaxis_title="Date",
            yaxis_title="Weekly Sales ($)",
            height=480
        )
        st.plotly_chart(fig_anom, use_container_width=True)
        
    with col2:
        st.markdown("### Anomaly Overview")
        st.markdown("""
        Our anomaly detection engine audits sales spikes or drops at a weekly granularity. 
        - **Isolation Forest:** Multi-dimensional ML anomaly flagging.
        - **Rolling Z-Score:** Flags deviation > 2 standard deviations from the 12-week moving average.
        """)
        
        overlap_cnt = len(anomaly_df[(anomaly_df['IForest_Anomaly'] == 1) & (anomaly_df['ZScore_Anomaly'] == 1)])
        st.metric("Overlapping Flags (High Severity)", f"{overlap_cnt} Weeks")
        st.metric("Total IForest Flags", f"{len(iforest_anom)} Weeks")
        st.metric("Total Z-Score Flags", f"{len(zscore_anom)} Weeks")
        
    st.markdown("### Detailed Audit Trail of Anomalous Weeks")
    # Display table of anomalous dates
    anom_table = anomaly_df[(anomaly_df['IForest_Anomaly'] == 1) | (anomaly_df['ZScore_Anomaly'] == 1)].copy()
    anom_table = anom_table.sort_values('Order Date', ascending=False)
    
    # Add human readable reasons
    reasons = []
    for idx, row in anom_table.iterrows():
        month = row['Order Date'].month
        sales = row['Sales']
        z = row['Z_Score']
        
        if month == 11 or month == 12:
            reasons.append("Festive/Holiday Season (Thanksgiving/Christmas) Sales Spike")
        elif month == 9 and sales > 35000:
            reasons.append("Back-to-School/Q3 End Corporate Purchases Spike")
        elif sales < 5000:
            reasons.append("Operational Holiday/Post-Festive Inventory Drop")
        elif z > 2.0:
            reasons.append("Mid-Quarter Promotional Inventory Push")
        else:
            reasons.append("Unusual Transaction Volume / Bulk Order")
            
    anom_table['Likely Explanation'] = reasons
    
    # Display columns nicely
    disp_table = anom_table[['Order Date', 'Sales', 'Z_Score', 'IForest_Anomaly', 'ZScore_Anomaly', 'Likely Explanation']].copy()
    disp_table.columns = ['Week Ending', 'Weekly Sales ($)', 'Z-Score', 'IForest Flag', 'Z-Score Flag', 'Likely Explanation']
    st.dataframe(disp_table.style.format({'Weekly Sales ($)': '{:,.2f}', 'Z-Score': '{:.2f}'}), use_container_width=True)

# ----------------- PAGE 4: PRODUCT DEMAND SEGMENTS -----------------
elif page == "🎯 Demand Segments":
    st.subheader("Product Sub-Category Clustering Analysis")
    
    st.markdown("""
    To optimize warehouse stocking levels, we group product sub-categories based on 4 metrics:
    **Total Sales Volume**, **Sales Growth Rate (YoY)**, **Volatility (Standard Deviation)**, and **Average Order Value**.
    """)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### 2D Product Segments Clusters (PCA Visualization)")
        # Plot PCA
        fig_cluster = px.scatter(
            cluster_df, x='PCA1', y='PCA2',
            color='Cluster',
            text='Sub-Category',
            color_continuous_scale=px.colors.qualitative.Set1,
            labels={'PCA1': 'Principal Component 1', 'PCA2': 'Principal Component 2'}
        )
        fig_cluster.update_traces(marker=dict(size=14, opacity=0.85, line=dict(width=1, color='white')))
        fig_cluster.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#cbd5e1',
            coloraxis_showscale=False,
            height=480
        )
        st.plotly_chart(fig_cluster, use_container_width=True)
        
    with col2:
        st.markdown("### Stocking Strategies by Cluster")
        
        # Generate cluster insights
        # Assign business names to clusters based on mean values
        # Let's read means and print nicely
        st.markdown("""
        **Cluster 0: High-Value, Growing & Highly Volatile**
        - *Stocking Strategy:* Keep moderate safety stock. Implement Just-In-Time (JIT) replenishment for copiers and machines to avoid locking up capital in expensive, slow-moving items.
        
        **Cluster 1: High Volume, Steady Demand (Cash Cows)**
        - *Stocking Strategy:* Maintain high stock levels (safety stock > 95% service level). Negotiate bulk shipping rates and discount vendor contracts as these items have highly predictable demands.
        
        **Cluster 2: Growing Demand / Emerging Categories**
        - *Stocking Strategy:* Gradually increase stock capacity. Closely monitor monthly growth rates and adjust lead times accordingly.
        
        **Cluster 3: Low Volume, Stagnant / Declining Demand**
        - *Stocking Strategy:* Minimize inventory holdings. Clean out slow-moving stock using clearing sales or bundle offers. Order only on demand.
        """)
        
    st.markdown("### Product Sub-Category Metrics & Segment Allocations")
    
    # Rename clusters for table representation
    cluster_names = {
        0: "High Value, Volatile",
        1: "High Volume, Stable (Cash Cows)",
        2: "Growing / Emerging",
        3: "Low Volume, Declining"
    }
    
    cluster_table = cluster_df.copy()
    cluster_table['Demand Segment'] = cluster_table['Cluster'].map(cluster_names)
    
    disp_cluster = cluster_table[['Sub-Category', 'Total_Sales', 'Avg_Order_Value', 'Volatility', 'Growth_Rate', 'Demand Segment']].copy()
    disp_cluster.columns = ['Sub-Category', 'Total Sales ($)', 'Avg Order Value ($)', 'Volatility (Std Dev)', 'YoY Growth Rate (%)', 'Demand Segment']
    
    st.dataframe(
        disp_cluster.style.format({
            'Total Sales ($)': '{:,.2f}',
            'Avg Order Value ($)': '{:,.2f}',
            'Volatility (Std Dev)': '{:,.2f}',
            'YoY Growth Rate (%)': '{:+.2f}%'
        }),
        use_container_width=True
    )
