import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
import re

# ==========================================
# 1. CORE MATH ENGINE (Linear & Stats)
# ==========================================

def monthly_series(series_df, date_col='Created_Date', value_col='TotSum (actual)'):
    """Converts raw data into monthly sums."""
    series_df = series_df.copy()
    series_df[date_col] = pd.to_datetime(series_df[date_col], errors='coerce')
    series_df = series_df.dropna(subset=[date_col])
    if series_df.empty:
        return pd.Series(dtype=float)
    series_df['Month'] = series_df[date_col].dt.to_period('M')
    ms = series_df.groupby('Month')[value_col].sum().sort_index()
    return ms

def predict_linear_monthly(ms, months_ahead=1, min_months=2):
    """
    Simple linear trend forecasting. 
    Returns a list of predicted values for the next N months.
    """
    if ms is None or len(ms) < min_months:
        # Fallback: Return average if not enough data for trend
        if ms is not None and len(ms) > 0:
            return [float(ms.mean())] * months_ahead
        return [0.0] * months_ahead

    # Linear Regression
    x = np.arange(len(ms)).reshape(-1, 1)
    y = ms.values
    model = LinearRegression().fit(x, y)
    
    preds = []
    start = len(ms)
    for i in range(months_ahead):
        val = model.predict(np.array([[start + i]]))[0]
        preds.append(max(0.0, float(val))) # No negative costs
    return preds

def _plot_history_and_forecast(ms, preds, title, y_label="Cost (LKR)"):
    """Generates the standard Blue (History) -> Red (Forecast) chart."""
    fig = go.Figure()
    
    # History
    if ms is not None and not ms.empty:
        x_hist = ms.index.astype(str)
        fig.add_trace(go.Scatter(x=x_hist, y=ms.values, mode='lines+markers', name='History', line=dict(color='#1f77b4', width=3)))
        last_date = ms.index[-1]
        last_val = ms.values[-1]
    else:
        last_date = pd.Period(datetime.now(), 'M')
        last_val = 0

    # Forecast
    if preds:
        # Create future dates
        future_dates = [str(last_date + i + 1) for i in range(len(preds))]
        
        # Connect last history point to first forecast point
        x_conn = [str(last_date), future_dates[0]]
        y_conn = [last_val, preds[0]]
        fig.add_trace(go.Scatter(x=x_conn, y=y_conn, mode='lines', showlegend=False, line=dict(color='#d62728', width=2, dash='dot')))
        
        # Plot rest of forecast
        fig.add_trace(go.Scatter(x=future_dates, y=preds, mode='lines+markers', name='Forecast', line=dict(color='#d62728', width=2, dash='dot')))

    fig.update_layout(title=title, xaxis_title="Month", yaxis_title=y_label, hovermode="x unified", template="plotly_dark")
    return fig

# ==========================================
# 2. DEEP ANALYZER (The Main Logic)
# ==========================================

def deep_six_month_analyzer(df):
    st.markdown("## 🏭 Deep Industrial Analyzer (6-Month Lookback)")
    
    if 'Created_Date' not in df.columns:
        st.error("Error: 'Created_Date' column missing.")
        return

    # 1. DATA PREP (Last 6 Months Only)
    df['Created_Date'] = pd.to_datetime(df['Created_Date'], errors='coerce')
    df = df.dropna(subset=['Created_Date'])
    
    max_date = df['Created_Date'].max()
    cutoff_date = max_date - relativedelta(months=6)
    df6 = df[df['Created_Date'] >= cutoff_date].copy()
    
    if df6.empty:
        st.warning("No data found in the last 6 months.")
        return

    # 2. CLEANUP
    if 'Equipment description' in df6.columns:
        # Remove entries where equipment is unknown or empty
        df6 = df6[df6['Equipment description'].notna()]
        df6 = df6[df6['Equipment description'].astype(str).str.strip() != '']
        df6 = df6[~df6['Equipment description'].astype(str).str.lower().isin(['nan', 'unknown', 'none', 'unknown asset'])]

    # --- SECTION A: CRITICAL ASSET FAILURE ANALYTICS ---
    st.markdown("### 1. 🚨 Critical Asset Reliability & Spare Parts")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # PIE CHART: Failure Distribution
        if 'Equipment description' in df6.columns:
            fail_counts = df6['Equipment description'].value_counts().head(10)
            fig_fail = px.pie(
                names=fail_counts.index, 
                values=fail_counts.values, 
                title="Top 10 Failing Assets (Failure %)",
                hole=0.4
            )
            fig_fail.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_fail, use_container_width=True)
            
            top_asset_list = fail_counts.index.tolist()
        else:
            st.warning("Equipment description column missing.")
            top_asset_list = []

    with col2:
        # INTERACTIVE: Asset Specific Spare Parts
        st.markdown("#### 🛠️ Asset Spare Part Forecaster")
        st.info("Select an asset below to see what parts it will need next month.")
        
        selected_asset = st.selectbox("Select Failing Asset:", top_asset_list)
        
        if selected_asset:
            # Filter data for this asset
            asset_df = df6[df6['Equipment description'] == selected_asset]
            
            # Extract Keywords from Description (Simple NLP)
            vectorizer = CountVectorizer(stop_words='english', max_features=20)
            try:
                X = vectorizer.fit_transform(asset_df['Description'].fillna("").astype(str))
                words = vectorizer.get_feature_names_out()
                counts = X.toarray().sum(axis=0)
                
                # Create a mini forecast for these parts
                part_forecasts = []
                for word, count in zip(words, counts):
                    # Filter for this word specifically
                    mask = asset_df['Description'].str.contains(word, case=False, na=False)
                    monthly_usage = monthly_series(asset_df[mask], value_col='Order') # Count of orders
                    next_month_qty = predict_linear_monthly(monthly_usage, months_ahead=1)[0]
                    
                    if next_month_qty > 0.1: # Only show significant predictions
                        part_forecasts.append({
                            'Spare Part Keyword': word.capitalize(),
                            'History (6m)': count,
                            'Forecast (Next Month)': round(next_month_qty, 1)
                        })
                
                pf_df = pd.DataFrame(part_forecasts).sort_values('Forecast (Next Month)', ascending=False)
                
                if not pf_df.empty:
                    st.dataframe(pf_df, use_container_width=True, hide_index=True)
                    st.caption(f"Based on consumption patterns for **{selected_asset}**.")
                else:
                    st.write("No specific spare part trends detected for this asset.")
                    
            except ValueError:
                st.write("Not enough text data to extract spare parts.")

    # --- SECTION B: MECHANICAL VS ELECTRICAL ---
    st.markdown("### 2. ⚡ Mechanical vs Electrical Forecast")
    
    mech_mask = pd.Series([False] * len(df6), index=df6.index)
    elec_mask = pd.Series([False] * len(df6), index=df6.index)
    
    if 'Main WorkCtr' in df6.columns:
        mech_mask = df6['Main WorkCtr'].astype(str).str.contains('Mech', case=False)
        elec_mask = df6['Main WorkCtr'].astype(str).str.contains('Elec', case=False)
    
    if not mech_mask.any() and not elec_mask.any():
        mech_keywords = ['pump', 'gear', 'bearing', 'mech', 'weld', 'pipe', 'valve']
        elec_keywords = ['motor', 'sensor', 'cable', 'fuse', 'switch', 'panel', 'electric', 'circuit']
        mech_mask = df6['Description'].str.contains('|'.join(mech_keywords), case=False, na=False)
        elec_mask = df6['Description'].str.contains('|'.join(elec_keywords), case=False, na=False)

    mech_series = monthly_series(df6[mech_mask])
    elec_series = monthly_series(df6[elec_mask])
    
    mech_pred = predict_linear_monthly(mech_series, months_ahead=3)
    elec_pred = predict_linear_monthly(elec_series, months_ahead=3)
    
    fig_split = go.Figure()
    if not mech_series.empty:
        fig_split.add_trace(go.Scatter(x=mech_series.index.astype(str), y=mech_series.values, name="Mechanical (Hist)", line=dict(color='cyan')))
        last_date = mech_series.index[-1]
        future_dates = [str(last_date + i + 1) for i in range(3)]
        fig_split.add_trace(go.Scatter(x=future_dates, y=mech_pred, name="Mech Forecast", line=dict(color='cyan', dash='dot')))

    if not elec_series.empty:
        fig_split.add_trace(go.Scatter(x=elec_series.index.astype(str), y=elec_series.values, name="Electrical (Hist)", line=dict(color='orange')))
        last_date = elec_series.index[-1]
        future_dates = [str(last_date + i + 1) for i in range(3)]
        fig_split.add_trace(go.Scatter(x=future_dates, y=elec_pred, name="Elec Forecast", line=dict(color='orange', dash='dot')))
        
    fig_split.update_layout(title="Departmental Spending Forecast", xaxis_title="Month", yaxis_title="Cost", template="plotly_dark")
    st.plotly_chart(fig_split, use_container_width=True)

    # --- SECTION C: FAILURE HOTSPOTS ---
    st.markdown("### 3. 📍 Failure Hotspot Detection")
    if 'Functional Location' in df6.columns:
        hotspots = df6['Functional Location'].value_counts().head(10).reset_index()
        hotspots.columns = ['Location', 'Failures']
        fig_hot = px.bar(hotspots, x='Failures', y='Location', orientation='h', title="Most Problematic Locations", color='Failures', color_continuous_scale='Reds')
        st.plotly_chart(fig_hot, use_container_width=True)
    else:
        st.info("Functional Location column not found.")

    # --- SECTION D: CRITICAL LINE FAILURE FORECAST (Markov) ---
    st.markdown("### 4. 📉 Critical Line Failure Forecast (Markov Chain)")
    daily_status = df6.groupby(df6['Created_Date'].dt.date)['Order Type'].apply(lambda x: 'Breakdown' if 'Breakdown maintenance' in x.values else 'Running')
    
    if len(daily_status) > 10:
        transitions = {'Running->Running': 0, 'Running->Down': 0, 'Down->Running': 0, 'Down->Down': 0}
        states = daily_status.values
        for i in range(len(states)-1):
            current, nxt = states[i], states[i+1]
            key, key_next = ('Running' if current == 'Running' else 'Down'), ('Running' if nxt == 'Running' else 'Down')
            transitions[f"{key}->{key_next}"] += 1
            
        run_runs = transitions['Running->Running'] + transitions['Running->Down']
        down_downs = transitions['Down->Running'] + transitions['Down->Down']
        p_fail = transitions['Running->Down'] / run_runs if run_runs > 0 else 0
        p_recover = transitions['Down->Running'] / down_downs if down_downs > 0 else 0
        p_stuck = transitions['Down->Down'] / down_downs if down_downs > 0 else 0
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk of Failure (Daily)", f"{p_fail:.1%}", "If currently running")
        c2.metric("Recovery Probability", f"{p_recover:.1%}", "Chance to fix in 1 day")
        c3.metric("Critical Stuck Risk", f"{p_stuck:.1%}", "Chance failure lasts >24h")
        
        fig_markov = go.Figure(data=[go.Sankey(
            node = dict(pad = 15, thickness = 20, line = dict(color = "black", width = 0.5), label = ["Running", "Down"], color = ["green", "red"]),
            link = dict(source = [0, 0, 1, 1], target = [0, 1, 0, 1], value = [transitions['Running->Running'], transitions['Running->Down'], transitions['Down->Running'], transitions['Down->Down']])
        )])
        fig_markov.update_layout(title="Operational State Transitions", font_size=10)
        st.plotly_chart(fig_markov, use_container_width=True)
    else:
        st.write("Not enough daily data points to build a Markov Model.")

    # --- SECTION E: TOTAL COST FORECAST ---
    st.markdown("### 5. 💰 Global Budget Forecast")
    total_series = monthly_series(df6)
    total_preds = predict_linear_monthly(total_series, months_ahead=3)
    fig_total = _plot_history_and_forecast(total_series, total_preds, "Total Maintenance Cost Forecast")
    st.plotly_chart(fig_total, use_container_width=True)


# ==========================================
# 3. COMPATIBILITY WRAPPERS (Prevents ImportErrors)
# ==========================================
def forecast_spare_parts(df):
    """
    Compatibility wrapper: Redirects to Deep Analyzer or returns placeholder.
    This prevents app.py from crashing when it imports this function.
    """
    return None, "Please use 'AI Analyzer' mode for advanced spare part forecasts."

def forecast_cost_prophet(df):
    """
    Compatibility wrapper for Prophet forecasting.
    """
    # Simple logic to satisfy the import
    return None, None, "Prophet logic moved to Deep Analyzer."

def forecast_failure_rf(df):
    """
    Compatibility wrapper for RF/XGBoost.
    """
    return None, "ML Logic moved to Deep Analyzer."
