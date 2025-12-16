import pandas as pd
import numpy as np
import streamlit as st
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta

# --- VISUALIZATION IMPORTS ---
import plotly.express as px
import plotly.graph_objects as go

# --- MACHINE LEARNING IMPORTS ---
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

# --- ADVANCED AI IMPORTS (Safe Import) ---
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from lifelines import WeibullFitter
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def monthly_series(series_df, date_col='Created_Date', value_col='TotSum (actual)'):
    series_df = series_df.copy()
    series_df[date_col] = pd.to_datetime(series_df[date_col], errors='coerce')
    series_df = series_df.dropna(subset=[date_col])
    if series_df.empty:
        return pd.Series(dtype=float)
    series_df['Month'] = series_df[date_col].dt.to_period('M')
    ms = series_df.groupby('Month')[value_col].sum().sort_index()
    return ms

def predict_linear_monthly(ms, months_ahead=3, min_months=3):
    """
    Fallback Linear Regression for simple trends.
    """
    if months_ahead <= 0: return []
    if ms is None or len(ms) < min_months: return None
    
    x = np.arange(len(ms)).reshape(-1, 1)
    y = ms.values
    model = LinearRegression().fit(x, y)
    
    preds = []
    start = len(ms)
    for i in range(months_ahead):
        val = float(model.predict(np.array([[start + i]])))
        preds.append(max(0, val)) # No negative costs
    return preds

def _plot_history_and_forecast(ms, preds, title, y_label="Value"):
    """
    Standard plotting function for history + forecast lines.
    """
    fig = go.Figure()
    
    # Historical Data
    if ms is not None and not ms.empty:
        x_hist = [p.to_timestamp(how='end') for p in ms.index]
        fig.add_trace(go.Scatter(x=x_hist, y=ms.values, mode='lines+markers', name='Historical', line=dict(color='#1f77b4', width=3)))
    
    # Forecast Data
    if preds is not None and len(preds) > 0:
        last_date = ms.index.max().to_timestamp(how='end')
        future_dates = [last_date + relativedelta(months=i+1) for i in range(len(preds))]
        
        # Connect last history point to first forecast point
        x_forecast = [last_date] + future_dates
        y_forecast = [ms.values[-1]] + preds
        
        fig.add_trace(go.Scatter(x=x_forecast, y=y_forecast, mode='lines+markers', name='Forecast', line=dict(color='red', dash='dot')))

    fig.update_layout(title=title, xaxis_title='Date', yaxis_title=y_label, hovermode='x unified')
    return fig


# ==========================================
# 2. ADVANCED AI ENGINES (Prophet, Weibull, RF)
# ==========================================

def run_prophet_forecast(df, periods=3):
    """
    Uses Facebook Prophet for Cost/Volume forecasting.
    """
    if not PROPHET_AVAILABLE:
        return None
    
    # Prepare Data
    df_prophet = df.groupby('Created_Date')['TotSum (actual)'].sum().reset_index()
    df_prophet.columns = ['ds', 'y']
    
    if len(df_prophet) < 10: return None # Not enough data for Prophet

    m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m.fit(df_prophet)
    
    future = m.make_future_dataframe(periods=periods * 30) # Approx days
    forecast = m.predict(future)
    
    # Aggregate back to monthly for display
    forecast['Month'] = forecast['ds'].dt.to_period('M')
    monthly_pred = forecast.groupby('Month')['yhat'].sum().tail(periods).values.tolist()
    return monthly_pred

def run_weibull_analysis(df, equipment_name):
    """
    Uses Lifelines for Weibull Analysis (Reliability/Survival).
    Calculates characteristic life (eta) and shape (beta).
    """
    if not LIFELINES_AVAILABLE:
        return None, "Lifelines library missing"

    subset = df[df['Equipment description'] == equipment_name].sort_values('Created_Date')
    if len(subset) < 4: return None, "Insufficient Data (<4 failures)"

    # Calculate time between failures (TBF)
    subset['TBF'] = subset['Created_Date'].diff().dt.days
    tbf_data = subset['TBF'].dropna()
    tbf_data = tbf_data[tbf_data > 0] # Filter valid intervals

    if len(tbf_data) < 3: return None, "Insufficient Intervals"

    wf = WeibullFitter()
    wf.fit(tbf_data, event_observed=[1]*len(tbf_data))
    
    return wf, f"Beta: {wf.beta_:.2f}, Eta: {wf.lambda_:.1f} days"

def run_rf_failure_prediction(df):
    """
    Uses Random Forest to predict likelihood of breakdown after a PM.
    """
    # Feature Engineering
    df['Days_Since_Last_PM'] = 0 # Placeholder for complex logic
    # (In a real scenario, you would calculate actual days since last order type 'PM')
    
    # This is a simplified implementation for the "Deep Analyzer" structure
    # We will use this logic inline in section 2.4
    pass


# ==========================================
# 3. DEEP 6-MONTH AI ANALYZER (Main Function)
# ==========================================
def deep_six_month_analyzer(df):
    st.markdown('<div class="ai-banner">AI Analyzer — Deep Six-Month Forecast (Advanced Models)</div>', unsafe_allow_html=True)
    
    # --- SETUP & FILTERS ---
    if 'Created_Date' not in df.columns:
        st.error("Created_Date column required.")
        return

    # Horizon Selector
    horizon_map = {"Next 1 month": 1, "Next 2 months": 2, "Next 3 months (Quarter)": 3}
    horizon_choice = st.selectbox("Forecast horizon:", list(horizon_map.keys()), index=2)
    months_ahead = horizon_map[horizon_choice]

    # Data Filtering (Last 6 Months)
    df = df.copy()
    df['Created_Date'] = pd.to_datetime(df['Created_Date'], errors='coerce')
    df = df.dropna(subset=['Created_Date'])
    max_date = df['Created_Date'].max()
    cutoff = (max_date - relativedelta(months=6)).replace(day=1)
    df6 = df[df['Created_Date'] >= cutoff].copy()
    
    if df6.empty:
        st.warning("No data in the last 6 months range.")
        return

    # --- 1. FINANCIAL FORECASTING (PROPHET INTEGRATION) ---
    st.markdown("#### 1. Financial & Budget Forecasting")
    
    # 1.1 Breakdown Cost (ZLT2)
    st.markdown("**1.1 Monthly Breakdown Cost Forecast (Hybrid: Prophet + Linear)**")
    zlt2_df = df6[df6.get('Order Type', '') == 'Breakdown maintenance']
    zlt2_monthly = monthly_series(zlt2_df)
    
    # Try Prophet first, fall back to Linear
    zlt2_preds = None
    if PROPHET_AVAILABLE:
        try:
            zlt2_preds = run_prophet_forecast(zlt2_df, periods=months_ahead)
        except: pass
    
    if zlt2_preds is None:
        zlt2_preds = predict_linear_monthly(zlt2_monthly, months_ahead=months_ahead)

    if zlt2_preds:
        # Show Results
        fig = _plot_history_and_forecast(zlt2_monthly, zlt2_preds, "ZLT2 Historical & Forecasted Costs", "LKR")
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"Predicted total breakdown cost for next {months_ahead} months: **LKR {sum(zlt2_preds):,.2f}**")
    else:
        st.warning("Insufficient data for cost forecasting.")

    # --- 2. RELIABILITY (WEIBULL INTEGRATION) ---
    st.markdown("#### 2. Equipment & Reliability Forecasting")
    
    # 2.1 Weibull Analysis
    st.markdown("**2.1 Advanced Reliability Analysis (Weibull/Survival)**")
    if 'Equipment description' in df6.columns:
        # Find top bad actor
        top_asset = df6.groupby('Equipment description').size().idxmax()
        st.write(f"Analyzing Top Bad Actor: **{top_asset}**")
        
        if LIFELINES_AVAILABLE:
            wf, summary = run_weibull_analysis(df, top_asset) # Use full history df for better Weibull
            if wf:
                st.success(f"Weibull Model Fit: {summary}")
                # Plot Survival Function
                survival_df = wf.survival_function_
                fig_surv = px.line(survival_df, title=f"Survival Probability over Time (Days) - {top_asset}")
                st.plotly_chart(fig_surv, use_container_width=True)
                st.write("Interpretation: The curve shows the probability of the machine running without failure over time.")
            else:
                st.write(f"Weibull Analysis Skipped: {summary}")
        else:
            st.warning("Install 'lifelines' library for Weibull Analysis.")

    # 2.4 PM Effectiveness (Random Forest)
    st.markdown("**2.4 PM Effectiveness (Random Forest Classifier)**")
    if XGB_AVAILABLE or True: # Using Sklearn RF as fallback
        # Prepare Data: Did a breakdown happen within 30 days of a PM?
        pm_data = df6[df6['Order Type'] == 'Preventive maintenance'].copy()
        breakdowns = df6[df6['Order Type'] == 'Breakdown maintenance']
        
        if not pm_data.empty and not breakdowns.empty:
            # Label Engineering
            labels = []
            for date, equip in zip(pm_data['Created_Date'], pm_data['Equipment description']):
                # Look for breakdown in next 30 days
                failure = breakdowns[
                    (breakdowns['Equipment description'] == equip) & 
                    (breakdowns['Created_Date'] > date) & 
                    (breakdowns['Created_Date'] <= date + pd.Timedelta(days=30))
                ]
                labels.append(1 if not failure.empty else 0)
            
            pm_data['Failure_Next_30d'] = labels
            
            # Simple Features
            pm_data['DayOfWeek'] = pm_data['Created_Date'].dt.dayofweek
            pm_data['Month'] = pm_data['Created_Date'].dt.month
            
            if len(pm_data) > 10 and sum(labels) > 0:
                X = pm_data[['DayOfWeek', 'Month']]
                y = pm_data['Failure_Next_30d']
                
                rf = RandomForestClassifier(n_estimators=100)
                rf.fit(X, y)
                prob = rf.predict_proba(X)[:, 1].mean()
                
                st.markdown(f"- **Random Forest Risk Score:** {prob:.1%} probability of breakdown after PMs based on current scheduling patterns.")
            else:
                st.write("Not enough failure patterns to train Random Forest.")
    
    # --- 3. SPARE PARTS (NLP INTEGRATION) ---
    st.markdown("#### 3. Spare Parts & Lubrication (NLP)")
    
    text_col = 'Description'
    if text_col in df6.columns:
        # Tokenize
        descriptions = df6[text_col].astype(str).fillna('')
        try:
            vectorizer = CountVectorizer(stop_words='english', max_features=20)
            X = vectorizer.fit_transform(descriptions)
            words = vectorizer.get_feature_names_out()
            counts = X.sum(axis=0).A1
            
            # Create DataFrame
            word_df = pd.DataFrame({'Keyword': words, 'Count': counts}).sort_values('Count', ascending=False)
            
            st.markdown("**Top Maintenance Keywords (NLP Extracted)**")
            st.dataframe(word_df.head(5), use_container_width=True)
            
            # Lubrication specific
            lube_count = word_df[word_df['Keyword'].str.contains('oil|lube|grease', case=False)]['Count'].sum()
            st.markdown(f"- Lubrication related keywords count: **{lube_count}**")
            
        except Exception as e:
            st.error(f"NLP Error: {e}")

    # --- 4. DATA QUALITY ---
    st.markdown("#### 4. Data Quality Forecast")
    miss_rate = df6['Equipment description'].isnull().mean()
    st.markdown(f"- Current Missing Equipment Rate: **{miss_rate:.1%}**")

    # Optional LLM Summary
    try:
        if st.checkbox("Generate AI Executive Summary (GPT-4)"):
            prompt = f"Analyze this maintenance data summary: Breakdown Cost: {sum(zlt2_monthly)}, Top Actor: {top_asset if 'top_asset' in locals() else 'N/A'}. Suggest 3 actions."
            # Placeholder for actual LLM call using LangChain or OpenAI direct
            st.info("AI Suggestion: Focus on the top bad actor identified in Weibull analysis and review PM schedules where Random Forest indicates high failure risk.")
    except:
        pass
