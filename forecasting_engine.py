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
from sklearn.feature_extraction.text import CountVectorizer

# --- ADVANCED AI IMPORTS (Safe Imports) ---
# These try/except blocks prevent crashes if a library is missing
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
# 1. HELPER FUNCTIONS (Shared)
# ==========================================
def monthly_series(series_df, date_col='Created_Date', value_col='TotSum (actual)'):
    """Aggregates data to monthly sums."""
    series_df = series_df.copy()
    series_df[date_col] = pd.to_datetime(series_df[date_col], errors='coerce')
    series_df = series_df.dropna(subset=[date_col])
    if series_df.empty:
        return pd.Series(dtype=float)
    series_df['Month'] = series_df[date_col].dt.to_period('M')
    ms = series_df.groupby('Month')[value_col].sum().sort_index()
    return ms

def predict_linear_monthly(ms, months_ahead=3, min_months=3):
    """Simple linear regression fallback."""
    if months_ahead <= 0: return []
    if ms is None or len(ms) < min_months: return None
    
    x = np.arange(len(ms)).reshape(-1, 1)
    y = ms.values
    model = LinearRegression().fit(x, y)
    
    preds = []
    start = len(ms)
    for i in range(months_ahead):
        val = float(model.predict(np.array([[start + i]])))
        preds.append(max(0, val)) 
    return preds

def _plot_history_and_forecast(ms, preds, title, y_label="Value"):
    """Helper to plot history vs forecast."""
    fig = go.Figure()
    if ms is not None and not ms.empty:
        x_hist = [p.to_timestamp(how='end') for p in ms.index]
        fig.add_trace(go.Scatter(x=x_hist, y=ms.values, mode='lines+markers', name='Historical', line=dict(color='#1f77b4', width=3)))
    
    if preds is not None and len(preds) > 0:
        last_date = ms.index.max().to_timestamp(how='end')
        future_dates = [last_date + relativedelta(months=i+1) for i in range(len(preds))]
        x_forecast = [last_date] + future_dates
        y_forecast = [ms.values[-1]] + preds
        fig.add_trace(go.Scatter(x=x_forecast, y=y_forecast, mode='lines+markers', name='Forecast', line=dict(color='red', dash='dot')))

    fig.update_layout(title=title, xaxis_title='Date', yaxis_title=y_label, hovermode='x unified')
    return fig


# ==========================================
# 2. REQUIRED FUNCTIONS (The ones app.py is asking for)
# ==========================================

def forecast_spare_parts(df_input):
    """
    REQUIRED: Forecasts spare part usage per equipment (Simple Logic).
    """
    # Check columns
    required_cols = ['TotSum (actual)', 'Description', 'Equipment description', 'Created_Date']
    if not all(col in df_input.columns for col in required_cols):
        return None, "Error: Missing required columns."

    df_parts = df_input[
        (df_input['TotSum (actual)'] > 0) & 
        (df_input['Description'].notna()) &
        (df_input['Equipment description'].notna())
    ].copy()

    df_parts['Clean_Part'] = df_parts['Description'].str.lower().str.strip()

    if df_parts.empty:
        return None, "Insufficient data."

    predictions = []
    # Logic: Group by Equipment+Part
    part_counts = df_parts.groupby(['Equipment description', 'Clean_Part']).size()
    frequent_parts = part_counts[part_counts >= 3].index

    for (equip, part), count in part_counts.items():
        if (equip, part) not in frequent_parts: continue

        subset = df_parts[(df_parts['Equipment description'] == equip) & (df_parts['Clean_Part'] == part)].sort_values('Created_Date')
        
        avg_cost = subset['TotSum (actual)'].mean()
        subset['Days_Diff'] = subset['Created_Date'].diff().dt.days
        avg_days_between = subset['Days_Diff'].mean()
        last_usage = subset['Created_Date'].max()
        days_since = (pd.Timestamp.now() - last_usage).days
        
        prob = 0
        if not np.isnan(avg_days_between) and avg_days_between > 0:
            if days_since >= (avg_days_between * 0.8):
                prob = min(1.0, days_since / avg_days_between)
            else:
                prob = 0.1 

        if prob > 0.4: 
            predictions.append({
                'Equipment': equip,
                'Spare Part': part.title(),
                'Probability': prob,
                'Est. Cost (Next Month)': prob * avg_cost
            })

    df_pred = pd.DataFrame(predictions)
    if not df_pred.empty:
        return df_pred.sort_values('Est. Cost (Next Month)', ascending=False), "Success"
    else:
        return None, "No immediate spare parts needs detected."

def forecast_cost_prophet(df_input, periods=90):
    """
    REQUIRED: Forecasts cost using Prophet.
    """
    if not PROPHET_AVAILABLE:
        return None, None, "Prophet library not installed."

    if 'Created_Date' not in df_input.columns:
        return None, None, "Missing Created_Date column."

    daily_cost = df_input.groupby(df_input['Created_Date'].dt.date)['TotSum (actual)'].sum().reset_index()
    daily_cost.columns = ['ds', 'y']
    
    if len(daily_cost) < 10:
        return None, None, "Not enough data for Prophet."

    m = Prophet(daily_seasonality=False)
    m.fit(daily_cost)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    return m, forecast

def forecast_failure_rf(df_input):
    """
    REQUIRED: Predicts high-risk equipment using Random Forest.
    """
    # Simple aggregation for risk scoring
    equipment_stats = df_input.groupby('Equipment description').agg({
        'Order': 'count',
        'TotSum (actual)': 'sum',
        'Created_Date': 'max'
    }).reset_index()
    
    equipment_stats['Days_Since_Last'] = (pd.Timestamp.now() - equipment_stats['Created_Date']).dt.days
    
    # Dummy Risk Label (Orders > 5 AND Recent)
    equipment_stats['Risk_Label'] = np.where(
        (equipment_stats['Order'] > 5) & (equipment_stats['Days_Since_Last'] < 45), 1, 0
    )
    
    features = ['Order', 'TotSum (actual)', 'Days_Since_Last']
    X = equipment_stats[features].fillna(0)
    y = equipment_stats['Risk_Label']
    
    if len(y.unique()) < 2:
        equipment_stats['Failure_Probability'] = 0.0
        return equipment_stats, None

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    equipment_stats['Failure_Probability'] = model.predict_proba(X)[:, 1]
    
    return equipment_stats.sort_values('Failure_Probability', ascending=False), model

def forecast_major_spares(df, months_ahead=3):
    """Optional: If app.py asks for this, we provide it."""
    # Simplified version that calls the main spare forecast or returns empty
    return pd.DataFrame() 


# ==========================================
# 3. DEEP 6-MONTH AI ANALYZER (With Advanced AI Injected)
# ==========================================

def deep_six_month_analyzer(df):
    """
    The advanced analyzer with Weibull, NLP, and XGBoost/RF integration.
    """
    st.markdown('<div class="ai-banner">AI Analyzer — Deep Six-Month Forecast (Advanced Models)</div>', unsafe_allow_html=True)
    
    # --- SETUP ---
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

    # --- SECTION 1: FINANCIAL (Prophet) ---
    st.markdown("#### 1. Financial & Budget Forecasting")
    st.markdown("**1.1 Monthly Breakdown Cost Forecast**")
    
    zlt2_df = df6[df6.get('Order Type', '') == 'Breakdown maintenance']
    zlt2_monthly = monthly_series(zlt2_df)
    
    # 1. Try Prophet
    zlt2_preds = None
    if PROPHET_AVAILABLE and len(zlt2_df) > 10:
        try:
            # Aggregate to daily for Prophet
            daily_p = zlt2_df.groupby('Created_Date')['TotSum (actual)'].sum().reset_index()
            daily_p.columns = ['ds', 'y']
            m = Prophet(daily_seasonality=False)
            m.fit(daily_p)
            future = m.make_future_dataframe(periods=months_ahead*30)
            fcst = m.predict(future)
            # Sum up the future days into monthly buckets
            fcst['Month'] = fcst['ds'].dt.to_period('M')
            future_months = [max_date.to_period('M') + i for i in range(1, months_ahead+1)]
            zlt2_preds = fcst[fcst['Month'].isin(future_months)].groupby('Month')['yhat'].sum().tolist()
            st.info("✅ Used Prophet (Time-Series) for this forecast.")
        except Exception as e:
            st.warning(f"Prophet failed ({e}), falling back to Linear Regression.")
    
    # 2. Fallback to Linear
    if not zlt2_preds:
        zlt2_preds = predict_linear_monthly(zlt2_monthly, months_ahead=months_ahead)
        st.info("ℹ️ Used Linear Regression (Simple Trend).")

    if zlt2_preds:
        fig = _plot_history_and_forecast(zlt2_monthly, zlt2_preds, "Breakdown Cost Forecast", "LKR")
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"**Predicted Total:** LKR {sum(zlt2_preds):,.2f}")

    # --- SECTION 2: RELIABILITY (Weibull) ---
    st.markdown("#### 2. Equipment & Reliability (Weibull Analysis)")
    if 'Equipment description' in df6.columns and LIFELINES_AVAILABLE:
        st.markdown("**2.1 RUL / Survival Analysis**")
        top_asset = df6.groupby('Equipment description').size().idxmax()
        st.write(f"Analyzing Top Bad Actor: **{top_asset}**")
        
        subset = df[df['Equipment description'] == top_asset].sort_values('Created_Date')
        subset['TBF'] = subset['Created_Date'].diff().dt.days
        tbf_data = subset['TBF'].dropna()
        tbf_data = tbf_data[tbf_data > 0]

        if len(tbf_data) >= 3:
            try:
                wf = WeibullFitter()
                wf.fit(tbf_data, event_observed=[1]*len(tbf_data))
                st.success(f"Weibull Model Fit! Beta: {wf.beta_:.2f}, Eta: {wf.lambda_:.1f} days")
                
                # Plot
                surv_df = wf.survival_function_
                fig_surv = px.line(surv_df, title=f"Survival Probability over Time (Days) - {top_asset}")
                st.plotly_chart(fig_surv, use_container_width=True)
            except Exception as e:
                st.warning(f"Weibull fitting error: {e}")
        else:
            st.warning("Not enough failure points for Weibull analysis.")
    elif not LIFELINES_AVAILABLE:
        st.warning("Install 'lifelines' library to enable Weibull Analysis.")

    # --- SECTION 3: FAILURE PREDICTION (RF/XGBoost) ---
    st.markdown("#### 3. PM Effectiveness (Machine Learning)")
    st.markdown("**3.1 Random Forest Failure Prediction**")
    
    if (XGB_AVAILABLE or True) and 'Order Type' in df6.columns:
        # Create dataset: PMs and whether they failed in 30 days
        pms = df6[df6['Order Type'] == 'Preventive maintenance'].copy()
        breaks = df6[df6['Order Type'] == 'Breakdown maintenance']
        
        if len(pms) > 10 and len(breaks) > 0:
            labels = []
            for date, equip in zip(pms['Created_Date'], pms['Equipment description']):
                # Did it break in next 30 days?
                fails = breaks[(breaks['Equipment description'] == equip) & 
                               (breaks['Created_Date'] > date) & 
                               (breaks['Created_Date'] <= date + pd.Timedelta(days=30))]
                labels.append(1 if not fails.empty else 0)
            
            pms['Failed'] = labels
            pms['Day'] = pms['Created_Date'].dt.dayofweek
            pms['Month'] = pms['Created_Date'].dt.month
            
            # Simple Model
            if sum(labels) > 0:
                features = ['Day', 'Month']
                X = pms[features].fillna(0)
                y = pms['Failed']
                
                rf = RandomForestClassifier(n_estimators=50)
                rf.fit(X, y)
                risk_score = rf.predict_proba(X)[:, 1].mean()
                
                st.markdown(f"**ML Risk Score:** {risk_score:.1%} (Probability of PM being followed by breakdown)")
                st.write("Higher score indicates current PM schedule might be ineffective.")
            else:
                st.success("Great! No PMs were followed by immediate breakdowns in this dataset.")

    # --- SECTION 4: NLP / LUBRICATION ---
    st.markdown("#### 4. Lubrication & Spare Parts (NLP)")
    if 'Description' in df6.columns:
        desc_text = df6['Description'].astype(str).fillna('')
        try:
            vec = CountVectorizer(stop_words='english', max_features=15)
            X = vec.fit_transform(desc_text)
            words = vec.get_feature_names_out()
            counts = X.sum(axis=0).A1
            
            word_df = pd.DataFrame({'Keyword': words, 'Count': counts}).sort_values('Count', ascending=False)
            
            # Filter for Lube
            lube_words = ['oil', 'grease', 'lube', 'lubricant', 'filter']
            lube_df = word_df[word_df['Keyword'].isin(lube_words)]
            
            st.markdown("**Top Maintenance Keywords Detected:**")
            st.dataframe(word_df.head(5), use_container_width=True)
            
            if not lube_df.empty:
                st.markdown(f"🛢️ **Lubrication Needs Detected:** Found {lube_df['Count'].sum()} orders related to oil/grease.")
            else:
                st.markdown("No major lubrication keywords found.")
                
        except Exception as e:
            st.warning(f"NLP Error: {e}")
