# forecasting_engine.py
import pandas as pd
from prophet import Prophet
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import streamlit as st
# ... other imports ...
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
    Predict next months using linear regression.
    Require at least min_months months of history to produce forecasts.
    If insufficient data, returns None.
    """
    if months_ahead <= 0:
        return []
    if ms is None or len(ms) == 0:
        return None
    if len(ms) < min_months:
        return None
    if len(ms) == 1:
        return [float(ms.iloc[0])] * months_ahead
    x = np.arange(len(ms)).reshape(-1, 1)
    y = ms.values
    model = LinearRegression().fit(x, y)
    preds = []
    start = len(ms)
    for i in range(months_ahead):
        preds.append(float(model.predict(np.array([[start + i]]))))
    return preds

def _plot_history_and_forecast(ms: pd.Series, preds, title: str, y_label: str = "Value"):
    """
    ms: pd.Series indexed by Period('M'), numeric values (historical monthly sums)
    preds: list of forecast numbers (length months_ahead) or None
    This function plots historical monthly points (month-end) and a daily-resolution forecast line (red dotted)
    that interpolates between the last historical month-end and the forecast month-end points.
    """
    fig = go.Figure()
    # Historical
    if ms is not None and len(ms) > 0:
        x_hist = [p.to_timestamp(how='end') for p in ms.index]
        y_hist = ms.values
        fig.add_trace(go.Scatter(x=x_hist, y=y_hist, mode='lines+markers', name='Historical', line=dict(color='#1f77b4', width=3)))
    else:
        x_hist = []
        y_hist = np.array([])

    # Forecast
    if preds is not None and len(preds) > 0:
        # build forecast period-ends
        if len(x_hist) > 0:
            last_period = ms.index.max()
            forecast_periods = [last_period + (i+1) for i in range(len(preds))]
            forecast_ts = [p.to_timestamp(how='end') for p in forecast_periods]
            # numeric points for interpolation
            x_points = np.array([dt.timestamp() for dt in (x_hist + forecast_ts)])
            y_points = np.concatenate([y_hist, np.array(preds)])
            # create daily timeline from last historical day+1 to final forecast end
            start_date = x_hist[-1] + pd.Timedelta(days=1)
            end_date = forecast_ts[-1]
            if start_date <= end_date:
                x_daily = pd.date_range(start=start_date, end=end_date, freq='D').to_pydatetime().tolist()
                x_daily_num = np.array([d.timestamp() for d in x_daily])
                # interpolation (linear)
                y_daily = np.interp(x_daily_num, x_points, y_points)
                # plot forecast dotted red line (daily)
                fig.add_trace(go.Scatter(x=x_daily, y=y_daily, mode='lines', name='Forecast (daily)', line=dict(color='red', dash='dot')))
            # also add markers at forecast month-ends
            fig.add_trace(go.Scatter(x=forecast_ts, y=preds, mode='markers+lines', name='Forecast (month-end)', line=dict(color='red', dash='dot'), marker=dict(symbol='circle', size=8, color='red')))
        else:
            # No historical data: create forecast points anchored at month-ends from today
            today = pd.Timestamp(datetime.now().date())
            forecast_ts = [(today + relativedelta(months=i+1)).to_pydatetime() for i in range(len(preds))]
            # daily series from tomorrow to final forecast end
            start_date = today + pd.Timedelta(days=1)
            end_date = forecast_ts[-1]
            x_daily = pd.date_range(start=start_date, end=end_date, freq='D').to_pydatetime().tolist()
            # linear interpolation between 0 and forecast month-end values (simple ramp)
            x_points = np.array([start_date.timestamp()] + [dt.timestamp() for dt in forecast_ts])
            y_points = np.concatenate([[0.0], np.array(preds)])
            x_daily_num = np.array([d.timestamp() for d in x_daily])
            if len(x_daily_num) > 0:
                y_daily = np.interp(x_daily_num, x_points, y_points)
                fig.add_trace(go.Scatter(x=x_daily, y=y_daily, mode='lines', name='Forecast (daily)', line=dict(color='red', dash='dot')))
            fig.add_trace(go.Scatter(x=forecast_ts, y=preds, mode='markers+lines', name='Forecast (month-end)', line=dict(color='red', dash='dot'), marker=dict(symbol='circle', size=8, color='red')))

    fig.update_layout(title=title, xaxis_title='Date', yaxis_title=y_label, template='plotly_white', hovermode='x unified')
    return fig

# --- DEEP 6-MONTH AI ANALYZER (NEW MODE) ---
def deep_six_month_analyzer(df):
    st.markdown('<div class="ai-banner">AI Analyzer — Deep Six-Month Forecast (Adjustable Horizon)</div>', unsafe_allow_html=True)
    with st.expander("ℹ️ About this AI Analyzer"):
        st.write("- Uses the last 6 months of data only (by Created_Date).")
        st.write("- Forecast horizon is configurable (Next 1 month, Next 2 months, Next 3 months). Minimum 3 monthly historical points are required for linear forecasting where noted.")
        st.write("- All charts are interactive; select items from tables to view detailed drilldowns.")

    if 'Created_Date' not in df.columns:
        st.error("Created_Date column required. Ensure files contain 'Created On' so Created_Date exists after load.")
        return

    # Forecast horizon selector (New UI element)
    horizon_map = {"Next 1 month": 1, "Next 2 months": 2, "Next 3 months (Quarter)": 3}
    horizon_choice = st.selectbox("Forecast horizon:", list(horizon_map.keys()), index=2)
    months_ahead = horizon_map[horizon_choice]

    df = df.copy()
    df['Created_Date'] = pd.to_datetime(df['Created_Date'], errors='coerce')
    df = df.dropna(subset=['Created_Date'])
    if df.empty:
        st.warning("No valid Created_Date entries found.")
        return

    max_date = df['Created_Date'].max()
    cutoff = (max_date - relativedelta(months=6)).replace(day=1)
    df6 = df[df['Created_Date'] >= cutoff].copy()
    if df6.empty:
        st.warning("No data in the last 6 months range.")
        return

    text_col = 'Description' if 'Description' in df6.columns else None
    descriptions = df6[text_col].astype(str).fillna('') if text_col else pd.Series([], dtype=str)

    # Tokenization/keyword cost weighting (fixed sparse multiply bug)
    token_df = pd.DataFrame(columns=['token', 'total_cost', 'count'])
    if text_col and not descriptions.empty:
        try:
            vectorizer = CountVectorizer(stop_words='english', token_pattern=r'(?u)\b[a-zA-Z]{3,}\b')
            X = vectorizer.fit_transform(descriptions)            # shape: (n_rows, n_tokens)
            tokens = vectorizer.get_feature_names_out()
            costs = df6['TotSum (actual)'].fillna(0).values       # shape: (n_rows,)
            # compute total cost per token using X.T.dot(costs)
            token_costs = np.asarray(X.T.dot(costs)).ravel()      # shape: (n_tokens,)
            token_counts = np.asarray(X.sum(axis=0)).ravel()      # shape: (n_tokens,)
            token_df = pd.DataFrame({'token': tokens, 'total_cost': token_costs, 'count': token_counts}).sort_values('total_cost', ascending=False)
        except Exception as e:
            st.info("Tokenization or token-cost weighting failed: " + str(e))
            token_df = pd.DataFrame(columns=['token', 'total_cost', 'count'])

    st.markdown("#### 1. Financial & Budget Forecasting")

    # 1.1 Monthly Breakdown Cost Forecast (ZLT2)
    zlt2_df = df6[df6.get('Order Type', '') == 'Breakdown maintenance'] if 'Order Type' in df6.columns else df6.iloc[0:0]
    zlt2_monthly = monthly_series(zlt2_df, value_col='TotSum (actual)') if not zlt2_df.empty else pd.Series(dtype=float)
    zlt2_preds = predict_linear_monthly(zlt2_monthly, months_ahead=months_ahead, min_months=3)
    st.markdown("**1.1 Monthly Breakdown Cost Forecast**")
    if zlt2_preds is None:
        st.warning("Not enough monthly ZLT2 history (minimum 3 monthly points required) to produce a reliable forecast. Showing historical totals where available.")
        if not zlt2_monthly.empty:
            fig = _plot_history_and_forecast(zlt2_monthly, None, "ZLT2 Historical Monthly Costs", "LKR")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No ZLT2 data available.")
    else:
        out1_1 = pd.DataFrame({
            'Month': [(max_date + relativedelta(months=i+1)).strftime('%Y-%m') for i in range(months_ahead)],
            'Predicted_Breakdown_Cost_LKR': [max(0.0, p) for p in zlt2_preds]
        })
        st.dataframe(out1_1.style.format({"Predicted_Breakdown_Cost_LKR": "LKR {:,.2f}"}), use_container_width=True)
        fig = _plot_history_and_forecast(zlt2_monthly, zlt2_preds, "ZLT2 Historical & Forecasted Monthly Costs", "LKR")
        st.plotly_chart(fig, use_container_width=True)

    # 1.2 Annual Budget Deviation Forecast (line chart with projection)
    st.markdown("**1.2 Annual Budget Deviation Forecast**")
    df6['Cost_Variance'] = df6.get('Cost_Variance', df6.get('TotSum (actual)', 0) - df6.get('TotSum (plan)', 0))
    var_monthly = df6.groupby(df6['Created_Date'].dt.to_period('M'))['Cost_Variance'].sum().sort_index()
    last_month = max_date.month
    months_remaining = 12 - last_month
    projected_remaining = 0.0
    var_preds = None
    if months_remaining > 0 and len(var_monthly) >= 3:
        var_preds = predict_linear_monthly(var_monthly, months_ahead=months_remaining, min_months=3)
        if var_preds is not None:
            projected_remaining = sum(var_preds)
    elif months_remaining > 0 and len(var_monthly) > 0:
        projected_remaining = var_monthly.mean() * months_remaining
    current_year_variance = var_monthly.sum()
    annual_overrun_forecast = current_year_variance + projected_remaining
    st.markdown(f"- Forecasted maintenance cost overrun (Actual vs Plan) for current year: LKR {annual_overrun_forecast:,.2f}")

    # plot monthly variance with projection to year end
    fig_var = _plot_history_and_forecast(var_monthly, var_preds, "Monthly Cost Variance (Actual - Plan) and Projection to Year End", "LKR")
    st.plotly_chart(fig_var, use_container_width=True)

    # Provide drilldown: select month to view raw rows
    if not var_monthly.empty:
        with st.expander("🔎 Explore monthly variance details"):
            month_options = [m.strftime('%Y-%m') for m in var_monthly.index.to_timestamp()]
            sel_month = st.selectbox("Select month", month_options)
            sel_period = pd.Period(sel_month, freq='M')
            rows = df6[df6['Created_Date'].dt.to_period('M') == sel_period]
            st.dataframe(rows.sort_values('Created_Date', ascending=False), use_container_width=True)
            if 'Equipment description' in rows.columns:
                sel_eq = st.selectbox("Select equipment from this month to view details", sorted(rows['Equipment description'].dropna().unique().tolist()))
                if sel_eq:
                    if st.button("View Equipment History (selected)"):
                        show_machine_details(sel_eq, df6)

    # 1.3 Departmental Spending Forecast (with line charts)
    st.markdown("**1.3 Departmental Spending Forecast (Mechanical vs Electrical)**")
    mech = df6[df6.get('Main WorkCtr', '') == 'Mechanical Dpt.'] if 'Main WorkCtr' in df6.columns else df6.iloc[0:0]
    elec = df6[df6.get('Main WorkCtr', '') == 'Electrical Dpt.'] if 'Main WorkCtr' in df6.columns else df6.iloc[0:0]
    mech_monthly = monthly_series(mech) if not mech.empty else pd.Series(dtype=float)
    elec_monthly = monthly_series(elec) if not elec.empty else pd.Series(dtype=float)
    mech_preds = predict_linear_monthly(mech_monthly, months_ahead=months_ahead, min_months=3)
    elec_preds = predict_linear_monthly(elec_monthly, months_ahead=months_ahead, min_months=3)
    if mech_preds is None or elec_preds is None:
        st.warning("Not enough departmental monthly history (minimum 3 months) to produce reliable dept forecasts; using simple averages where necessary.")
    mech_total = sum(max(0, p) for p in mech_preds) if mech_preds is not None else (mech_monthly.mean() * months_ahead if not mech_monthly.empty else 0.0)
    elec_total = sum(max(0, p) for p in elec_preds) if elec_preds is not None else (elec_monthly.mean() * months_ahead if not elec_monthly.empty else 0.0)
    st.markdown(f"- Mechanical forecast for selected horizon: LKR {mech_total:,.2f}")
    st.markdown(f"- Electrical forecast for selected horizon: LKR {elec_total:,.2f}")
    winner = "Mechanical Dpt." if mech_total > elec_total else "Electrical Dpt." if elec_total > mech_total else "Tie"
    st.markdown(f"**Projected higher increase:** {winner}")

    # Plot departmental historical + forecast lines
    fig_mech = _plot_history_and_forecast(mech_monthly, mech_preds, "Mechanical Dept: Historical & Forecasted Spend", "LKR")
    fig_elec = _plot_history_and_forecast(elec_monthly, elec_preds, "Electrical Dept: Historical & Forecasted Spend", "LKR")
    st.plotly_chart(fig_mech, use_container_width=True)
    st.plotly_chart(fig_elec, use_container_width=True)

    # 1.4 Major Spare Parts Cost Prediction (bar chart + select for details)
    st.markdown("**1.4 Major Spare Parts Cost Prediction (Top tokens)**")
    likely_spare_tokens = [k for k in ['bearing','seal','motor','valve','gear','pump','shaft','coupling','belt','sensor'] if k in token_df['token'].tolist()]
    top_spares = token_df[token_df['token'].isin(likely_spare_tokens)].head(5)
    if top_spares.empty and not token_df.empty:
        top_spares = token_df.head(5)
    spare_forecasts = []
    for tok in top_spares['token'].tolist():
        mask = df6[text_col].str.contains(r'\b' + re.escape(tok) + r'\b', case=False, na=False) if text_col else pd.Series(False, index=df6.index)
        tok_monthly = monthly_series(df6[mask], value_col='TotSum (actual)') if mask.any() else pd.Series(dtype=float)
        preds = predict_linear_monthly(tok_monthly, months_ahead=months_ahead, min_months=3)
        if preds is None:
            next_q_spend = tok_monthly.mean() * months_ahead if not tok_monthly.empty else 0.0
        else:
            next_q_spend = sum(max(0, p) for p in preds)
        spare_forecasts.append({'keyword': tok, 'next_period_spend': next_q_spend, 'historical_total': tok_monthly.sum() if not tok_monthly.empty else 0.0})
    spare_df = pd.DataFrame(spare_forecasts).sort_values('next_period_spend', ascending=False)
    if not spare_df.empty:
        st.dataframe(spare_df.style.format({"next_period_spend": "LKR {:,.2f}", "historical_total": "LKR {:,.2f}"}), use_container_width=True)
        st.plotly_chart(px.bar(spare_df, x='keyword', y='next_period_spend', title="Forecasted Spend on Top Spare Keywords", labels={'next_period_spend':'LKR'}), use_container_width=True)
        sel_spare = st.selectbox("Select spare keyword to view related records", spare_df['keyword'].tolist())
        if sel_spare:
            related_rows = df6[df6['Description'].astype(str).str.contains(r'\b' + re.escape(sel_spare) + r'\b', case=False, na=False)]
            st.markdown(f"Related records count: {len(related_rows)}")
            st.dataframe(related_rows[['Created_Date','Order','Description','Equipment description','TotSum (actual)']].sort_values('Created_Date', ascending=False), use_container_width=True)
            if 'Equipment description' in related_rows.columns:
                equipment_options = sorted(related_rows['Equipment description'].dropna().unique().tolist())
                if equipment_options:
                    sel_eq = st.selectbox("Select equipment (from related rows) to view full history", equipment_options, key=f"spare_eq_{sel_spare}")
                    if sel_eq and st.button("View Equipment History (spare selection)"):
                        show_machine_details(sel_eq, df6)
    else:
        st.write("No spare-part related tokens detected in Description.")

    # 1.5 Cost per Asset Trend (bar + clickable)
    st.markdown("**1.5 Cost per Asset Trend (Top 10 machines)**")
    if 'Equipment description' in df6.columns:
        asset_costs = df6.groupby('Equipment description')['TotSum (actual)'].sum().sort_values(ascending=False)
        top10 = asset_costs.head(10).index.tolist()
        asset_forecasts = []
        for a in top10:
            a_df = df6[df6['Equipment description'] == a]
            a_monthly = monthly_series(a_df)
            preds = predict_linear_monthly(a_monthly, months_ahead=months_ahead, min_months=3)
            if preds is None:
                est = a_monthly.mean() * months_ahead if not a_monthly.empty else 0.0
            else:
                est = sum(max(0, p) for p in preds)
            asset_forecasts.append({'Machine': a, 'next_period_cost': est})
        af_df = pd.DataFrame(asset_forecasts).sort_values('next_period_cost', ascending=False)
        st.dataframe(af_df.style.format({"next_period_cost": "LKR {:,.2f}"}), use_container_width=True)
        st.plotly_chart(px.bar(af_df, x='Machine', y='next_period_cost', title=f"Top {len(af_df)} Machines: Forecasted Spend for Selected Horizon", labels={'next_period_cost':'LKR'}), use_container_width=True)
        sel_machine = st.selectbox("Select machine to view full history", af_df['Machine'].tolist())
        if sel_machine:
            if st.button("View Machine History (from asset forecast)"):
                show_machine_details(sel_machine, df6)
    else:
        st.write("No Equipment description column to evaluate assets.")

    st.markdown("#### 2. Equipment & Reliability Forecasting")

    # 2.1 MTBF Prediction for Top 3 (by ZLT2 count) with timeline chart
    st.markdown("**2.1 MTBF Prediction (Top 3 assets by ZLT2 count)**")
    if 'Order Type' in df6.columns and 'Equipment description' in df6.columns:
        zlt2_counts = df6[df6['Order Type'] == 'Breakdown maintenance'].groupby('Equipment description').size().sort_values(ascending=False)
        top3 = zlt2_counts.head(3).index.tolist()
        mtbf_results = []
        for mach in top3:
            sub = df6[(df6['Equipment description'] == mach) & (df6['Order Type'] == 'Breakdown maintenance')].sort_values('Created_Date')
            if len(sub) > 1:
                diffs = sub['Created_Date'].diff().dt.days.dropna()
                mean_days = diffs.mean()
                last_date = sub['Created_Date'].max()
                low = max(1, int(mean_days * 0.8))
                high = max(low, int(mean_days * 1.2))
                next_low = (last_date + pd.Timedelta(days=low)).date()
                next_high = (last_date + pd.Timedelta(days=high)).date()
                mtbf_results.append({'Machine': mach, 'mtbf_days': float(mean_days), 'next_failure_window': f"{next_low} to {next_high}"})
        if mtbf_results:
            st.dataframe(pd.DataFrame(mtbf_results), use_container_width=True)
            # timeline scatter of last events for top3
            timeline_df = df6[df6['Equipment description'].isin(top3) & (df6.get('Order Type','') == 'Breakdown maintenance')]
            if not timeline_df.empty:
                timeline_df = timeline_df.sort_values('Created_Date')
                fig_t = px.scatter(timeline_df, x='Created_Date', y='Equipment description', color='Equipment description', size='TotSum (actual)' if 'TotSum (actual)' in timeline_df.columns else None, title="Breakdown Events Timeline for Top 3 ZLT2 Assets", hover_data=['Description','Order'])
                st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.write("Not enough breakdown history to compute MTBF for top assets.")
    else:
        st.write("Order Type or Equipment description columns missing for MTBF analysis.")

    # (2.2 Failure Mode Seasonality removed as requested)

    # 2.3 Expected Equipment Life Remaining
    st.markdown("**2.3 Expected Equipment Life Remaining (qualitative)**")
    if 'Equipment description' in df6.columns:
        recent_cost = df6.groupby('Equipment description')['TotSum (actual)'].sum()
        first_seen = df6.groupby('Equipment description')['Created_Date'].min()
        candidates = pd.DataFrame({'first_seen': first_seen, 'recent_cost': recent_cost}).sort_values('recent_cost', ascending=False)
        if not candidates.empty:
            a = candidates.index[0]
            a_monthly = monthly_series(df6[df6['Equipment description'] == a])
            slope = 0.0
            if len(a_monthly) >= 2:
                x = np.arange(len(a_monthly)).reshape(-1, 1)
                slope = LinearRegression().fit(x, a_monthly.values).coef_[0]
            if slope > 0 and (a_monthly.mean() != 0 and slope / (a_monthly.mean()) > 0.2):
                rul = "Less than 6 months RUL"
            elif slope > 0:
                rul = "6-12 months RUL"
            else:
                rul = "More than 12 months RUL"
            st.markdown(f"- Asset prioritized: **{a}**")
            st.markdown(f"- Estimated Remaining Useful Life (qualitative): **{rul}**")
            # show trend chart for this asset
            asset_ts = monthly_series(df6[df6['Equipment description'] == a])
            preds_asset = predict_linear_monthly(asset_ts, months_ahead=months_ahead, min_months=3)
            fig_asset_trend = _plot_history_and_forecast(asset_ts, preds_asset, f"Cost Trend for {a}", "LKR")
            st.plotly_chart(fig_asset_trend, use_container_width=True)
        else:
            st.write("Insufficient asset cost data for RUL estimation.")
    else:
        st.write("No Equipment description column.")

    # 2.4 PM Effectiveness Failure Forecast
    st.markdown("**2.4 PM Effectiveness Failure Forecast**")
    if 'Order Type' in df6.columns and 'Equipment description' in df6.columns:
        pm_df = df6[df6['Order Type'] == 'Preventive maintenance']
        breakdown_df = df6[df6['Order Type'] == 'Breakdown maintenance']
        pm_count = len(pm_df)
        if pm_count > 0:
            pm_followed_break = 0
            for idx, pm in pm_df.iterrows():
                eq = pm.get('Equipment description')
                pm_date = pm.get('Created_Date')
                window = breakdown_df[(breakdown_df['Equipment description'] == eq) & (breakdown_df['Created_Date'] > pm_date) & (breakdown_df['Created_Date'] <= pm_date + pd.Timedelta(days=30))]
                if not window.empty:
                    pm_followed_break += 1
            rate = pm_followed_break / pm_count
            avg_monthly_pms = pm_df.groupby(pm_df['Created_Date'].dt.to_period('M')).size().mean() if not pm_df.empty else 0
            expected_pms_next_period = (avg_monthly_pms * months_ahead) if not np.isnan(avg_monthly_pms) else 0
            expected_poor_pm_breakdowns = rate * expected_pms_next_period
            st.markdown(f"- Historical PMs: {pm_count}, PMs followed by breakdown within 30 days: {pm_followed_break}")
            st.markdown(f"- Forecasted breakdowns within 1 month after PM for selected horizon: {expected_poor_pm_breakdowns:.1f}")
            # Pie chart of PM outcomes if possible (fixed column naming)
            pm_outcomes = pd.Series(['Followed by Breakdown' if (breakdown_df[(breakdown_df['Equipment description'] == pm.get('Equipment description')) & (breakdown_df['Created_Date'] > pm.get('Created_Date')) & (breakdown_df['Created_Date'] <= pm.get('Created_Date') + pd.Timedelta(days=30))].shape[0] > 0) else 'No Breakdown' for _, pm in pm_df.iterrows()])
            if not pm_outcomes.empty:
                pm_counts = pm_outcomes.value_counts().reset_index()
                pm_counts.columns = ['Outcome', 'Count']
                st.plotly_chart(px.pie(pm_counts, names='Outcome', values='Count', title="PM Outcomes (last 6 months)"), use_container_width=True)
        else:
            st.write("No PM (ZLT4) records in the last 6 months to evaluate.")
    else:
        st.write("Order Type or Equipment description missing for PM effectiveness.")

    # 2.5 Lubrication Failure Prediction
    st.markdown("**2.5 Lubrication Failure Prediction**")
    bearing_mask = df6['Description'].astype(str).str.contains(r'\bbearing\b', case=False, na=False) if 'Description' in df6.columns else pd.Series(False, index=df6.index)
    trip_mask = df6['Description'].astype(str).str.contains(r'\btrip\b|\btripped\b', case=False, na=False) if 'Description' in df6.columns else pd.Series(False, index=df6.index)
    related_mask = bearing_mask | trip_mask
    related_count = int(related_mask.sum()) if hasattr(related_mask, 'sum') else 0
    overall_count = len(df6)
    related_prop = related_count / overall_count if overall_count > 0 else 0
    related_monthly = df6[related_mask].groupby(df6['Created_Date'].dt.to_period('M')).size() if related_count > 0 else pd.Series(dtype=float)
    if len(related_monthly) >= 2:
        recent = related_monthly.iloc[-1]
        prev = related_monthly.iloc[-2] if len(related_monthly) >= 2 else recent
        pct_change = ((recent - prev) / prev) * 100 if prev != 0 else 0.0
    else:
        pct_change = 0.0
    suggested_change_pct = max(-50, min(200, pct_change))
    st.markdown(f"- Bearing/Motor related events proportion (last 6 months): {related_prop:.2%}")
    st.markdown(f"- Month-over-month change (most recent): {pct_change:.1f}%")
    st.markdown(f"- Suggested change in lubricant purchases for next period: {suggested_change_pct:.1f}%")
    if not related_monthly.empty:
        fig_rel = _plot_history_and_forecast(related_monthly, predict_linear_monthly(related_monthly, months_ahead=months_ahead, min_months=3), "Bearing/Trip Related Events: Historical & Forecast", "Events")
        st.plotly_chart(fig_rel, use_container_width=True)

    st.markdown("#### 3. Workload & Resource Forecasting")

    # 3.1 Total Order Volume Forecast
    st.markdown("**3.1 Total Order Volume Forecast**")
    vol_monthly = df6.groupby(df6['Created_Date'].dt.to_period('M')).size().sort_index()
    vol_preds = predict_linear_monthly(vol_monthly, months_ahead=months_ahead, min_months=3)
    if vol_preds is None:
        st.warning("Not enough monthly volume history (minimum 3 months) to make a reliable forecast. Using average instead.")
        vol_pred_val = int(np.round(vol_monthly.mean())) if not vol_monthly.empty else len(df6)
        fig_vol = _plot_history_and_forecast(vol_monthly, None, "Order Volume (Historical)")
    else:
        vol_pred_val = int(np.round(sum(vol_preds))) if months_ahead > 1 else int(np.round(vol_preds[0]))
        fig_vol = _plot_history_and_forecast(vol_monthly, vol_preds, "Order Volume: Historical & Forecast", "Orders")
    st.markdown(f"- Predicted total number of work orders for selected horizon: **{vol_pred_val}**")
    st.plotly_chart(fig_vol, use_container_width=True)

    # 3.2 Craft/Skill Demand (Welding)
    st.markdown("**3.2 Craft/Skill Demand — Welding hours (selected horizon)**")
    if text_col:
        welding_mask = df6['Description'].astype(str).str.contains(r'\bweld|welding\b', case=False, na=False)
        welding_monthly = welding_mask.groupby(df6['Created_Date'].dt.to_period('M')).sum()
        welding_preds = predict_linear_monthly(welding_monthly, months_ahead=months_ahead, min_months=3)
        if welding_preds is None:
            welding_pred_orders = int(np.round(welding_monthly.mean() * months_ahead)) if not welding_monthly.empty else 0
        else:
            welding_pred_orders = int(np.round(sum(welding_preds)))
        avg_hours_per_welding = 4
        welding_hours_next_period = welding_pred_orders * avg_hours_per_welding
        st.markdown(f"- Estimated welding maintenance hours needed for selected horizon: **{welding_hours_next_period:.0f} hours** (assumed {avg_hours_per_welding}h per welding order)")
        if not welding_monthly.empty:
            fig_weld = _plot_history_and_forecast(welding_monthly, welding_preds, "Welding Order Count: Historical & Forecast", "Orders")
            st.plotly_chart(fig_weld, use_container_width=True)
    else:
        st.write("No Description column to analyze craft-specific keywords.")

    # 3.3 Manpower Utilization Forecast
    st.markdown("**3.3 Manpower Utilization Forecast (selected horizon)**")
    tech_count = df6['Entered By'].nunique() if 'Entered By' in df6.columns and df6['Entered By'].nunique() > 0 else 10
    avg_hours_per_order = 3
    vol_preds_q = predict_linear_monthly(vol_monthly, months_ahead=months_ahead, min_months=3)
    if vol_preds_q is None:
        predicted_orders_q = int(round(vol_monthly.mean() * months_ahead)) if not vol_monthly.empty else len(df6)
    else:
        predicted_orders_q = int(round(sum(vol_preds_q)))
    required_hours_q = predicted_orders_q * avg_hours_per_order
    available_hours_q = tech_count * 8 * 22 * months_ahead
    utilization_pct = min(100.0, (required_hours_q / available_hours_q) * 100) if available_hours_q > 0 else 0.0
    st.markdown(f"- Estimated technician count: {tech_count}")
    st.markdown(f"- Forecasted manpower utilization for selected horizon: **{utilization_pct:.1f}%** (based on {avg_hours_per_order}h/order)")

    # 3.4 Daily Order Creation Volatility
    st.markdown("**3.4 Daily Order Creation Volatility — Peak Day**")
    if 'Created_Date' in df6.columns:
        day_counts = df6.groupby(df6['Created_Date'].dt.day_name()).size().sort_values(ascending=False)
        peak_day = day_counts.idxmax() if not day_counts.empty else "N/A"
        peak_count = int(day_counts.max()) if not day_counts.empty else 0
        st.markdown(f"- Peak day of the week (persistent): **{peak_day}** with average {peak_count} orders on peak days.")
        if not day_counts.empty:
            day_df = day_counts.reset_index()
            day_df.columns = ['Day', 'Count']
            st.plotly_chart(px.bar(day_df, x='Day', y='Count', title="Orders by Day of Week"), use_container_width=True)
    else:
        st.write("No Created_Date column to compute daily volatility.")

    # 3.5 Backlog Duration Forecast
    st.markdown("**3.5 Backlog Duration Forecast**")
    if 'User Status Text' in df6.columns and 'Created_Date' in df6.columns and 'Order' in df6.columns:
        awaiting = df6[df6['User Status Text'].astype(str).str.contains('Awaiting', case=False, na=False)]
        closed = df6[df6['User Status Text'].astype(str).str.contains('Closed', case=False, na=False)]
        if not awaiting.empty and not closed.empty:
            merged = pd.merge(awaiting[['Order','Created_Date']], closed[['Order','Created_Date']], on='Order', suffixes=('_await','_closed'))
            merged = merged[merged['Created_Date_closed'] >= merged['Created_Date_await']]
            merged['delta_days'] = (merged['Created_Date_closed'] - merged['Created_Date_await']).dt.days
            avg_delta = merged['delta_days'].mean() if not merged.empty else np.nan
            if not np.isnan(avg_delta):
                st.markdown(f"- Forecast average time for non-critical order to move from 'Awaiting Approval' to 'Closed': **{avg_delta:.1f} days**")
                st.plotly_chart(px.histogram(merged, x='delta_days', nbins=20, title="Awaiting -> Closed: Distribution of Days"), use_container_width=True)
            else:
                st.write("- Insufficient data to calculate backlog duration for non-critical orders.")
        else:
            st.write("Insufficient awaiting/closed transition data to compute backlog duration.")
    else:
        st.write("User Status Text, Created_Date or Order missing for backlog duration analysis.")

    st.markdown("#### 4. Data & Process Improvement Forecasting")

    # 4.1 Missing Data Rate Forecast
    st.markdown("**4.1 Missing Data Rate Forecast (Equipment or Functional Loc.)**")
    miss_equipment_rate = df6['Equipment'].isnull().mean() if 'Equipment' in df6.columns else 0.0
    miss_funcloc_rate = df6['Functional Loc.'].isnull().mean() if 'Functional Loc.' in df6.columns else 0.0
    st.markdown(f"- Predicted missing Equipment rate next period: **{miss_equipment_rate:.1%}**")
    st.markdown(f"- Predicted missing Functional Loc. rate next period: **{miss_funcloc_rate:.1%}**")
    # bar chart
    miss_df = pd.DataFrame({
        'Field': ['Equipment', 'Functional Loc.'],
        'MissingRate': [miss_equipment_rate, miss_funcloc_rate]
    })
    st.plotly_chart(px.bar(miss_df, x='Field', y='MissingRate', title="Missing Data Rates (last 6 months)", labels={'MissingRate':'Missing Rate'}), use_container_width=True)

    # 4.2 PM Compliance Rate Forecast
    st.markdown("**4.2 PM Compliance Rate Forecast (on-time completion of ZLT4)**")
    if 'Order Type' in df6.columns and 'Response_Time' in df6.columns:
        pm = df6[df6['Order Type'] == 'Preventive maintenance']
        if not pm.empty:
            on_time_rate = (pm['Response_Time'] <= 7).mean()
            pm_monthly_on_time = pm.groupby(pm['Created_Date'].dt.to_period('M')).apply(lambda x: (x['Response_Time']<=7).mean())
            projected_rate = on_time_rate
            if len(pm_monthly_on_time) >= 3:
                slope = np.polyfit(np.arange(len(pm_monthly_on_time)), pm_monthly_on_time.values, 1)[0]
                projected_rate = max(0, min(1, pm_monthly_on_time.iloc[-1] + slope * months_ahead))
            st.markdown(f"- Forecast PM on-time completion for selected horizon: **{projected_rate:.1%}**")
            if not pm_monthly_on_time.empty:
                fig_pm = _plot_history_and_forecast(pm_monthly_on_time, None, "PM On-time Rate (Historical)", "On-time Rate")
                st.plotly_chart(fig_pm, use_container_width=True)
        else:
            st.write("No PM (ZLT4) records in the last 6 months.")
    else:
        st.write("Order Type or Response_Time missing for PM compliance calculation.")

    # 4.3 Time-to-Approve Forecast (hours)
    st.markdown("**4.3 Time-to-Approve Forecast (average hours to move out of 'Awaiting Approval')**")
    if 'User Status Text' in df6.columns and 'Created_Date' in df6.columns:
        awaiting_mask = df6['User Status Text'].astype(str).str.contains('Awaiting', case=False, na=False)
        awaiting_df = df6[awaiting_mask]
        if not awaiting_df.empty and 'Response_Time' in awaiting_df.columns:
            avg_days = awaiting_df['Response_Time'].mean()
            avg_hours = avg_days * 24
            st.markdown(f"- Expected time-to-approve next period (avg): **{avg_hours:.1f} hours**")
        else:
            st.write("Insufficient 'Awaiting' records with Response_Time to estimate time-to-approve.")
    else:
        st.write("User Status Text or Created_Date missing.")

    # 4.4 Top Generic Description Persistence
    st.markdown("**4.4 Top Generic Description Persistence**")
    if 'Description' in df6.columns:
        desc_counts = df6['Description'].astype(str).value_counts()
        top3_generic = desc_counts.head(3).index.tolist()
        total_desc = desc_counts.sum()
        persistence = []
        for d in top3_generic:
            share = desc_counts[d] / total_desc if total_desc > 0 else 0
            prob = min(100, share * 120)
            persistence.append({'Description': d, 'Current_Share_pct': share * 100, 'Persistence_pct_next_period': prob})
        pers_df = pd.DataFrame(persistence)
        st.dataframe(pers_df.style.format({"Current_Share_pct":"{:.1f}%","Persistence_pct_next_period":"{:.1f}%"}), use_container_width=True)
        sel_desc = st.selectbox("Select a top generic description to see records", top3_generic)
        if sel_desc:
            recs = df6[df6['Description'] == sel_desc]
            st.dataframe(recs[['Created_Date','Order','Equipment description','TotSum (actual)']].sort_values('Created_Date', ascending=False), use_container_width=True)
    else:
        st.write("No Description column available.")

    # 4.5 RCA Priority Forecast
    st.markdown("**4.5 Root Cause Analysis (RCA) Priority Forecast**")
    if 'Description' in df6.columns:
        if 'Order' in df6.columns:
            rc = df6.groupby('Description').agg(Count=('Order','count'), TotalCost=('TotSum (actual)','sum')).reset_index()
        else:
            rc = df6.groupby('Description').agg(Count=('Description','count'), TotalCost=('TotSum (actual)','sum')).reset_index()
        if not rc.empty:
            rc['score'] = rc['Count'] * (rc['TotalCost'].abs() + 1)
            rca_priority = rc.sort_values('score', ascending=False).iloc[0]
            st.markdown(f"- Priority RCA target for next period: **{rca_priority['Description']}**")
            st.markdown(f"  - Occurrences: {int(rca_priority['Count'])}, Total Cost: LKR {rca_priority['TotalCost']:,.2f}")
            st.dataframe(rc.sort_values('score', ascending=False).head(10), use_container_width=True)
            sel_rc = st.selectbox("Select event/description to view related records", rc['Description'].head(10).tolist())
            if sel_rc:
                rel = df6[df6['Description'] == sel_rc]
                st.dataframe(rel[['Created_Date','Order','Equipment description','TotSum (actual)']].sort_values('Created_Date', ascending=False), use_container_width=True)
        else:
            st.write("No descriptive events found to prioritize.")
    else:
        st.write("Description column not present for RCA identification.")

    # Optional LLM summary (safe/optional)
    try:
        with st.expander("🧠 AI Summary (optional)"):
            prompt_summary = ("You are a reliability engineer. Provide a concise 5-bullet executive summary "
                              "highlighting critical forecasts and recommended immediate actions from the analysis.")
            llm = ChatOpenAI(temperature=0.0, model="gpt-4", openai_api_key=OPENAI_API_KEY)
            summary = llm(prompt_summary)
            st.markdown(summary.content if hasattr(summary, 'content') else str(summary))
    except Exception:
        pass

