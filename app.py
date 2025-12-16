# Full updated file with fixes:
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import base64
import os
import numpy as np
import re
from datetime import datetime
from fpdf import FPDF
import tempfile
import time
from dateutil.relativedelta import relativedelta
import streamlit as st
# ... standard imports ...

# LINK TO YOUR NEW FILES
from data_handler import load_and_clean_data, ITEM_DB
from forecasting_engine import forecast_spare_parts, forecast_cost_prophet, forecast_failure_rf, deep_six_month_analyzer

# --- AI LIBRARIES ---
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

# Additional ML/NLP imports for advanced analyzer mode
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer

# --- CONFIGURATION ---
OPENAI_API_KEY = "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"



st.set_page_config(layout="wide", page_title="Maintenance Analyzer Lankatile PLC", page_icon="🏭")

# --- CSS STYLING ---
st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .watermark {
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1px;
            padding: 6px 10px;
            border-radius: 6px;
            display: inline-block;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            background: rgba(255,255,255,0.85);
            color: #111;
            font-family: 'Courier New', monospace;
        }
        @media (prefers-color-scheme: dark) {
            .watermark {
                background: rgba(0,0,0,0.75);
                color: #fff;
                box-shadow: 0 2px 6px rgba(255,255,255,0.03);
            }
        }
        .logo-container { 
            text-align: center; margin-bottom: 20px;
            padding: 15px; background: white; border-radius: 12px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .metric-card {
            background-color: white; padding: 15px; border-radius: 10px;
            border-left: 5px solid #3498db; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .ai-banner {
            background: linear-gradient(90deg,#0ea5a4,#06b6d4);
            color: white;
            padding: 14px;
            border-radius: 10px;
            font-weight: 700;
            margin-bottom: 18px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# --- PDF GENERATOR ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Maintenance Analyzer Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf(df):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    act = df['TotSum (actual)'].sum() if 'TotSum (actual)' in df.columns else 0
    plan = df['TotSum (plan)'].sum() if 'TotSum (plan)' in df.columns else 0
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "1. Financial Summary", 0, 1)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Total Actual Cost: LKR {act:,.2f}", 0, 1)
    pdf.cell(0, 10, f"Total Planned Cost: LKR {plan:,.2f}", 0, 1)
    pdf.cell(0, 10, f"Variance: LKR {act-plan:,.2f}", 0, 1)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "2. Top 5 Costliest Assets", 0, 1)
    pdf.set_font("Arial", size=10)
    if 'Equipment description' in df.columns:
        cost = df.groupby('Equipment description')['TotSum (actual)'].sum().sort_values(ascending=False).head(5)
        for name, val in cost.items():
            clean_name = str(name).encode('latin-1', 'replace').decode('latin-1')
            pdf.cell(0, 8, f" - {clean_name}: LKR {val:,.2f}", 0, 1)
    pdf.ln(5)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        pdf.output(tmp_file.name)
        return tmp_file.name

# --- DATA LOADER ---
@st.cache_data
def load_data(files):
    all_data = []
    removed_all = []

    for idx, file in enumerate(files):
        try:
            if file.name.endswith('.csv'):
                df_raw = pd.read_csv(file)
            else:
                df_raw = pd.read_excel(file)

            df = df_raw.copy()

            for c in ['TotSum (actual)', 'TotSum (plan)', 'Order']:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

            zlt_map = {
                'ZLT1': 'General maintenance',
                'ZLT2': 'Breakdown maintenance',
                'ZLT4': 'Preventive maintenance',
                'ZLT5': 'Vehicle maintenance'
            }
            if 'Order Type' in df.columns:
                df['Order Type'] = df['Order Type'].astype(str).map(zlt_map).fillna(df['Order Type'])

            if 'Main WorkCtr' in df.columns:
                df['Main WorkCtr'] = df['Main WorkCtr'].replace({
                    'MECH': 'Mechanical Dpt.', 
                    'ELEC': 'Electrical Dpt.',
                    'mech': 'Mechanical Dpt.',
                    'elec': 'Electrical Dpt.'
                })

            if 'Equipment' in df.columns:
                def clean_code(val):
                    try:
                        return str(int(float(val)))
                    except Exception:
                        return str(val).strip()
                df['Equip_Key'] = df['Equipment'].apply(clean_code)
                df['Mapped_Desc'] = df['Equip_Key'].map(ITEM_DB)
                if 'Equipment description' not in df.columns:
                    df['Equipment description'] = df['Mapped_Desc']
                else:
                    df['Equipment description'] = df['Equipment description'].fillna(df['Mapped_Desc'])
                df['Equipment description'] = df['Equipment description'].fillna("Unknown Asset")

            if 'Functional Loc.' in df.columns:
                df['Loc_Display'] = df['Functional Loc.']
            else:
                df['Loc_Display'] = ''

            if 'Location description' in df.columns:
                target_col = 'Location description'
            elif 'Description.1' in df.columns:
                target_col = 'Description.1'
            elif 'Functional Loc. Desc' in df.columns:
                target_col = 'Functional Loc. Desc'
            else:
                target_col = None

            if target_col:
                df['Loc_Display'] = df.apply(
                    lambda x: x[target_col] if (pd.notnull(x[target_col]) and str(x[target_col]) != '0') else x.get('Functional Loc.', ''),
                    axis=1
                )

            if 'Created On' in df.columns:
                df['Created_Date'] = pd.to_datetime(df['Created On'].astype(str).str.split('.').str[0], format='%Y%m%d', errors='coerce')
                df['Day_Name'] = df['Created_Date'].dt.day_name()
                df['Day_Num'] = df['Created_Date'].dt.date
                df['Month_Year'] = df['Created_Date'].dt.to_period('M')
            else:
                df['Created_Date'] = pd.NaT
                df['Day_Name'] = None
                df['Day_Num'] = None
                df['Month_Year'] = None

            if 'Bas. start date' in df.columns:
                df['Start_Date'] = pd.to_datetime(df['Bas. start date'], errors='coerce')
            else:
                df['Start_Date'] = pd.NaT

            if 'Start_Date' in df.columns and 'Created_Date' in df.columns:
                df['Response_Time'] = (df['Start_Date'] - df['Created_Date']).dt.days
                df['Response_Time'] = df['Response_Time'].fillna(0).clip(lower=0)
            else:
                df['Response_Time'] = 0

            df['Cost_Variance'] = df.get('TotSum (actual)', 0) - df.get('TotSum (plan)', 0)

            def categorize(row):
                d = str(row.get('Description', '')).lower()
                if 'pm' in d or 'preventive' in d:
                    return 'Preventive'
                return 'Corrective'
            df['Work_Type'] = df.apply(categorize, axis=1)

            df['File_Index'] = idx + 1

            neg_cols = [c for c in ['TotSum (actual)', 'TotSum (plan)', 'Order'] if c in df.columns]
            if neg_cols:
                neg_mask = pd.Series(False, index=df.index)
                for c in neg_cols:
                    neg_mask = neg_mask | (df[c] < 0)
                if neg_mask.any():
                    removed_rows = df.loc[neg_mask].copy()
                    removed_rows['Removed_Reason'] = 'Negative value in key numeric columns'
                    removed_rows['File_Index'] = idx + 1
                    removed_all.append(removed_rows)
                    df = df.loc[~neg_mask].copy()

            all_data.append(df)

        except Exception as e:
            st.error(f"Error loading {getattr(file,'name',str(idx))}: {e}")

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
    else:
        combined = None

    if removed_all:
        removed_combined = pd.concat(removed_all, ignore_index=True)
    else:
        removed_combined = None

    return combined, removed_combined

# --- DIALOGS ---
@st.dialog("📝 Machine History", width="large")
def show_machine_details(machine_name, df):
    st.markdown(f"### History for: **{machine_name}**")
    sub_df = df[df['Equipment description'] == machine_name].sort_values('Created_Date', ascending=False)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Jobs", len(sub_df))
    c2.metric("Total Cost", f"LKR {sub_df['TotSum (actual)'].sum():,.2f}" if 'TotSum (actual)' in sub_df.columns else "LKR 0.00")
    if len(sub_df) > 1:
        dates = sub_df['Created_Date'].sort_values()
        mtbf = dates.diff().dt.days.mean()
        c3.metric("MTBF", f"{mtbf:.1f} Days")
    else:
        c3.metric("MTBF", "N/A")
    st.divider()
    cols_to_show = ['Created_Date', 'Order', 'Description', 'Order Type', 'TotSum (actual)', 'User Status Text', 'Main WorkCtr']
    cols_to_show = [c for c in cols_to_show if c in sub_df.columns]
    st.dataframe(sub_df[cols_to_show], use_container_width=True, hide_index=True)

@st.dialog("📍 Location Details", width="large")
def show_location_details(loc_name, df, include_negatives=False, removed_df=None):
    st.markdown(f"### Activity at: **{loc_name}**")
    display_df = df.copy()
    if include_negatives and (removed_df is not None) and (not removed_df.empty):
        display_df = pd.concat([display_df, removed_df], ignore_index=True)
    if 'Location description' in display_df.columns:
        loc_col = 'Location description'
    else:
        loc_col = 'Loc_Display' if 'Loc_Display' in display_df.columns else 'Functional Loc.' if 'Functional Loc.' in display_df.columns else None
    if loc_col is None:
        st.write("No location column available for grouping.")
        return
    sub_df = display_df[display_df[loc_col] == loc_name].sort_values('Created_Date', ascending=False)
    c1, c2 = st.columns(2)
    c1.metric("Total Orders", len(sub_df))
    c2.metric("Total Spend", f"LKR {sub_df['TotSum (actual)'].sum():,.2f}" if 'TotSum (actual)' in sub_df.columns else "LKR 0.00")
    st.markdown("#### 📅 Activity Timeline")
    if 'TotSum (actual)' in sub_df.columns and not sub_df.empty:
        sub_df = sub_df.copy()
        sub_df['neg_flag'] = sub_df['TotSum (actual)'].apply(lambda x: 'Negative' if x < 0 else 'Positive')
        sub_df['plot_size'] = sub_df['TotSum (actual)'].abs() + 1
        color_map = {'Negative': 'red', 'Positive': '#636efa'}
        fig = px.scatter(sub_df, x='Created_Date', y='TotSum (actual)', color='neg_flag', size='plot_size',
                         color_discrete_map=color_map, hover_data=['Description', 'Order', 'Order Type'])
    else:
        fig = px.scatter(sub_df, x='Created_Date', y='TotSum (actual)' if 'TotSum (actual)' in sub_df.columns else None, hover_data=['Description'])
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### 📋 Work Log")
    show_cols = [c for c in ['Created_Date', 'Order', 'Description', 'Equipment description', 'TotSum (actual)'] if c in sub_df.columns]
    st.dataframe(sub_df[show_cols], use_container_width=True)

@st.dialog("📅 Daily Report", width="large")
def show_day_details(day_val, df):
    st.markdown(f"### Report for: **{day_val}**")
    sub_df = df[df['Day_Num'] == day_val]
    c1, c2, c3 = st.columns(3)
    c1.metric("Jobs Created", len(sub_df))
    c2.metric("Total Cost", f"LKR {sub_df['TotSum (actual)'].sum():,.2f}" if 'TotSum (actual)' in sub_df.columns else "LKR 0.00")
    c3.metric("Depts Involved", len(sub_df['Main WorkCtr'].unique()) if 'Main WorkCtr' in sub_df.columns else 0)
    st.divider()
    cols = [c for c in ['Order', 'Description', 'Equipment description', 'Main WorkCtr', 'TotSum (actual)'] if c in sub_df.columns]
    st.dataframe(sub_df[cols], use_container_width=True)
    
# --- COST POPUP ---
@st.dialog("💰 Detailed Cost & Budget Analysis", width="large")
def show_cost_popup(df):
    st.markdown("### 📊 Financial Breakdown")
    c1, c2, c3 = st.columns([2, 2, 2])
    depts = ['All'] + sorted(df['Main WorkCtr'].astype(str).unique().tolist()) if 'Main WorkCtr' in df.columns else ['All']
    types = ['All'] + sorted([x for x in df['Order Type'].unique() if str(x) != 'nan']) if 'Order Type' in df.columns else ['All']
    sel_d = c1.selectbox("Department:", depts, key='pop_d')
    sel_t = c2.selectbox("Order Type:", types, key='pop_t')
    show_all = c3.checkbox("Show All Rows", value=True)
    dff = df.copy()
    if sel_d != 'All':
        dff = dff[dff['Main WorkCtr'] == sel_d]
    if sel_t != 'All':
        dff = dff[dff['Order Type'] == sel_t]
    st.markdown("#### 🚨 Top 10 Budget Overruns")
    over = dff[dff.get('Cost_Variance', 0) > 0].nlargest(10, 'Cost_Variance') if 'Cost_Variance' in dff.columns else pd.DataFrame()
    show_cols = [c for c in ['Order', 'Description', 'TotSum (plan)', 'TotSum (actual)', 'Cost_Variance'] if c in over.columns]
    if not over.empty:
        st.dataframe(over[show_cols].style.format({"TotSum (plan)": "LKR {:,.2f}", "TotSum (actual)": "LKR {:,.2f}", "Cost_Variance": "LKR {:,.2f}"}), use_container_width=True)
        if 'Order' in over.columns:
            sel_over_order = st.selectbox("Select an overrun Order to view details", over['Order'].astype(str).tolist())
            if sel_over_order:
                rows = dff[dff['Order'].astype(str) == sel_over_order]
                st.dataframe(rows.T, use_container_width=True)
    else:
        st.write("No overruns found.")
    st.markdown("#### 💸 Top 20 Costliest Machines")
    if 'Equipment description' in dff.columns:
        cost_df = dff[dff['Equipment description'] != "Unknown Asset"].groupby('Equipment description')['TotSum (actual)'].sum().reset_index().sort_values('TotSum (actual)', ascending=False)
        cost_df.columns = ['Machine Name', 'Total Cost (LKR)']
        st.dataframe((cost_df if show_all else cost_df.head(20)).style.format({"Total Cost (LKR)": "LKR {:,.2f}"}), use_container_width=True, height=400)
        sel_machine = st.selectbox("Select machine to view history", cost_df['Machine Name'].head(20).tolist())
        if sel_machine and st.button("View Machine History (from costs)"):
            show_machine_details(sel_machine, df)

# --- MAIN APP ---
def main():
    if 'removed_negatives' not in st.session_state:
        st.session_state['removed_negatives'] = None
    if 'negatives_restored' not in st.session_state:
        st.session_state['negatives_restored'] = False
    if 'show_filtered_negatives' not in st.session_state:
        st.session_state['show_filtered_negatives'] = False

    with st.sidebar:
        st.markdown('<div class="watermark">Created By Trainee at Engineering Dpt.</div>', unsafe_allow_html=True)
        st.markdown("""
            <div class="logo-container">
                <img src="https://cdn-icons-png.flaticon.com/512/2920/2920349.png" width="80">
                <h3>Maintenance Analyzer</h3>
            </div>
        """, unsafe_allow_html=True)

        mode = st.radio("Select Mode", ["Standard Dashboard", "AI Analyzer (Deep 6-Month Forecast)"])

        files_to_process = []
        if mode == "Standard Dashboard":
            uploaded = st.file_uploader("Upload Monthly File", type=['xlsx', 'csv'], accept_multiple_files=True)
            if uploaded:
                files_to_process = uploaded
        else:
            st.markdown("### Upload files covering the last 6 months (multiple files allowed)")
            uploaded_ai = st.file_uploader("Upload files (last 6 months)", type=['xlsx', 'csv'], accept_multiple_files=True, key="ai_uploads")
            if uploaded_ai:
                files_to_process = uploaded_ai

        st.markdown("---")

        if 'df_main' in st.session_state:
            st.markdown("### 🔎 Universal Search")
            search_q = st.text_input("Order # or Keyword", placeholder="Type here...")
            if search_q:
                mask = st.session_state['df_main'].astype(str).apply(lambda x: x.str.contains(search_q, case=False, na=False)).any(axis=1)
                search_results = st.session_state['df_main'][mask]
                st.write(f"Found {len(search_results)} matches:")
                show_cols = [c for c in ['Order', 'Description', 'TotSum (actual)'] if c in search_results.columns]
                st.dataframe(search_results[show_cols], hide_index=True)

        st.markdown("---")

        if 'df_main' in st.session_state and mode == "Standard Dashboard":
            if st.button("📥 Download Report (PDF)"):
                try:
                    pdf_path = generate_pdf(st.session_state['df_main'])
                    with open(pdf_path, "rb") as f:
                        st.download_button("Click to Save PDF", f, file_name="Maintenance_Report.pdf")
                except Exception as e:
                    st.error(f"PDF Error: {e}")

        st.markdown("---")
        if st.session_state.get('removed_negatives') is not None and not (st.session_state['removed_negatives'] is None) and not st.session_state['removed_negatives'].empty:
            if st.button("➕ Add Filtered Negative Rows to Dataset (Permanent)"):
                if 'df_main' in st.session_state and st.session_state['df_main'] is not None:
                    merged = pd.concat([st.session_state['df_main'], st.session_state['removed_negatives']], ignore_index=True)
                    st.session_state['df_main'] = merged
                    st.session_state['removed_negatives'] = pd.DataFrame()
                    st.session_state['negatives_restored'] = True
                    st.success("Filtered negative rows have been permanently added to the main dataset.")
                else:
                    st.error("No main dataset found to merge into. Please upload files first.")



   
    deep_six_month_analyzer(df)
    
            def main():
            # ... previous code ...
            st.write("End of previous section")
        
            # CORRECT: Aligned with the code above
            st.header("Deep AI Analysis") 
            if st.button("Run Deep 6-Month Analyzer"):
                deep_six_month_analyzer(df)
                
    # HEADER & CLOCK & LOGO
    c_left, c_right = st.columns([5, 2])
    with c_left:
        st.markdown("""
            <div style="border-left: 6px solid #e11d48; padding-left: 15px; margin-top: 10px;">
                <h1 style='font-family: "Segoe UI", sans-serif; font-weight: 800; font-size: 2.5rem; color: #0f172a; margin: 0; line-height: 1.1; letter-spacing: -1px;'>
                    MAINTENANCE <span style='color: #e11d48;'>ANALYZER</span>
                </h1>
                <h3 style='font-family: "Segoe UI", sans-serif; font-weight: 500; font-size: 1.1rem; color: #64748b; margin-top: 5px; text-transform: uppercase; letter-spacing: 2px;'>
                    Lankatiles PLC
                </h3>
            </div>
        """, unsafe_allow_html=True)
    with c_right:
        if os.path.exists("lankatiles.png"):
            st.image("lankatiles.png", use_container_width=True)
        else:
            st.image("https://upload.wikimedia.org/wikipedia/commons/2/25/Lanka_Tiles_Logo.jpg", use_container_width=True)
        clock_html = """
        <!DOCTYPE html>
        <html>
        <head>
        <style>
            body { margin: 0; background: transparent; display: flex; flex-direction: column; align-items: center; justify-content: flex-start; padding-top: 10px; height: 60px; font-family: Arial, Helvetica, sans-serif; }
            .clock-time { font-size: 1.3em; font-weight: bold; color: #333; line-height: 1.2; }
            .clock-date { font-size: 0.85em; color: #666; }
            @media (prefers-color-scheme: dark) { .clock-time { color: #eee; } .clock-date { color: #bbb; } }
        </style>
        </head>
        <body>
            <div id="live-time" class="clock-time">--:--:--</div>
            <div id="live-date" class="clock-date">...</div>
            <script>
                function updateClock() {
                    var now = new Date();
                    document.getElementById('live-time').innerHTML = now.toLocaleTimeString();
                    document.getElementById('live-date').innerHTML = now.toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
                }
                setInterval(updateClock, 1000);
                updateClock();
            </script>
        </body>
        </html>
        """
        components.html(clock_html, height=80)

    if files_to_process:
        df_loaded, removed_neg = load_data(files_to_process)
        if df_loaded is not None:
            st.session_state['df_main'] = df_loaded
            if (st.session_state.get('removed_negatives') is None) or (st.session_state.get('negatives_restored') is False):
                st.session_state['removed_negatives'] = removed_neg
            st.session_state['negatives_restored'] = False

            df = df_loaded.copy()

            with st.sidebar:
                st.markdown("### 🔍 Filters")
                all_depts = ['All'] + sorted(df['Main WorkCtr'].astype(str).unique().tolist()) if 'Main WorkCtr' in df.columns else ['All']
                sel_dept = st.selectbox("Select Department", all_depts)
                all_types = ['All'] + sorted([x for x in df['Order Type'].unique() if str(x) != 'nan']) if 'Order Type' in df.columns else ['All']
                sel_type = st.selectbox("Select Order Type", all_types)

            if sel_dept != 'All':
                df = df[df['Main WorkCtr'] == sel_dept]
            if sel_type != 'All':
                df = df[df['Order Type'] == sel_type]

            if mode == "AI Analyzer (Deep 6-Month Forecast)":
                deep_six_month_analyzer(df)
            else:
                tabs = st.tabs(["🤖 AI Chat", "💰 Cost", "⚡ Efficiency", "🔧 Reliability", "👷 Workforce", "🧹 Quality"])

                # 1. AI CHAT
                with tabs[0]:
                    st.subheader("💬 Maintenance AI Assistant")
                    with st.expander("ℹ️ About this Section"):
                        st.write("Ask natural language questions about your data.")
                    st.write("⚡ **Quick Actions:**")
                    col_q1, col_q2, col_q3 = st.columns(3)
                    prompt = None
                    if col_q1.button("📉 Analyze Top Spenders"):
                        prompt = "List the top 5 most expensive machines and their total cost."
                    if col_q2.button("🔍 Find Common Failures"):
                        prompt = "What are the most common descriptions for Breakdown maintenance?"
                    if col_q3.button("📊 Dept. Breakdown"):
                        prompt = "Give me a summary of total costs by Main Work Center."
                    if "messages" not in st.session_state:
                        st.session_state.messages = []
                    for msg in st.session_state.messages:
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])
                    user_input = st.chat_input("Ask about costs, breakdowns...")
                    if user_input:
                        prompt = user_input
                    if prompt:
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        with st.chat_message("user"):
                            st.markdown(prompt)
                        with st.chat_message("assistant"):
                            with st.spinner("AI is analyzing data..."):
                                try:
                                    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
                                    agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True, agent_type="openai-functions")
                                    response = agent.run(prompt)
                                    st.markdown(response)
                                    st.session_state.messages.append({"role": "assistant", "content": response})
                                except Exception as e:
                                    st.error(f"AI Error: {str(e)}")

                # 2. COST
                with tabs[1]:
                    st.subheader("Financial Overview")
                    m1, m2, m3 = st.columns(3)
                    act = df['TotSum (actual)'].sum() if 'TotSum (actual)' in df.columns else 0
                    plan = df['TotSum (plan)'].sum() if 'TotSum (plan)' in df.columns else 0
                    m1.metric("Total Actual", f"LKR {act:,.0f}")
                    m2.metric("Total Plan", f"LKR {plan:,.0f}")
                    m3.metric("Variance", f"LKR {act-plan:,.0f}", delta_color="inverse")
                    st.markdown("#### 🚨 Cost Anomalies (3-Sigma)")
                    if 'TotSum (actual)' in df.columns:
                        mean_cost = df['TotSum (actual)'].mean()
                        std_cost = df['TotSum (actual)'].std()
                        anomalies = df[df['TotSum (actual)'] > (mean_cost + 3 * std_cost)]
                        if not anomalies.empty:
                            st.error(f"Found {len(anomalies)} outliers (Cost > LKR {mean_cost + 3*std_cost:,.0f})")
                            cols_show = [c for c in ['Order', 'Description', 'Equipment description', 'TotSum (actual)'] if c in anomalies.columns]
                            st.dataframe(anomalies[cols_show], use_container_width=True)
                        else:
                            st.success("No significant cost anomalies detected.")
                    else:
                        st.write("No TotSum (actual) column for anomaly detection.")
                    st.divider()
                    c_btn, c_ch = st.columns([1, 2])
                    with c_btn:
                        st.info("View Detailed Tables")
                        if st.button("🔎 Open Detailed Costs", type="primary", use_container_width=True):
                            show_cost_popup(df)
                    with c_ch:
                        color_map = {'Mechanical Dpt.': 'red', 'Electrical Dpt.': 'blue'}
                        if 'Main WorkCtr' in df.columns and 'TotSum (actual)' in df.columns:
                            st.plotly_chart(px.pie(df, values='TotSum (actual)', names='Main WorkCtr', title="Cost by Department", color='Main WorkCtr', color_discrete_map=color_map), use_container_width=True)
                    if 'Order Type' in df.columns and 'TotSum (actual)' in df.columns:
                        st.plotly_chart(px.bar(df.groupby('Order Type')['TotSum (actual)'].sum().reset_index(), x='Order Type', y='TotSum (actual)', title="Cost per Order Type"), use_container_width=True)

                # 3. EFFICIENCY
                with tabs[2]:
                    st.subheader("Operational Efficiency")
                    k1, k2, k3 = st.columns(3)
                    wait = df['User Status Text'].str.contains('Awaiting', na=False).sum() if 'User Status Text' in df.columns else 0
                    k1.metric("Approvals Pending", int(wait), delta="Bottleneck", delta_color="inverse")
                    avg_resp = df['Response_Time'].mean() if 'Response_Time' in df.columns else 0
                    k2.metric("Avg Response Time", f"{avg_resp:.1f} Days")
                    k3.metric("Immediate Starts", int(df[df.get('Response_Time', 0) == 0].shape[0]) if 'Response_Time' in df.columns else 0)
                    if 'Response_Time' in df.columns:
                        st.plotly_chart(px.histogram(df, x='Response_Time', nbins=20, title="Response Time Distribution"), use_container_width=True)

                # 4. RELIABILITY
                with tabs[3]:
                    st.subheader("Asset Reliability & MTBF")
                    include_neg_toggle = st.checkbox("Temporarily include filtered negative rows in reliability timelines", value=False)
                    df_for_rel = df.copy()
                    if include_neg_toggle and (st.session_state.get('removed_negatives') is not None) and (not st.session_state['removed_negatives'].empty):
                        df_for_rel = pd.concat([df_for_rel, st.session_state['removed_negatives']], ignore_index=True)
                    if 'Equipment description' in df_for_rel.columns:
                        clean_rel_df = df_for_rel[df_for_rel['Equipment description'] != "Unknown Asset"]
                        st.markdown("#### ⏱️ Mean Time Between Failures (MTBF) Analysis")
                        mtbf_data = []
                        for mach, group in clean_rel_df.groupby('Equipment description'):
                            if len(group) > 1:
                                group = group.sort_values('Created_Date')
                                diffs = group['Created_Date'].diff().dt.days.mean()
                                mtbf_data.append({'Machine': mach, 'MTBF_Days': diffs, 'Total_Failures': len(group)})
                        if mtbf_data:
                            mtbf_df = pd.DataFrame(mtbf_data).sort_values('MTBF_Days')
                            col_m1, col_m2 = st.columns(2)
                            with col_m1:
                                st.write("**Lowest MTBF (Frequent Failures)**")
                                st.dataframe(mtbf_df.head(10).style.format({'MTBF_Days': "{:.1f} days"}), use_container_width=True)
                            with col_m2:
                                st.plotly_chart(px.bar(mtbf_df.head(10), x='MTBF_Days', y='Machine', orientation='h', title="Assets requiring frequent attention"), use_container_width=True)
                        else:
                            st.info("Not enough historical data to calculate MTBF.")
                        st.divider()
                        bad = clean_rel_df['Equipment description'].value_counts().reset_index()
                        bad.columns = ['Machine', 'Breakdowns']
                        b1, b2 = st.columns([1, 2])
                        with b1:
                            st.write("**Top 20 Bad Actors (Volume)**")
                            st.dataframe(bad.head(20), use_container_width=True)
                        with b2:
                            st.plotly_chart(px.bar(bad.head(20), x='Breakdowns', y='Machine', orientation='h'), use_container_width=True)
                        st.markdown("#### 📅 Timeline & Deep Dive")
                        col_tool1, col_tool2 = st.columns(2)
                        with col_tool1:
                            threshold = st.slider("Filter: Show machines with >= X failures", 1, 50, 5)
                        with col_tool2:
                            machines_list = sorted(clean_rel_df['Equipment description'].unique().tolist())
                            selected_machine = st.selectbox("🔍 Select Machine to View Details", machines_list)
                            if st.button("View Machine History"):
                                show_machine_details(selected_machine, clean_rel_df)
                        high = bad[bad['Breakdowns'] >= threshold]['Machine']
                        t = clean_rel_df[clean_rel_df['Equipment description'].isin(high)]
                        if not t.empty and 'Start_Date' in t.columns:
                            if 'TotSum (actual)' in t.columns:
                                t = t.copy()
                                t['neg_flag'] = t['TotSum (actual)'].apply(lambda x: 'Negative' if x < 0 else 'Positive')
                                t['plot_size'] = t['TotSum (actual)'].abs() + 1
                                color_map = {'Negative': 'red', 'Positive': '#636efa'}
                                fig_time = px.scatter(t, x='Start_Date', y='Equipment description', size='plot_size', color='neg_flag',
                                                     color_discrete_map=color_map, hover_data=['Description', 'Order', 'Main WorkCtr'], title="Breakdown Timeline")
                            else:
                                fig_time = px.scatter(t, x='Start_Date', y='Equipment description', size='TotSum (actual)' if 'TotSum (actual)' in t.columns else None, color='Main WorkCtr', hover_data=['Description', 'Order'], title="Breakdown Timeline")
                            st.plotly_chart(fig_time, use_container_width=True)
                        if 'Loc_Display' in df_for_rel.columns:
                            st.markdown("#### 🔥 Location Hotspots")
                            loc_source = 'Location description' if 'Location description' in df_for_rel.columns else 'Loc_Display'
                            loc_list = sorted(df_for_rel[loc_source].astype(str).unique().tolist())
                            c_loc1, c_loc2 = st.columns([1, 2])
                            with c_loc1:
                                selected_loc = st.selectbox("🔍 Select Location to View", loc_list)
                                if st.button("View Location Details"):
                                    show_location_details(selected_loc, df_for_rel, include_negatives=include_neg_toggle, removed_df=st.session_state.get('removed_negatives'))
                            locs = df_for_rel[loc_source].value_counts().head(15).reset_index()
                            locs.columns = ['Location', 'count']
                            st.plotly_chart(px.treemap(locs, path=['Location'], values='count'), use_container_width=True)

                # 5. WORKFORCE
                with tabs[4]:
                    st.subheader("Workforce Analysis")
                    w1, w2 = st.columns(2)
                    color_map = {'Mechanical Dpt.': 'red', 'Electrical Dpt.': 'blue'}
                    with w1:
                        if 'Main WorkCtr' in df.columns:
                            st.plotly_chart(px.pie(df, names='Main WorkCtr', title="Dept Share", color='Main WorkCtr', color_discrete_map=color_map), use_container_width=True)
                    with w2:
                        if 'Entered By' in df.columns:
                            st.bar_chart(df['Entered By'].value_counts().head(10))
                    st.markdown("##### 📅 Daily Workload")
                    if 'Day_Num' in df.columns:
                        day_list = sorted(df['Day_Num'].unique().tolist(), reverse=True)
                        c_day1, c_day2 = st.columns([1, 2])
                        with c_day1:
                            selected_day = st.selectbox("🔍 Select Date to Inspect", day_list)
                            if st.button("View Day Details"):
                                show_day_details(selected_day, df)
                        daily = df.groupby('Day_Num').agg(Orders=('Order','count') if 'Order' in df.columns else ('Day_Num','count'), Works_Done=('Description', lambda x: '<br>'.join(x.astype(str).head(5)) if len(x)>0 else '')).reset_index()
                        fig_line = px.line(daily, x='Day_Num', y='Orders', title="Daily Orders Trend", markers=True)
                        st.plotly_chart(fig_line, use_container_width=True)

                # 6. QUALITY
                with tabs[5]:
                    st.subheader("Data Quality")
                    q1, q2 = st.columns(2)
                    q1.metric("Missing Equipment IDs", int(df['Equipment'].isnull().sum()) if 'Equipment' in df.columns else 0, delta_color="inverse")
                    q2.metric("Missing Func. Locs", int(df['Functional Loc.'].isnull().sum()) if 'Functional Loc.' in df.columns else 0)
                    st.write("Top Generic Descriptions:")
                    if 'Description' in df.columns:
                        st.dataframe(df['Description'].value_counts().head(15))
                    else:
                        st.write("No Description column found.")
                    st.markdown("---")
                    st.markdown("#### ⚠️ Filtered Negative Rows (removed during upload)")
                    if st.session_state.get('removed_negatives') is not None and not (st.session_state['removed_negatives'] is None) and not st.session_state['removed_negatives'].empty:
                        if st.button("Show Filtered Negative Rows"):
                            st.session_state['show_filtered_negatives'] = True
                        if st.session_state.get('show_filtered_negatives'):
                            rn = st.session_state['removed_negatives']
                            show_cols = [c for c in ['Order', 'Description', 'Equipment description', 'TotSum (actual)', 'TotSum (plan)', 'Removed_Reason', 'File_Index'] if c in rn.columns]
                            st.dataframe(rn[show_cols], use_container_width=True)
                            if st.button("Undo Filter — Restore Negative Rows to Dataset"):
                                merged = pd.concat([st.session_state.get('df_main', pd.DataFrame()), st.session_state['removed_negatives']], ignore_index=True)
                                st.session_state['df_main'] = merged
                                st.session_state['removed_negatives'] = pd.DataFrame()
                                st.session_state['negatives_restored'] = True
                                st.success("Negative rows restored into main dataset. Reload filters if needed.")
                    else:
                        st.info("No filtered negative rows captured for this upload.")

if __name__ == "__main__":
    main()
