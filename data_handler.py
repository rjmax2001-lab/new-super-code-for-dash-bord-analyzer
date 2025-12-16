import pandas as pd
import streamlit as st
import base64
import json
import numpy as np

# 1. Import the data string from your config file
from config.item_data import ENCODED_ITEM_DB_DATA 

# 2. Decode the ITEM_DB (This is the part you already have)
try:
    encoded_data = ENCODED_ITEM_DB_DATA
    # Fix padding if necessary
    padding = "=" * (-len(encoded_data) % 4)
    decoded_bytes = base64.b64decode(encoded_data + padding)
    decoded_str = decoded_bytes.decode('utf-8')
    ITEM_DB = json.loads(decoded_str)
except Exception as e:
    print(f"Error decoding ITEM_DB: {e}")
    ITEM_DB = {}

# 3. DEFINE THE MISSING FUNCTION (This is what you need to add!)
@st.cache_data
def load_and_clean_data(file):
    """
    Loads the Excel file, cleans column names, and maps equipment IDs.
    """
    # Load the file
    try:
        df = pd.read_excel(file, sheet_name='Sheet1')
    except:
        # Fallback if Sheet1 isn't found
        df = pd.read_excel(file)

    # Standardize Column Names (Adjust these to match your actual Excel file)
    # This renaming step ensures your code works even if Excel headers change slightly
    df.rename(columns={
        'Order': 'Order ID',
        'Description': 'Description',
        'Equipment': 'Equipment ID', 
        'Total act.cost': 'TotSum (actual)', 
        'Created on': 'Created_Date'
    }, inplace=True, errors='ignore')

    # Convert Dates
    if 'Created_Date' in df.columns:
        df['Created_Date'] = pd.to_datetime(df['Created_Date'], errors='coerce')
        # Add useful time columns
        df['Month'] = df['Created_Date'].dt.to_period('M')

    # Map Equipment Names using the ITEM_DB we decoded above
    if 'Equipment ID' in df.columns:
        # Convert ID to string and remove decimals (e.g. "20002.0" -> "20002")
        df['Equipment ID'] = df['Equipment ID'].astype(str).str.replace(r'\.0$', '', regex=True)
        # Map the ID to the Name
        df['Equipment description'] = df['Equipment ID'].map(ITEM_DB).fillna(df['Equipment ID'])

    return df
