# data_handler.py
import pandas as pd
import streamlit as st
import base64
import json
# IMPORT THE DATA YOU JUST CREATED
from config.item_data import ENCODED_ITEM_DB_DATA 

# ... Paste decoding logic and load_and_clean_data function ...
# Decode Item DB
try:
    encoded_data = ENCODED_ITEM_DB_DATA
    if encoded_data:
        ITEM_DB = json.loads(base64.b64decode(encoded_data).decode())
    else:
        ITEM_DB = {}
except Exception:
    ITEM_DB = {}
