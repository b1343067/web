import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ==========================================
# 1. 核心數據模組 (含自動修正與長回溯期)
# ==========================================

@st.cache_data(ttl=3600)
def fetch_financial_data(ticker_name):
    """
    抓取 2 年數據以確保 200MA 完整，並只回傳可快取的資料類型
    """
    try:
        # 自動修正代號格式 (如 BRK/B -> BRK-B)
        clean_ticker = ticker_name.upper().replace("/", "-").strip()
        ticker_obj = yf.Ticker(clean_ticker)
        
        # 抓取 2 年數據 (確保 200MA 不會斷掉)
        history = ticker_obj.history(period="2y")
        
        try:
            info = ticker_obj.info
        except:
            info = {}
            
        if history.empty:
            return None, None, f"找不到代號 {clean_ticker}"
            
        return history, info, None
    except Exception as e:
        return None, None, str(e)

def calculate_advanced_indicators(df):
    """計算 RSI 與三重均線 (10, 50, 200)"""
    # RSI 計算
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 三重均線計算
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # 只取最近一年的資料來畫圖，但前面的數據已用來算好 200MA 了
    return df.tail(252)

# ==========================================
# 2. 網頁 UI 佈局
# ==========================================

st.set_page_config(page_title="AlphaCheck Elite | 數位金融終端", layout="wide")

# 主標題
st.title("🏛️ AlphaCheck Elite: 智慧型金融決策終端")
st.caption(f"系統狀態：正常 | 數據更新時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- 側邊欄：全球市場監控 ---
st.sidebar.title("📊 市場監控中心")
with st.sidebar:
