import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="買賣決策系統", layout="centered")
st.title("🤖 AlphaCheck: 智慧買賣決策系統")

ticker = st.text_input("輸入美股代號 (如: PLTR, NVDA, TSLA)", "PLTR")

if ticker:
    stock = yf.Ticker(ticker)
    # 抓取即時數據
    info = stock.info
    hist = stock.history(period="6mo")
    
    # --- 計算指標 ---
    current_price = hist['Close'].iloc[-1]
    ma50 = hist['Close'].rolling(window=50).mean().iloc[-1]
    pe_ratio = info.get('forwardPE', 0)
    
    # --- 決策邏輯 ---
    buy_signals = 0
    if current_price > ma50: buy_signals += 1  # 趨勢向上
    if pe_ratio > 0 and pe_ratio < 50: buy_signals += 1 # 估值尚可
    if info.get('recommendationKey') == 'buy': buy_signals += 1 # 分析師看好

    # --- 顯示結果 ---
    st.divider()
    if buy_signals >= 2:
        st.success(f"✅ 決策建議：【值得關注 / 買入】")
        st.balloons() # 撒花特效，增加厲害感
    elif buy_signals == 1:
        st.warning(f"⚠️ 決策建議：【中立 / 觀望】")
    else:
        st.error(f"❌ 決策建議：【風險較高 / 避開】")

    # 顯示數據小卡
    col1, col2, col3 = st.columns(3)
    col1.metric("目前股價", f"${current_price:.2f}")
    col2.metric("50日均線", f"${ma50:.2f}")
    col3.metric("本益比", f"{pe_ratio:.1f}")
