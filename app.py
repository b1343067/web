import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="FinTech 投資領航員", layout="wide")

# --- 側邊欄：投資組合管理 ---
st.sidebar.header("📂 我的投資組合")
portfolio = st.sidebar.multiselect(
    "選擇追蹤標的", 
    ["NVDA", "PLTR", "TSLA", "AAPL", "GOOGL", "QQQM", "MSFT"],
    default=["NVDA", "PLTR"]
)

st.title("🚀 AlphaCheck 2.0: 全方位決策系統")

# 選擇目前要詳細分析的股票
target = st.selectbox("查看詳細分析", portfolio)

if target:
    stock = yf.Ticker(target)
    df = stock.history(period="1y")
    info = stock.info
    
    # --- 計算指標 ---
    price = df['Close'].iloc[-1]
    ma20 = df['Close'].rolling(20).mean().iloc[-1]
    ma200 = df['Close'].rolling(200).mean().iloc[-1]
    beta = info.get('beta', 1.0)
    
    # --- 風險評級邏輯 ---
    if beta < 0.8: risk_level, color = "低風險 (保守型)", "blue"
    elif beta <= 1.3: risk_level, color = "中風險 (穩健型)", "orange"
    else: risk_level, color = "高風險 (積極型)", "red"

    # --- 顯示面板 ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔭 長線評估 (趨勢與價值)")
        if price > ma200 and info.get('forwardPE', 100) < 50:
            st.success("✅ 長線建議：適合佈局")
        else:
            st.warning("⚠️ 長線建議：保守觀望")
        st.write(f"200日均線: ${ma200:.2f} | P/E: {info.get('forwardPE', 'N/A')}")

    with col2:
        st.subheader("⚡ 短線評估 (動能與情緒)")
        if price > ma20:
            st.success("🔥 短線建議：動能強勁")
        else:
            st.error("❄️ 短線建議：走勢疲軟")
        st.write(f"20日均線: ${ma20:.2f}")

    st.divider()
    st.markdown(f"### 🛡️ 風險評級：:{color}[{risk_level}] (Beta: {beta})")
