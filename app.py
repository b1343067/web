import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ==========================================
# 1. 核心邏輯：修正快取錯誤
# ==========================================

@st.cache_data(ttl=3600)
def fetch_financial_data(ticker_name):
    """
    修正版：不回傳 yf.Ticker 物件，只回傳 DataFrame 和 Dict (可序列化)
    """
    try:
        ticker = yf.Ticker(ticker_name)
        history = ticker.history(period="1y")
        
        # 抓取 info 並處理限流
        try:
            info = ticker.info
        except:
            info = {}
        
        if history.empty:
            return None, None, "無此標的數據"
        
        # 注意：這裡只回傳資料，不回傳 ticker 物件
        return history, info, None
    except Exception as e:
        return None, None, str(e)

def calculate_indicators(df):
    """計算 RSI 與均線"""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['MA200'] = df['Close'].rolling(window=200).mean()
    return df

# ==========================================
# 2. 網頁介面佈局
# ==========================================

st.set_page_config(page_title="AlphaCheck Elite | 數位金融", layout="wide")
st.title("🏛️ AlphaCheck Elite: 智慧型金融決策終端")

# --- 側邊欄監控 ---
st.sidebar.title("📊 市場監控")
with st.sidebar:
    # 這裡解包改成 3 個參數
    tnx_h, _, _ = fetch_financial_data("^TNX")
    if tnx_h is not None:
        cur_y = tnx_h['Close'].iloc[-1]
        st.metric("美債 10Y 殖利率", f"{cur_y:.2f}%", delta=f"{cur_y - tnx_h['Close'].iloc[-2]:.2f}%")
        st.line_chart(tnx_h['Close'].tail(60))
    st.info("💡 資料已實作安全快取保護。")

# --- 功能分頁 ---
tab1, tab2, tab3 = st.tabs(["🔍 個股診斷", "🛡️ 投資組合風險", "📖 系統分析邏輯"])

with tab1:
    target = st.text_input("輸入代號 (如: VOO, NVDA)", "NVDA").upper()
    if target:
        # 解包改成 3 個參數
        hist, info, err = fetch_financial_data(target)
        if err:
            st.error(f"目前 API 暫時受限，請稍候再試。")
        else:
            hist = calculate_indicators(hist)
            
            # K線圖
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'],
                                        low=hist['Low'], close=hist['Close'], name='K線'))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['MA200'], line=dict(color='orange'), name='200MA'))
            fig.update_layout(title=f"{target} 歷史走勢", template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)

            # 數據面板與評分
            rsi_val = hist['RSI'].iloc[-1]
            cur_p = hist['Close'].iloc[-1]
            ma200_v = hist['MA200'].iloc[-1]
            
            c1, c2, c3 = st.columns(3)
            c1.metric("目前股價", f"${cur_p:.2f}")
            c2.metric("RSI (14D)", f"{rsi_val:.1f}")
            c3.metric("P/E (估值)", info.get('forwardPE', 'N/A'))

            # 評分邏輯
            score = 0
            if cur_p > ma200_v: score += 40
            if 30 <= rsi_val <= 65: score += 30
            if (info.get('forwardPE', 100) < 50) or (info.get('quoteType') == 'ETF'): score += 30

            st.markdown(f"### 系統綜合評分：{score} 分")

with tab2:
    st.header("🛡️ 組合風險量化分析")
    portfolio_df = pd.DataFrame([{"代號": "NVDA", "金額": 5000}, {"代號": "VOO", "金額": 5000}])
    edited = st.data_editor(portfolio_df, num_rows="dynamic", key="portfolio_edit")
    
    if st.button("開始評估"):
        total_amt = edited["金額"].sum()
        weighted_beta = 0
        for _, row in edited.iterrows():
            # 解包改成 3 個參數
            _, i, _ = fetch_financial_data(row["代號"])
            if i:
                weighted_beta += (i.get('beta', 1.0) * (row["金額"] / total_amt))
        st.metric("組合加權 Beta 值", f"{weighted_beta:.2f}")

with tab3:
    st.header("📖 邏輯說明")
    st.markdown("""
    1. **RSI 公式**: $RSI = 100 - \\frac{100}{1 + RS}$
    2. **加權風險**: $\\beta_p = \\sum w_i \\beta_i$
    """)
