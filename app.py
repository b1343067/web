import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ==========================================
# 1. 核心邏輯模組 (Functions)
# ==========================================

@st.cache_data(ttl=3600)
def fetch_financial_data(ticker_name):
    """抓取數據，只回傳可快取的內容"""
    try:
        ticker = yf.Ticker(ticker_name)
        history = ticker.history(period="1y")
        try:
            info = ticker.info
        except:
            info = {}
        if history.empty:
            return None, None, "無此標的數據"
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
# 2. 網頁介面佈局 (UI Layout)
# ==========================================

st.set_page_config(page_title="AlphaCheck Elite | 數位金融", layout="wide")
st.title("🏛️ AlphaCheck Elite: 智慧型金融決策終端")

# --- 側邊欄監控 ---
st.sidebar.title("📊 市場監控")
with st.sidebar:
    tnx_h, _, _ = fetch_financial_data("^TNX")
    if tnx_h is not None:
        cur_y = tnx_h['Close'].iloc[-1]
        st.metric("美債 10Y 殖利率", f"{cur_y:.2f}%")
        st.line_chart(tnx_h['Close'].tail(60))
    st.info("💡 資料已實作安全快取保護。")

# --- 重要：先定義分頁，才不會報 NameError ---
tab1, tab2, tab3 = st.tabs(["🔍 個股診斷", "🛡️ 投資組合風險", "📖 系統分析邏輯"])

# --- Tab 1: 個股診斷 (含代號自動修正) ---
with tab1:
    raw_input = st.text_input("輸入代號 (如: VOO, BRK/B, TSLA)", "NVDA")
    # 自動將 / 轉成 - 並清理格式
    target = raw_input.upper().replace("/", "-").strip()
    
    if target:
        with st.spinner(f'正在分析 {target}...'):
            hist, info, err = fetch_financial_data(target)
            
            if err:
                if "無此標的" in err:
                    st.error(f"❌ 找不到代號 '{target}'，請檢查格式。")
                else:
                    st.error(f"🚨 API 目前受限，請幾分鐘後再試。")
            else:
                hist = calculate_indicators(hist)
                
                # A. K線圖
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'],
                                            low=hist['Low'], close=hist['Close'], name='K線'))
                fig.add_trace(go.Scatter(x=hist.index, y=hist['MA200'], line=dict(color='orange'), name='200MA'))
                fig.update_layout(title=f"{target} 歷史走勢", template="plotly_dark", height=500)
                st.plotly_chart(fig, use_container_width=True)

                # B. 數據面板
                c1, c2, c3 = st.columns(3)
                rsi_val = hist['RSI'].iloc[-1]
                cur_p = hist['Close'].iloc[-1]
                ma200_v = hist['MA200'].iloc[-1]
                
                c1.metric("目前股價", f"${cur_p:.2f}")
                c2.metric("RSI (14D)", f"{rsi_val:.1f}")
                c3.metric("P/E (估值)", info.get('forwardPE', 'N/A'))

                # C. 評分邏輯
                score = 0
                if cur_p > ma200_v: score += 40
                if 30 <= rsi_val <= 65: score += 30
                if (info.get('forwardPE', 100) < 55) or (info.get('quoteType') == 'ETF'): score += 30
                
                s_color = "green" if score >= 70 else "orange" if score >= 40 else "red"
                st.markdown(f"### 系統綜合評分：<span style='color:{s_color}'>{score} 分</span>", unsafe_allow_html=True)

# --- Tab 2: 組合風險 ---
with tab2:
    st.header("🛡️ 組合風險量化分析")
    portfolio_df = pd.DataFrame([{"代號": "NVDA", "金額": 5000}, {"代號": "VOO", "金額": 5000}])
    edited = st.data_editor(portfolio_df, num_rows="dynamic", key="p_edit")
    
    if st.button("運行分析"):
        total_amt = edited["金額"].sum()
        w_beta = 0
        for _, row in edited.iterrows():
            _, i, _ = fetch_financial_data(row["代號"].upper().replace("/", "-"))
            if i:
                w_beta += (i.get('beta', 1.0) * (row["金額"] / total_amt))
        st.metric("組合加權 Beta 值", f"{w_beta:.2f}")

# --- Tab 3: 理論說明 ---
with tab3:
    st.header("📖 系統分析邏輯")
    st.markdown("""
    1. **趨勢判斷**: 股價必須高於 200 日移動平均線 (200MA)。
    2. **情緒指標**: RSI 處於 30-65 之間視為健康區間。
    3. **風險管理**: 使用 Beta 值衡量相對於標普 500 的波動性。
    """)
