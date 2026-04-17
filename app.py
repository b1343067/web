import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ==========================================
# 1. 核心邏輯與數據處理模組 (Functions)
# ==========================================

@st.cache_data(ttl=3600)
def fetch_financial_data(ticker_name):
    """
    從 Yahoo Finance 抓取數據並進行預處理
    """
    try:
        ticker = yf.Ticker(ticker_name)
        history = ticker.history(period="1y")
        info = ticker.info
        if history.empty:
            return None, None, "無此標的數據"
        return ticker, history, info, None
    except Exception as e:
        return None, None, None, str(e)

def calculate_technical_indicators(df):
    """
    計算進階技術指標：RSI, MACD, 波動率
    """
    # 計算 RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 計算 MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 計算 20日年化波動率
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['Log_Ret'].rolling(window=20).std() * np.sqrt(252)
    
    return df

# ==========================================
# 2. 介面與排版設計 (UI Layout)
# ==========================================

st.set_page_config(page_title="AlphaCheck Elite | 數位金融終端", layout="wide")

# 自定義 CSS 讓介面更帥
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ AlphaCheck Elite: 智慧型金融決策終端")
st.sidebar.title("📊 市場監控中心")

# --- 側邊欄：美債與市場背景 ---
with st.sidebar:
    st.subheader("美債 10Y 殖利率")
    _, tnx_h, _, _ = fetch_financial_data("^TNX")
    if tnx_h is not None:
        cur_y = tnx_h['Close'].iloc[-1]
        st.metric("目前水準", f"{cur_y:.2f}%", delta=f"{cur_y - tnx_h['Close'].iloc[-2]:.2f}%")
        st.line_chart(tnx_h['Close'].tail(30))
    st.divider()
    st.info("💡 提示：本系統採用快取技術，每小時自動更新數據。")

# --- 主功能區：分頁系統 ---
tab1, tab2, tab3 = st.tabs(["🔍 深度個股掃描", "🛡️ 投資組合風險", "📖 系統分析邏輯"])

with tab1:
    col_input, col_status = st.columns([2, 1])
    target = col_input.text_input("請輸入美股代號", "NVDA").upper()
    
    if target:
        with st.spinner('AI 引擎運算中...'):
            obj, hist, info, err = fetch_financial_data(target)
            
            if err:
                st.error(f"連線異常: {err}")
            else:
                hist = calculate_technical_indicators(hist)
                
                # A. 視覺化圖表
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'],
                                            low=hist['Low'], close=hist['Close'], name='K線圖'))
                fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(200).mean(), 
                                        line=dict(color='orange', width=2), name='200MA'))
                fig.update_layout(title=f"{target} 歷史走勢與長線支撐", template="plotly_dark", height=500)
                st.plotly_chart(fig, use_container_width=True)

                # B. 指標儀表板
                c1, c2, c3, c4 = st.columns(4)
                rsi_val = hist['RSI'].iloc[-1]
                vol_val = hist['Volatility'].iloc[-1]
                
                c1.metric("目前股價", f"${hist['Close'].iloc[-1]:.2f}")
                c2.metric("RSI (14D)", f"{rsi_val:.1f}", delta="-超買" if rsi_val > 70 else "+超跌" if rsi_val < 30 else "正常")
                c3.metric("年化波動率", f"{vol_val:.1%}")
                c4.metric("本益比 (PE)", f"{info.get('forwardPE', 'N/A')}")

                # C. 綜合評分邏輯 (更複雜的算法)
                st.subheader("🎯 智能投資點評")
                score = 0
                if hist['Close'].iloc[-1] > ma200 := hist['Close'].rolling(200).mean().iloc[-1]: score += 40
                if 30 <= rsi_val <= 60: score += 20
                if info.get('beta', 1) < 1.5: score += 20
                if info.get('recommendationKey') == 'buy': score += 20
                
                score_color = "green" if score >= 75 else "orange" if score >= 50 else "red"
                st.markdown(f"### 系統綜合評分：<span style='color:{score_color}'>{score} 分</span>", unsafe_allow_html=True)

with tab2:
    st.header("🛡️ 組合風險量化分析")
    portfolio_data = pd.DataFrame([{"代號": "NVDA", "金額": 5000}, {"代號": "VOO", "金額": 5000}])
    edited = st.data_editor(portfolio_data, num_rows="dynamic")
    
    if st.button("運行風險壓力測試"):
        total = edited["金額"].sum()
        p_beta = 0
        for _, row in edited.iterrows():
            _, _, i, _ = fetch_financial_data(row["代號"])
            if i: p_beta += (i.get('beta', 1.0) * (row["金額"] / total))
        
        st.metric("組合加權 Beta (相對於 S&P 500)", f"{p_beta:.2f}")
        st.progress(min(p_beta/2.0, 1.0), text=f"風險暴露程度：{p_beta:.2f}")

with tab3:
    st.header("📖 系統數學模型說明")
    st.markdown(f"""
    本系統基於以下金融數學模型進行開發：
    1. **相對強弱指標 (RSI)**：
       $RSI = 100 - \\frac{{100}}{{1 + RS}}$，其中 $RS = \\frac{{平均上漲幅度}}{{平均下跌幅度}}$。
    2. **組合風險 (Beta)**：
       $\\beta_p = \\sum w_i \\beta_i$，用於衡量組合對於大盤波動的敏感度。
    3. **技術面權重算法**：
       結合長線趨勢 (200MA)、情緒指標 (RSI) 與基本面 (P/E) 進行多因子評分。
    """)
