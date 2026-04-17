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
    從 Yahoo Finance 抓取數據並進行預處理，加入異常處理機制
    """
    try:
        ticker = yf.Ticker(ticker_name)
        history = ticker.history(period="1y")
        # 抓取 info 時改為更保守的寫法，防止被限流時程式崩潰
        try:
            info = ticker.info
        except:
            info = {}
        
        if history.empty:
            return None, None, None, "無此標的數據"
        return ticker, history, info, None
    except Exception as e:
        return None, None, None, str(e)

def calculate_indicators(df):
    """
    手寫計算技術指標：RSI、移動平均線
    """
    # 計算 RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 計算均線
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    return df

# ==========================================
# 2. 網頁介面佈局 (UI Layout)
# ==========================================

st.set_page_config(page_title="AlphaCheck Elite | 數位金融終端", layout="wide")

# 增加專業感
st.title("🏛️ AlphaCheck Elite: 智慧型金融決策終端")
st.caption(f"數據最後更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- 側邊欄：全球市場監控 ---
st.sidebar.title("📊 市場監控中心")
with st.sidebar:
    st.subheader("美債 10Y 殖利率 (^TNX)")
    _, tnx_h, _, _ = fetch_financial_data("^TNX")
    if tnx_h is not None:
        cur_y = tnx_h['Close'].iloc[-1]
        st.metric("目前水準", f"{cur_y:.2f}%", delta=f"{cur_y - tnx_h['Close'].iloc[-2]:.2f}%")
        st.line_chart(tnx_h['Close'].tail(60))
    st.divider()
    st.info("💡 系統已實作資料快取與限流保護。")

# --- 主功能區：分頁系統 ---
tab1, tab2, tab3 = st.tabs(["🔍 深度個股掃描", "🛡️ 投資組合風險", "📖 系統分析邏輯"])

# --- Tab 1: 個股診斷 ---
with tab1:
    col_in, _ = st.columns([2, 2])
    target = col_in.text_input("請輸入美股代號 (如: VOO, NVDA, TSLA)", "NVDA").upper()
    
    if target:
        with st.spinner(f'正在分析 {target}...'):
            obj, hist, info, err = fetch_financial_data(target)
            
            if err:
                st.error(f"目前 API 連線受限 (Rate Limit)，請幾分鐘後再試。")
            else:
                hist = calculate_indicators(hist)
                
                # A. K線圖與均線
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'],
                                            low=hist['Low'], close=hist['Close'], name='K線'))
                fig.add_trace(go.Scatter(x=hist.index, y=hist['MA200'], line=dict(color='orange'), name='200MA'))
                fig.update_layout(title=f"{target} 歷史趨勢與支撐分析", template="plotly_dark", height=500)
                st.plotly_chart(fig, use_container_width=True)

                # B. 重要數據儀表板
                c1, c2, c3, c4 = st.columns(4)
                rsi_val = hist['RSI'].iloc[-1]
                cur_p = hist['Close'].iloc[-1]
                ma200_v = hist['MA200'].iloc[-1]
                
                c1.metric("目前股價", f"${cur_p:.2f}")
                c2.metric("RSI (14D)", f"{rsi_val:.1f}")
                c3.metric("本益比 (PE)", info.get('forwardPE', 'N/A'))
                c4.metric("Beta 值", info.get('beta', 'N/A'))

                # C. 修正後的打分邏輯 (解決之前的語法錯誤)
                st.subheader("🎯 智能投資點評")
                score = 0
                reasons = []

                if cur_p > ma200_v:
                    score += 40
                    reasons.append("✅ 股價站穩 200 日長線均線，趨勢偏多。")
                
                if 30 <= rsi_val <= 65:
                    score += 20
                    reasons.append("✅ RSI 處於健康區間，未過熱。")
                elif rsi_val < 30:
                    score += 25
                    reasons.append("💎 RSI 顯示超跌，可能具備反彈機會。")

                if (info.get('forwardPE', 100) < 50) or (info.get('quoteType') == 'ETF'):
                    score += 20
                    reasons.append("✅ 估值尚屬合理或為指數型工具。")

                if info.get('recommendationKey') == 'buy':
                    score += 20
                    reasons.append("✅ 華爾街分析師給予買入評級。")

                # 顯示分數
                s_color = "green" if score >= 75 else "orange" if score >= 50 else "red"
                st.markdown(f"### 系統綜合評分：<span style='color:{s_color}'>{score} 分</span>", unsafe_allow_html=True)
                for r in reasons: st.write(r)

# --- Tab 2: 投資組合 ---
with tab2:
    st.header("🛡️ 組合風險量化分析")
    portfolio_df = pd.DataFrame([{"代號": "NVDA", "金額": 5000}, {"代號": "VOO", "金額": 5000}])
    edited = st.data_editor(portfolio_df, num_rows="dynamic", key="portfolio_edit")
    
    if st.button("運行風險測試"):
        total_amt = edited["金額"].sum()
        weighted_beta = 0
        p_list = []
        
        for _, row in edited.iterrows():
            _, _, i, _ = fetch_financial_data(row["代號"])
            if i:
                b = i.get('beta', 1.0)
                w = row["金額"] / total_amt
                weighted_beta += b * w
                p_list.append({"股票": row["代號"], "權重": w})
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.plotly_chart(px.pie(p_list, values='權重', names='股票', hole=0.4, title="資產比例"))
        with col_p2:
            st.metric("組合加權 Beta 值", f"{weighted_beta:.2f}")
            risk_desc = "大 (積極型)" if weighted_beta > 1.3 else "中 (穩健型)" if weighted_beta >= 0.9 else "小 (防禦型)"
            st.write(f"### 總體風險等級：**{risk_desc}**")

# --- Tab 3: 理論說明 ---
with tab3:
    st.header("📖 系統數學模型與邏輯說明")
    st.markdown("""
    本系統結合技術指標分析與馬可維茲資產定價概念：
    
    1. **相對強弱指標 (RSI)**：
    $$RSI = 100 - \\frac{100}{1 + RS}$$
    用於衡量市場超買或超賣的情況。
    
    2. **投資組合風險 (Weighted Beta)**：
    $$\\beta_p = \\sum_{i=1}^{n} w_i \\beta_i$$
    代表您的組合相對於標普 500 指數的波動敏感度。
    
    3. **長線過濾**：
    採用 200 日移動平均線 (200MA) 作為牛熊分界線，確保投資者站在趨勢的一方。
    """)
