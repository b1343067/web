import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ==========================================
# 1. 核心 AI 引擎：三路徑場景模擬
# ==========================================

@st.cache_data(ttl=3600)
def fetch_financial_data(ticker_name):
    try:
        clean_ticker = ticker_name.upper().replace("/", "-").strip()
        ticker_obj = yf.Ticker(clean_ticker)
        history = ticker_obj.history(period="2y")
        info = ticker_obj.info if hasattr(ticker_obj, 'info') else {}
        if history.empty: return None, None, f"找不到代號 {clean_ticker}"
        return history, info, None
    except Exception as e:
        return None, None, str(e)

def calculate_indicators(df):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    return df

def get_scenario_predictions(df, days=7):
    """
    將抽象預測轉化為具體的 樂觀/中立/悲觀 三路徑
    """
    df_p = df.tail(60).reset_index() # 取最近兩個月數據
    df_p['Date_n'] = pd.to_datetime(df_p['Date']).apply(lambda x: x.toordinal())
    X = df_p[['Date_n']].values
    y = df_p['Close'].values
    
    # 核心模型 (中立)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    
    last_d_n = df_p['Date_n'].max()
    future_n = np.array([last_d_n + i for i in range(1, days + 1)]).reshape(-1, 1)
    base_preds = model.predict(poly.transform(future_n))
    
    # 計算波動度 (Volatility) 來建立上下路徑
    volatility = df['Close'].tail(20).std()
    bull_preds = base_preds + (volatility * 1.5) # 樂觀路徑
    bear_preds = base_preds - (volatility * 1.5) # 悲觀路徑
    
    last_d = df_p['Date'].max()
    future_d = [last_d + timedelta(days=i) for i in range(1, days + 1)]
    return future_d, bull_preds, base_preds, bear_preds

# ==========================================
# 2. UI 旗艦視覺設計
# ==========================================

st.set_page_config(page_title="AlphaCheck Elite", layout="wide")

st.markdown("""
    <style>
    .stApp, [data-testid="stSidebar"] { background-color: #0f172a !important; }
    h1, h2, h3, p, span, label, .stMarkdown { color: #f1f5f9 !important; }
    
    /* 側邊欄 Info Box */
    .stAlert { background-color: #1e293b !important; border: 1px solid #334155 !important; color: #f1f5f9 !important; }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    [data-testid="stMetricValue"] { color: #60a5fa !important; }

    /* 按鈕：極簡發光邊框 */
    .stButton>button {
        background-color: transparent !important;
        color: #60a5fa !important;
        border: 2px solid #60a5fa !important;
        border-radius: 25px !important;
        padding: 10px 40px !important;
        font-weight: 700 !important;
        transition: 0.3s all;
        width: 100% !important;
    }
    .stButton>button:hover {
        background-color: rgba(96, 165, 250, 0.15) !important;
        box-shadow: 0 0 20px rgba(96, 165, 250, 0.3) !important;
    }

    /* 決策盒 */
    .decision-card {
        padding: 25px; border-radius: 12px; margin-bottom: 25px;
        border: 1px solid; backdrop-filter: blur(10px);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ AlphaCheck Elite: 專業投資決策終端")

# --- 側邊欄 ---
with st.sidebar:
    st.markdown("### 🌍 市場監控中心")
    tnx_h, _, _ = fetch_financial_data("^TNX")
    if tnx_h is not None:
        st.metric("美債 10Y 殖利率", f"{tnx_h['Close'].iloc[-1]:.2f}%")
        fig_side = px.line(tnx_h.tail(30), y='Close', template="plotly_dark")
        fig_side.update_traces(line_color='#60a5fa', line_width=2)
        fig_side.update_layout(height=120, margin=dict(l=0,r=0,t=0,b=0), xaxis_visible=False, yaxis_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_side, use_container_width=True)
    st.divider()
    st.info("💡 系統已啟用 AI 場景模擬 (Scenario Analysis)。")

tab1, tab2, tab3 = st.tabs(["🔍 AI 市場分析", "🛡️ 投資組合風險", "📖 理論說明"])

# --- Tab 1: AI 診斷 ---
with tab1:
    col_in, _ = st.columns([2, 2])
    raw_ticker = col_in.text_input("輸入美股代號 (支援 BRK/B)", "NVDA")
    
    if raw_ticker:
        target = raw_ticker.upper().replace("/", "-").strip()
        with st.spinner('AI 正在運算多重路徑場景...'):
            hist, info, err = fetch_financial_data(target)
            if not err:
                hist = calculate_indicators(hist)
                f_dates, bull, base, bear = get_scenario_predictions(hist)
                
                # 決策盒
                cur_p = hist['Close'].iloc[-1]
                target_p = base[-1]
                expected_ret = ((target_p - cur_p) / cur_p) * 100
                
                if expected_ret > 2.0: b_bg, b_border, b_text = "rgba(20, 83, 45, 0.4)", "#4ade80", "多頭趨勢 / Buy"
                elif expected_ret < -2.0: b_bg, b_border, b_text = "rgba(127, 29, 29, 0.4)", "#f87171", "空頭預警 / Sell"
                else: b_bg, b_border, b_text = "rgba(30, 41, 59, 0.6)", "#94a3b8", "盤整觀望 / Neutral"

                st.markdown(f"""
                    <div class="decision-card" style="background-color: {b_bg}; border-color: {b_border};">
                        <h3 style="margin:0; color: white !important;">🤖 AI 投資指令：{b_text}</h3>
                        <p style="margin-top:8px; font-size:18px; color: white !important;">核心目標價：<b>${target_p:.2f}</b> | 期望收益：<b>{expected_ret:+.2f}%</b></p>
                    </div>
                """, unsafe_allow_html=True)

                # 圖表：三路徑展示
                plot_data = hist.tail(120)
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=plot_data.index, open=plot_data['Open'], high=plot_data['High'], low=plot_data['Low'], close=plot_data['Close'], name='歷史價格', increasing_line_color='#4ade80', decreasing_line_color='#f87171'))
                
                # 三重均線
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA10'], line=dict(color='#81d4fa', width=1), name='10MA'))
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA200'], line=dict(color='#64748b', width=2), name='200MA'))
                
                # AI 三路徑預測 (樂觀/中立/悲觀)
                fig.add_trace(go.Scatter(x=f_dates, y=bull, line=dict(color='#4ade80', dash='dot', width=1), name='AI 樂觀場景'))
                fig.add_trace(go.Scatter(x=f_dates, y=base, line=dict(color='#ffffff', dash='dash', width=3), name='AI 核心預期'))
                fig.add_trace(go.Scatter(x=f_dates, y=bear, line=dict(color='#f87171', dash='dot', width=1), name='AI 悲觀場景'))
                
                fig.update_layout(
                    template="plotly_dark", height=550, xaxis_rangeslider_visible=False,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#f1f5f9"))
                )
                st.plotly_chart(fig, use_container_width=True)

                # 指標卡片
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("即時股價", f"${cur_p:.2f}")
                c2.metric("RSI (14D)", f"{hist['RSI'].iloc[-1]:.1f}")
                c3.metric("本益比", info.get('forwardPE', 'N/A'))
                c4.metric("Beta 風險", info.get('beta', 'N/A'))

# --- Tab 2: 投資組合 ---
with tab2:
    st.markdown("### 🛡️ 投資組合風險量化測試")
    p_df = pd.DataFrame([{"代號": "NVDA", "金額": 5000}, {"代號": "VOO", "金額": 5000}])
    edited = st.data_editor(p_df, num_rows="dynamic", use_container_width=True)
    if st.button("🚀 開始分析"):
        total = edited["金額"].sum()
        w_beta = 0
        for _, row in edited.iterrows():
            _, i, _ = fetch_financial_data(row["代號"])
            if i: w_beta += i.get('beta', 1.0) * (row["金額"] / total)
        st.metric("組合加權 Beta 值", f"{w_beta:.2f}")
        st.info(f"此組合風險等級：{'高' if w_beta > 1.3 else '中' if w_beta >= 0.9 else '低'}")

with tab3:
    st.header("📖 系統邏輯說明")
    st.markdown("""
    1. **場景分析 (Scenario Analysis)**：AI 模型不僅提供核心預測 (Base)，更結合歷史波動率計算出樂觀 (Bull) 與悲觀 (Bear) 兩條邊界路徑。
    2. **多項式擬合**：利用 $y = ax^2 + bx + c$ 捕捉股價短期內部的加速或減速趨勢。
    """)
