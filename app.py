import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. 核心數據處理模組
# ==========================================

@st.cache_data(ttl=3600)
def fetch_financial_data(ticker_name):
    try:
        clean_ticker = ticker_name.upper().replace("/", "-").strip()
        ticker_obj = yf.Ticker(clean_ticker)
        history = ticker_obj.history(period="2y")
        try:
            info = ticker_obj.info
        except:
            info = {}
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
    # 三重均線系統 (修正點)
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    return df

def get_ai_prediction_model(df, days=7):
    df_p = df.tail(120).reset_index()
    df_p['Date_n'] = pd.to_datetime(df_p['Date']).apply(lambda x: x.toordinal())
    X = df_p[['Date_n']].values
    y = df_p['Close'].values
    model = LinearRegression().fit(X, y)
    last_d_n = df_p['Date_n'].max()
    future_n = np.array([last_d_n + i for i in range(1, days + 1)]).reshape(-1, 1)
    preds = model.predict(future_n)
    std_dev = df['Close'].tail(30).std() * 0.7
    last_d = df_p['Date'].max()
    future_d = [last_d + timedelta(days=i) for i in range(1, days + 1)]
    return future_d, preds, std_dev

# ==========================================
# 2. 全局視覺美化 CSS
# ==========================================

st.set_page_config(page_title="AlphaCheck Elite 18.0", layout="wide")

st.markdown("""
    <style>
    /* 1. 背景與側邊欄 */
    .stApp, [data-testid="stSidebar"] { background-color: #0f172a !important; }
    
    /* 2. 文字與標籤清晰度 */
    h1, h2, h3, p, span, label, .stMarkdown { color: #f8fafc !important; }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 16px !important; }
    [data-testid="stMetricValue"] { color: #60a5fa !important; font-weight: bold !important; }

    /* 3. 解決表格(DataEditor)很醜的問題：強制背景與文字 */
    div[data-testid="stDataEditor"] {
        background-color: #1e293b !important;
        border-radius: 10px;
        padding: 10px;
    }

    /* 4. 決策卡片與 Info Box */
    .decision-card {
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid;
        backdrop-filter: blur(10px);
    }
    .stAlert { background-color: #1e293b !important; border: 1px solid #60a5fa !important; color: white !important; }

    /* 5. 修正按鈕樣式 */
    .stButton>button {
        width: 100%;
        background-color: #60a5fa !important;
        color: white !important;
        border-radius: 8px;
        border: none;
        padding: 10px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ AlphaCheck Elite: 專業投資決策終端")

# --- 側邊欄 ---
with st.sidebar:
    st.markdown("### 🌍 市場監控中心")
    tnx_h, _, _ = fetch_financial_data("^TNX")
    if tnx_h is not None:
        cur_y = tnx_h['Close'].iloc[-1]
        st.metric("美債 10Y 殖利率", f"{cur_y:.2f}%", delta=f"{cur_y - tnx_h['Close'].iloc[-2]:.2f}%")
        fig_side = px.line(tnx_h.tail(30), y='Close', template="plotly_dark")
        fig_side.update_traces(line_color='#60a5fa', line_width=2)
        fig_side.update_layout(height=120, margin=dict(l=0,r=0,t=0,b=0), xaxis_visible=False, yaxis_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_side, use_container_width=True, config={'displayModeBar': False})
    st.info("💡 已啟用 AI 趨勢分析與三重均線監控。")

tab1, tab2, tab3 = st.tabs(["🔍 AI 深度診斷", "🛡️ 投資組合風險", "📖 模型說明"])

# --- Tab 1: 診斷 ---
with tab1:
    col_in, _ = st.columns([2, 2])
    raw_ticker = col_in.text_input("輸入美股代號", "NVDA")
    if raw_ticker:
        target = raw_ticker.upper().replace("/", "-").strip()
        hist, info, err = fetch_financial_data(target)
        if not err:
            hist = calculate_indicators(hist)
            f_dates, f_preds, std = get_ai_prediction_model(hist)
            
            # A. 決策盒
            cur_p = hist['Close'].iloc[-1]
            target_p = f_preds[-1]
            expected_ret = ((target_p - cur_p) / cur_p) * 100
            
            if expected_ret > 2.0: b_bg, b_border, b_text = "rgba(20, 83, 45, 0.5)", "#4ade80", "多頭趨勢 / Buy"
            elif expected_ret < -2.0: b_bg, b_border, b_text = "rgba(127, 29, 29, 0.5)", "#f87171", "空頭預警 / Sell"
            else: b_bg, b_border, b_text = "rgba(30, 41, 59, 0.7)", "#94a3b8", "盤整觀望 / Neutral"

            st.markdown(f"""
                <div class="decision-card" style="background-color: {b_bg}; border-color: {b_border};">
                    <h3 style="margin:0; color: white !important;">🤖 AI 智能評級：{b_text}</h3>
                    <p style="margin-top:10px; font-size:18px; color: white !important;">
                        7日目標價：<b>${target_p:.2f}</b> | 期望收益：<b>{expected_ret:+.2f}%</b>
                    </p>
                </div>
            """, unsafe_allow_html=True)

            # B. 繪製圖表 (把 MA10, MA50, MA200 全部畫上去)
            plot_data = hist.tail(120)
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=plot_data.index, open=plot_data['Open'], high=plot_data['High'], low=plot_data['Low'], close=plot_data['Close'], name='歷史走勢', increasing_line_color='#4ade80', decreasing_line_color='#f87171'))
            
            # --- 修正後的均線呈現 ---
            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA10'], line=dict(color='#81d4fa', width=1), name='10MA (短線)'))
            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA50'], line=dict(color='#fbbf24', width=1), name='50MA (中線)'))
            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA200'], line=dict(color='#94a3b8', width=2), name='200MA (生命線)'))
            
            # AI 預測
            fig.add_trace(go.Scatter(x=f_dates + f_dates[::-1], y=list(f_preds + std) + list(f_preds - std)[::-1], fill='toself', fillcolor='rgba(96, 165, 250, 0.1)', line_color='rgba(0,0,0,0)', name='AI 波動預估'))
            fig.add_trace(go.Scatter(x=f_dates, y=f_preds, line=dict(color='#60a5fa', dash='dot', width=2), name='AI 預測路徑'))
            
            fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)

            # C. 指標
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("即時股價", f"${cur_p:.2f}")
            c2.metric("RSI 指標", f"{hist['RSI'].iloc[-1]:.1f}")
            c3.metric("預估 P/E", f"{info.get('forwardPE', 'N/A')}")
            c4.metric("市場 Beta", f"{info.get('beta', 'N/A')}")

# --- Tab 2: 組合風險 (視覺大改造) ---
with tab2:
    st.markdown("### 🛡️ 組合風險量化壓力測試")
    with st.container():
        # 設定預設組合
        p_df = pd.DataFrame([{"代號": "NVDA", "金額": 5000}, {"代號": "VOO", "金額": 5000}])
        # 使用 data_editor
        edited = st.data_editor(p_df, num_rows="dynamic", key="final_p_edit", use_container_width=True)
        
        st.write("") # 留白
        if st.button("🚀 運行分析"):
            total = edited["金額"].sum()
            w_beta = 0
            p_list = []
            for _, row in edited.iterrows():
                _, i, _ = fetch_financial_data(row["代號"])
                if i:
                    b = i.get('beta', 1.0)
                    w = row["金額"] / total
                    w_beta += b * w
                    p_list.append({"股票": row["代號"], "權重": w})
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.plotly_chart(px.pie(p_list, values='權重', names='股票', hole=0.4, title="資產配比", template="plotly_dark"))
            with res_col2:
                st.metric("組合加權 Beta 值", f"{w_beta:.2f}")
                risk_lv = "高 (積極型)" if w_beta > 1.3 else "中 (穩健型)" if w_beta >= 0.9 else "低 (防禦型)"
                st.markdown(f"<div style='background-color: #1e293b; padding: 20px; border-radius: 10px; border: 1px solid #334155;'>"
                            f"<h4>總體風險等級</h4><h2 style='color: #60a5fa;'>{risk_lv}</h2>"
                            f"<p>此組合相對於標普 500 指數的波動敏感度為 {w_beta:.2f}。</p></div>", unsafe_allow_html=True)

with tab3:
    st.header("📖 模型說明")
    st.markdown("本系統採用 **OLS 線性回歸** 預測，並實作 **三重均線系統 (Triple MA System)** 與 **加權 Beta 風險模型**。")
