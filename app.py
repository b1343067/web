import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. 核心數據處理與 AI 引擎
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

def calculate_all_indicators(df):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['MA200'] = df['Close'].rolling(window=200).mean()
    return df

def get_ai_prediction_model(df, days=7):
    df_pred = df.tail(120).reset_index()
    df_pred['Date_num'] = pd.to_datetime(df_pred['Date']).apply(lambda x: x.toordinal())
    X = df_pred[['Date_num']].values
    y = df_pred['Close'].values
    model = LinearRegression().fit(X, y)
    last_date_num = df_pred['Date_num'].max()
    future_dates_num = np.array([last_date_num + i for i in range(1, days + 1)]).reshape(-1, 1)
    future_preds = model.predict(future_dates_num)
    std_dev = df['Close'].tail(30).std()
    last_date = df_pred['Date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    return future_dates, future_preds, std_dev

# ==========================================
# 2. UI 專業介面設計 (視覺強化版)
# ==========================================

st.set_page_config(page_title="AlphaCheck Elite Pro", layout="wide")

# 強制修正：加入文字顏色 white !important 確保字體清晰
st.markdown("""
    <style>
    .decision-card {
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 25px;
        border: 2px solid;
        color: white !important; /* 強制文字為白色 */
    }
    .stMetric { background-color: #161a24; border: 1px solid #2d3139; padding: 15px; border-radius: 10px; }
    h1, h2, h3, p { color: white !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ AlphaCheck Elite: 智慧金融決策終端")

# --- 側邊欄 ---
st.sidebar.title("📊 市場監控")
with st.sidebar:
    tnx_h, _, _ = fetch_financial_data("^TNX")
    if tnx_h is not None:
        cur_y = tnx_h['Close'].iloc[-1]
        st.metric("美債 10Y 殖利率", f"{cur_y:.2f}%", delta=f"{cur_y - tnx_h['Close'].iloc[-2]:.2f}%")
        st.line_chart(tnx_h['Close'].tail(30))

tab1, tab2, tab3 = st.tabs(["🔍 AI 深度診斷", "🛡️ 投資組合風險", "📖 模型說明"])

with tab1:
    raw_ticker = st.text_input("輸入美股代號 (如 BRK/B, VOO, NVDA)", "NVDA").upper().strip()
    
    if raw_ticker:
        target = raw_ticker.replace("/", "-")
        with st.spinner(f'AI 引擎正在分析...'):
            hist, info, err = fetch_financial_data(target)
            
            if err:
                st.error(f"數據分析失敗：{err}")
            else:
                hist = calculate_all_indicators(hist)
                f_dates, f_preds, std = get_ai_prediction_model(hist)
                
                # --- A. 決策指令盒配色優化 (提高飽和度與亮度) ---
                cur_p = hist['Close'].iloc[-1]
                target_p = f_preds[-1]
                expected_ret = ((target_p - cur_p) / cur_p) * 100
                
                if expected_ret > 2.5:
                    b_color, b_border, b_text, b_icon = "#064e3b", "#10b981", "Strong Buy / 強力買入", "🚀"
                elif expected_ret < -2.5:
                    b_color, b_border, b_text, b_icon = "#7f1d1d", "#f87171", "Strong Sell / 建議減持", "⚠️"
                else:
                    b_color, b_border, b_text, b_icon = "#1f2937", "#9ca3af", "Neutral / 持平觀望", "⚖️"

                st.markdown(f"""
                    <div class="decision-card" style="background-color: {b_color}; border-color: {b_border};">
                        <h2 style="color: white !important; margin:0;">{b_icon} AI 指令：{b_text}</h2>
                        <p style="color: white !important; margin-top:10px; font-size:20px; font-weight: bold;">
                            7天預期目標：${target_p:.2f} (<span style="color: {'#4ade80' if expected_ret > 0 else '#fb7185'}">{expected_ret:+.2f}%</span>)
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                # --- B. 繪圖 (維持專業黑金風) ---
                plot_data = hist.tail(120)
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=plot_data.index, open=plot_data['Open'], high=plot_data['High'], low=plot_data['Low'], close=plot_data['Close'], name='歷史K線'))
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA200'].tail(120), line=dict(color='#ff9800', width=2), name='200MA'))
                
                # AI 預測陰影與虛線
                fig.add_trace(go.Scatter(x=f_dates + f_dates[::-1], y=list(f_preds + std) + list(f_preds - std)[::-1], fill='toself', fillcolor='rgba(255, 255, 255, 0.1)', line_color='rgba(255,255,255,0)', name='AI 波動區間'))
                fig.add_trace(go.Scatter(x=f_dates, y=f_preds, line=dict(color='#ffffff', dash='dash', width=3), name='AI 預測路徑'))
                
                fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)

                # --- C. 指標面板 ---
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("即時股價", f"${cur_p:.2f}")
                c2.metric("RSI (14D)", f"{hist['RSI'].iloc[-1]:.1f}")
                c3.metric("預估 P/E", f"{info.get('forwardPE', 'N/A')}")
                c4.metric("Beta (風險)", f"{info.get('beta', 'N/A')}")

# ... (Tab 2 & 3 維持原本邏輯)
