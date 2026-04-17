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
# 1. 核心數據處理引擎
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

def get_ai_prediction_model(df, days=7):
    """多項式回歸趨勢預測"""
    df_p = df.tail(90).reset_index()
    df_p['Date_n'] = pd.to_datetime(df_p['Date']).apply(lambda x: x.toordinal())
    X = df_p[['Date_n']].values
    y = df_p['Close'].values
    
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    
    last_d_n = df_p['Date_n'].max()
    future_n = np.array([last_d_n + i for i in range(1, days + 1)]).reshape(-1, 1)
    future_preds = model.predict(poly.transform(future_n))
    
    base_std = df['Close'].tail(20).std()
    intervals = [base_std * (1 + (i * 0.2)) for i in range(len(future_preds))]
    
    last_d = df_p['Date'].max()
    future_d = [last_d + timedelta(days=i) for i in range(1, days + 1)]
    return future_d, future_preds, intervals

# ==========================================
# 2. UI 視覺設計與樣式優化
# ==========================================

st.set_page_config(page_title="AlphaCheck Elite", layout="wide")

st.markdown("""
    <style>
    /* 全局背景色 */
    .stApp, [data-testid="stSidebar"] { background-color: #0f172a !important; }
    h1, h2, h3, p, span, label, .stMarkdown { color: #f1f5f9 !important; }
    
    /* 側邊欄資訊框修復 */
    .stAlert { background-color: #1e293b !important; border: 1px solid #334155 !important; color: #f1f5f9 !important; }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    [data-testid="stMetricValue"] { color: #60a5fa !important; }

    /* 按鍵美化：發光邊框質感 */
    .stButton>button {
        background-color: transparent !important;
        color: #60a5fa !important;
        border: 2px solid #60a5fa !important;
        border-radius: 20px !important;
        padding: 8px 30px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease-in-out !important;
        width: 100% !important;
    }
    .stButton>button:hover {
        background-color: #60a5fa !important;
        color: #ffffff !important;
        box-shadow: 0 0 15px rgba(96, 165, 250, 0.4) !important;
        border-color: #60a5fa !important;
    }

    /* 決策盒：毛玻璃效果 */
    .decision-card {
        padding: 25px; border-radius: 12px; margin-bottom: 25px;
        border: 1px solid; backdrop-filter: blur(10px); color: white !important;
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
        st.plotly_chart(fig_side, use_container_width=True)
    st.divider()
    st.info("💡 數據已實作同步快取與 AI 曲線擬合。")

tab1, tab2, tab3 = st.tabs(["🔍 AI 深度分析", "🛡️ 投資組合風險", "📖 模型說明"])

# --- Tab 1: AI 診斷 ---
with tab1:
    col_in, _ = st.columns([2, 2])
    raw_ticker = col_in.text_input("輸入美股代號", "NVDA")
    
    if raw_ticker:
        target = raw_ticker.upper().replace("/", "-").strip()
        with st.spinner('正在計算市場趨勢...'):
            hist, info, err = fetch_financial_data(target)
            if not err:
                hist = calculate_indicators(hist)
                f_dates, f_preds, f_intervals = get_ai_prediction_model(hist)
                
                # 決策盒邏輯
                cur_p = hist['Close'].iloc[-1]
                target_p = f_preds[-1]
                expected_ret = ((target_p - cur_p) / cur_p) * 100
                
                if expected_ret > 2.0: b_bg, b_border, b_text = "rgba(20, 83, 45, 0.5)", "#4ade80", "多頭趨勢 / Buy"
                elif expected_ret < -2.0: b_bg, b_border, b_text = "rgba(127, 29, 29, 0.5)", "#f87171", "空頭預警 / Sell"
                else: b_bg, b_border, b_text = "rgba(30, 41, 59, 0.7)", "#94a3b8", "盤整觀望 / Neutral"

                st.markdown(f"""
                    <div class="decision-card" style="background-color: {b_bg}; border-color: {b_border};">
                        <h3 style="margin:0; color: white !important;">🤖 AI 智能評級：{b_text}</h3>
                        <p style="margin-top:8px; font-size:18px; color: white !important;">預估 7 日目標：<b>${target_p:.2f}</b> | 預期收益：<b>{expected_ret:+.2f}%</b></p>
                    </div>
                """, unsafe_allow_html=True)

                # 圖表渲染
                plot_data = hist.tail(120)
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=plot_data.index, open=plot_data['Open'], high=plot_data['High'], low=plot_data['Low'], close=plot_data['Close'], name='歷史價格', increasing_line_color='#4ade80', decreasing_line_color='#f87171'))
                
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA10'], line=dict(color='#81d4fa', width=1), name='10MA'))
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA200'], line=dict(color='#64748b', width=2), name='200MA'))
                
                # AI 曲線預測
                fig.add_trace(go.Scatter(
                    x=f_dates + f_dates[::-1],
                    y=[f_preds[i] + f_intervals[i] for i in range(len(f_preds))] + [f_preds[i] - f_intervals[i] for i in range(len(f_preds))][::-1],
                    fill='toself', fillcolor='rgba(96, 165, 250, 0.1)', line_color='rgba(0,0,0,0)', name='AI 波動預估'
                ))
                fig.add_trace(go.Scatter(x=f_dates, y=f_preds, line=dict(color='#60a5fa', dash='dot', width=3), name='AI 預測路徑'))
                
                fig.update_layout(
                    template="plotly_dark", height=550, xaxis_rangeslider_visible=False,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#f1f5f9"))
                )
                st.plotly_chart(fig, use_container_width=True)

                # 指標卡片
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("即時股價", f"${cur_p:.2f}")
                c2.metric("RSI 指標", f"{hist['RSI'].iloc[-1]:.1f}")
                c3.metric("預估 P/E", f"{info.get('forwardPE', 'N/A')}")
                c4.metric("市場 Beta", f"{info.get('beta', 'N/A')}")

# --- Tab 2: 投資組合 ---
with tab2:
    st.markdown("### 🛡️ 投資組合風險量化測試")
    p_df = pd.DataFrame([{"代號": "NVDA", "金額": 5000}, {"代號": "VOO", "金額": 5000}])
    edited = st.data_editor(p_df, num_rows="dynamic", key="final_p_edit", use_container_width=True)
    if st.button("🚀 開始量化分析"):
        total = edited["金額"].sum()
        w_beta = 0
        p_list = []
        for _, row in edited.iterrows():
            _, i, _ = fetch_financial_data(row["代號"])
            if i:
                b = i.get('beta', 1.0)
                w_beta += b * (row["金額"] / total)
                p_list.append({"股票": row["代號"].upper(), "權重": row["金額"] / total})
        
        ca, cb = st.columns(2)
        with ca:
            st.plotly_chart(px.pie(p_list, values='權重', names='股票', hole=0.4, title="資產配比", template="plotly_dark"))
        with cb:
            st.metric("組合加權 Beta 值", f"{w_beta:.2f}")
            risk_lv = "高 (積極型)" if w_beta > 1.3 else "中 (穩健型)" if w_beta >= 0.9 else "低 (防禦型)"
            st.markdown(f"<div style='background-color: #1e293b; padding: 25px; border-radius: 12px; border: 1px solid #334155;'>"
                        f"<h4 style='margin:0; opacity: 0.8;'>總體風險等級</h4><h1 style='color: #60a5fa;'>{risk_lv}</h1></div>", unsafe_allow_html=True)

# --- Tab 3: 理論說明 ---
with tab3:
    st.header("📖 模型理論基礎")
    st.markdown("""
    1. **多項式回歸 (Polynomial Regression)**：採用非線性擬合捕捉股價轉折與斜率。
    2. **動態信心區間**：
    $$Prediction \\pm \\sigma_{volatility} \\times (1 + \\Delta t)$$
    預測陰影隨時間擴張，符合金融市場不確定性原理。
    """)
