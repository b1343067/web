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
# 1. 核心數據處理模組
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
    df['MA200'] = df['Close'].rolling(window=200).mean()
    return df

def get_scenario_predictions(df, days=7):
    df_p = df.tail(90).reset_index()
    df_p['Date_n'] = pd.to_datetime(df_p['Date']).apply(lambda x: x.toordinal())
    X = df_p[['Date_n']].values
    y = df_p['Close'].values
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    last_d_n = df_p['Date_n'].max()
    future_n = np.array([last_d_n + i for i in range(1, days + 1)]).reshape(-1, 1)
    base_preds = model.predict(poly.transform(future_n))
    volatility = df['Close'].tail(20).std()
    bull_preds = base_preds + (volatility * 1.5)
    bear_preds = base_preds - (volatility * 1.5)
    last_d = df_p['Date'].max()
    future_d = [last_d + timedelta(days=i) for i in range(1, days + 1)]
    return future_d, bull_preds, base_preds, bear_preds

# ==========================================
# 2. UI 旗艦視覺設計 (Midnight Navy)
# ==========================================

st.set_page_config(page_title="AlphaCheck Elite", layout="wide")

st.markdown("""
    <style>
    .stApp, [data-testid="stSidebar"] { background-color: #0f172a !important; }
    h1, h2, h3, p, span, label, .stMarkdown { color: #f1f5f9 !important; }
    
    /* 按鈕：極簡發光邊框 */
    .stButton>button {
        background-color: transparent !important;
        color: #60a5fa !important;
        border: 2px solid #60a5fa !important;
        border-radius: 25px !important;
        padding: 10px 40px !important;
        font-weight: 700 !important;
        width: 100% !important;
    }
    .stButton>button:hover {
        background-color: rgba(96, 165, 250, 0.1) !important;
        box-shadow: 0 0 20px rgba(96, 165, 250, 0.3) !important;
    }

    /* 報告卡片與 AI 評價盒 */
    .report-card {
        padding: 25px; border-radius: 15px; margin-bottom: 25px;
        border: 1px solid #334155; backdrop-filter: blur(10px);
    }
    div[data-testid="stDataEditor"] {
        border: 1px solid #334155 !important; border-radius: 10px;
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
    st.info("💡 系統已啟用 AI 場景分析模擬與組合評價系統。")

tab1, tab2, tab3 = st.tabs(["🔍 AI 市場診斷", "🛡️ 投資組合分析", "📖 模型說明"])

# --- Tab 1: AI 診斷 ---
with tab1:
    col_in, _ = st.columns([2, 2])
    raw_ticker = col_in.text_input("輸入美股代號 (如 BRK/B)", "NVDA")
    if raw_ticker:
        target = raw_ticker.upper().replace("/", "-").strip()
        hist, info, err = fetch_financial_data(target)
        if not err:
            hist = calculate_indicators(hist)
            f_dates, bull, base, bear = get_scenario_predictions(hist)
            
            cur_p = hist['Close'].iloc[-1]
            target_p = base[-1]
            expected_ret = ((target_p - cur_p) / cur_p) * 100
            
            if expected_ret > 2.0: b_bg, b_border, b_text = "rgba(20, 83, 45, 0.4)", "#4ade80", "多頭趨勢 / Buy"
            elif expected_ret < -2.0: b_bg, b_border, b_text = "rgba(127, 29, 29, 0.4)", "#f87171", "空頭預警 / Sell"
            else: b_bg, b_border, b_text = "rgba(30, 41, 59, 0.6)", "#94a3b8", "中立觀望 / Hold"

            st.markdown(f"""
                <div class="report-card" style="background-color: {b_bg}; border-color: {b_border};">
                    <h3 style="margin:0; color: white !important;">🤖 AI 投資評級：{b_text}</h3>
                    <p style="margin-top:8px; font-size:18px;">核心目標價：<b>${target_p:.2f}</b> | 期望變動：<b>{expected_ret:+.2f}%</b></p>
                </div>
            """, unsafe_allow_html=True)

            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=hist.tail(120).index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='走勢', increasing_line_color='#4ade80', decreasing_line_color='#f87171'))
            fig.add_trace(go.Scatter(x=f_dates, y=bull, line=dict(color='#4ade80', dash='dot', width=1), name='樂觀場景'))
            fig.add_trace(go.Scatter(x=f_dates, y=base, line=dict(color='white', dash='dash', width=2), name='核心預期'))
            fig.add_trace(go.Scatter(x=f_dates, y=bear, line=dict(color='#f87171', dash='dot', width=1), name='悲觀場景'))
            fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, legend=dict(font=dict(color="white")))
            st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: 組合分析 (新增 AI 評價) ---
with tab2:
    st.markdown("### 🛡️ 投資組合智能量化報告")
    
    # 預設數據，讓介面一進來就很充實
    default_p = pd.DataFrame([
        {"代號": "VOO", "金額": 5000}, {"代號": "BRK-B", "金額": 2850},
        {"代號": "NVDA", "金額": 915}, {"代號": "GOOGL", "金額": 987}
    ])
    edited = st.data_editor(default_p, num_rows="dynamic", use_container_width=True, key="p_table")
    
    if st.button("🚀 執行 AI 綜合評價分析"):
        with st.spinner('AI 正在分析組合健康度...'):
            total_val = edited["金額"].sum()
            results = []
            weighted_beta = 0
            weighted_rsi = 0
            stocks_above_ma = 0
            
            for _, row in edited.iterrows():
                h, i, _ = fetch_financial_data(row["代號"])
                if h is not None:
                    h = calculate_indicators(h)
                    beta = i.get('beta', 1.0) if i else 1.0
                    rsi = h['RSI'].iloc[-1]
                    weight = row["金額"] / total_val
                    
                    weighted_beta += beta * weight
                    weighted_rsi += rsi * weight
                    if h['Close'].iloc[-1] > h['MA200'].iloc[-1]:
                        stocks_above_ma += 1
                        
                    results.append({"股票": row["代號"].upper(), "權重": weight, "Beta": beta, "RSI": rsi, "貢獻": beta * weight})
            
            res_df = pd.DataFrame(results)
            
            # --- AI 智能評價邏輯 ---
            st.divider()
            st.markdown("### 🤖 AI 組合診斷與評價")
            
            # A. 評價指標計算
            div_score = min(len(results) * 20, 100) # 多樣性分數
            mkt_pos = (stocks_above_ma / len(results)) * 100 # 市場位置分數
            
            # B. 評價文字生成
            if weighted_beta > 1.3:
                eval_title = "進攻型組合 (Aggressive)"
                eval_color = "#f87171"
                eval_desc = "組合波動較大，主要受科技股或高成長股驅動。建議市場下行時增加現金比例。"
            elif weighted_beta < 0.9:
                eval_title = "防禦型組合 (Defensive)"
                eval_color = "#4ade80"
                eval_desc = "表現極其穩健，能有效抵抗大盤跌幅。適合追求長期穩定增長的投資者。"
            else:
                eval_title = "平衡型組合 (Balanced)"
                eval_color = "#fbbf24"
                eval_desc = "與市場同步率高，風險配置合理。VOO 等核心資產提供了良好的穩定性。"

            # C. 顯示 AI 評價盒
            st.markdown(f"""
                <div style="background-color: #1e293b; padding: 30px; border-radius: 15px; border-top: 5px solid {eval_color};">
                    <h2 style="color: {eval_color}; margin:0;">{eval_title}</h2>
                    <p style="margin-top:15px; font-size:18px; line-height:1.6;">
                        <b>AI 評價：</b>{eval_desc}<br>
                        <b>組合平均 RSI：</b>{weighted_rsi:.1f} ({'市場情緒過熱' if weighted_rsi > 70 else '情緒健康' if weighted_rsi > 30 else '處於超跌區'})<br>
                        <b>多樣性評分：</b>{div_score} / 100
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # D. 圖表呈現
            col_a, col_b = st.columns([2, 3])
            with col_a:
                st.plotly_chart(px.pie(res_df, values='權重', names='股票', hole=0.4, title="資產配比", template="plotly_dark"), use_container_width=True)
            with col_b:
                st.metric("組合加權 Beta", f"{weighted_beta:.2f}")
                st.plotly_chart(px.bar(res_df, x='股票', y='貢獻', title="個股對組合波動之驅動力 (Market Influence)", template="plotly_dark").update_traces(marker_color='#60a5fa'), use_container_width=True)

with tab3:
    st.header("📖 理論基礎")
    st.markdown("""
    本系統採用 **馬可維茲組合風險模型** 與 **AI 情緒權重分析**：
    
    1. **風險驅動 (Market Influence)**：計算公式為 $W_i \times \beta_i$。這代表該標的對你整體錢包漲跌的影響力，而非單純指該股票很危險。
    2. **AI 評價系統**：綜合 Beta、加權 RSI 以及多樣性指標（持股數量）產出的智能化診斷建議。
    """)
