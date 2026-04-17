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
    future_d = [last_date + timedelta(days=i) for i in range(1, days + 1)] if 'last_date' in locals() else [df_p['Date'].max() + timedelta(days=i) for i in range(1, days + 1)]
    return future_d, preds, std_dev

# ==========================================
# 2. UI 旗艦級 CSS 視覺優化
# ==========================================

st.set_page_config(page_title="AlphaCheck Elite 16.0", layout="wide")

# 強制全局深藍色調，修復側邊欄色差與提示框
st.markdown("""
    <style>
    /* 全局背景與側邊欄背景同步 */
    .stApp, [data-testid="stSidebar"] {
        background-color: #0f172a !important;
    }
    /* 側邊欄文字顏色修正 */
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #f1f5f9 !important;
    }
    /* 指令盒：專業毛玻璃 */
    .decision-card {
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid;
        backdrop-filter: blur(10px);
        color: white !important;
    }
    /* 提示框 (Info Box) 顏色修正：深色底白字 */
    .stAlert {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
        border: 1px solid #334155 !important;
    }
    /* 指標卡片 (Metrics) */
    [data-testid="stMetricValue"] { font-size: 28px !important; color: #60a5fa !important; }
    div[data-testid="metric-container"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ AlphaCheck Elite: 專業投資決策終端")
st.caption(f"數位金融科技系專案 | 版本 16.0 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# --- 側邊欄：質感修復 ---
st.sidebar.title("📊 市場監控中心")
with st.sidebar:
    st.markdown("### 🌍 全球宏觀指標")
    tnx_h, _, _ = fetch_financial_data("^TNX")
    if tnx_h is not None:
        cur_y = tnx_h['Close'].iloc[-1]
        st.metric("美債 10Y 殖利率", f"{cur_y:.2f}%", delta=f"{cur_y - tnx_h['Close'].iloc[-2]:.2f}%")
        
        # 側邊欄小圖美化
        fig_side = px.line(tnx_h.tail(30), y='Close', template="plotly_dark")
        fig_side.update_traces(line_color='#60a5fa', line_width=2)
        fig_side.update_layout(height=120, margin=dict(l=0,r=0,t=0,b=0), xaxis_visible=False, yaxis_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_side, use_container_width=True, config={'displayModeBar': False})
    
    st.divider()
    st.info("💡 系統已同步 UI 色調，啟用 AI 信心區間預測。")

# --- 功能分頁 ---
tab1, tab2, tab3 = st.tabs(["🎯 AI 深度診斷", "🛡️ 投資組合風險", "📖 理論與模型說明"])

# --- Tab 1: AI 診斷 ---
with tab1:
    col_in, _ = st.columns([2, 2])
    raw_ticker = col_in.text_input("輸入美股代號 (支援 BRK/B)", "NVDA")
    
    if raw_ticker:
        target = raw_ticker.upper().replace("/", "-").strip()
        with st.spinner('AI 引擎正在掃描市場數據...'):
            hist, info, err = fetch_financial_data(target)
            
            if err:
                st.error(f"數據分析失敗：{err}")
            else:
                hist = calculate_indicators(hist)
                f_dates, f_preds, std = get_ai_prediction_model(hist)
                
                # --- A. 決策盒：質感配色 ---
                cur_p = hist['Close'].iloc[-1]
                target_p = f_preds[-1]
                expected_ret = ((target_p - cur_p) / cur_p) * 100
                
                if expected_ret > 2.0:
                    b_bg, b_border, b_text = "rgba(20, 83, 45, 0.5)", "#4ade80", "多頭趨勢 / Buy"
                elif expected_ret < -2.0:
                    b_bg, b_border, b_text = "rgba(127, 29, 29, 0.5)", "#f87171", "空頭預警 / Sell"
                else:
                    b_bg, b_border, b_text = "rgba(30, 41, 59, 0.7)", "#94a3b8", "盤整觀望 / Neutral"

                st.markdown(f"""
                    <div class="decision-card" style="background-color: {b_bg}; border-color: {b_border};">
                        <h3 style="margin:0; color: white !important;">🤖 AI 智能評級：{b_text}</h3>
                        <p style="margin-top:10px; font-size:18px; color: white !important;">
                            預估 7 日目標：<b>${target_p:.2f}</b> | 期望收益：<b>{expected_ret:+.2f}%</b>
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                # --- B. 專業 Plotly 圖表 ---
                plot_data = hist.tail(120)
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=plot_data.index, open=plot_data['Open'], high=plot_data['High'], low=plot_data['Low'], close=plot_data['Close'], name='歷史走勢', increasing_line_color='#4ade80', decreasing_line_color='#f87171'))
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA200'].tail(120), line=dict(color='#64748b', width=2), name='200MA'))
                # AI 預測
                fig.add_trace(go.Scatter(x=f_dates + f_dates[::-1], y=list(f_preds + std) + list(f_preds - std)[::-1], fill='toself', fillcolor='rgba(96, 165, 250, 0.1)', line_color='rgba(0,0,0,0)', name='AI 波動預估'))
                fig.add_trace(go.Scatter(x=f_dates, y=f_preds, line=dict(color='#60a5fa', dash='dot', width=2), name='AI 預測路徑'))
                
                fig.update_layout(template="plotly_dark", height=550, xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)

                # --- C. 指標卡片 ---
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("即時股價", f"${cur_p:.2f}")
                c2.metric("RSI 指標", f"{hist['RSI'].iloc[-1]:.1f}")
                c3.metric("預估 P/E", f"{info.get('forwardPE', 'N/A')}")
                c4.metric("市場 Beta", f"{info.get('beta', 'N/A')}")

# --- Tab 2 & 3 維持原本邏輯 ---
with tab2:
    st.header("🛡️ 組合風險量化測試")
    p_df = pd.DataFrame([{"代號": "NVDA", "金額": 5000}, {"代號": "VOO", "金額": 5000}])
    edited = st.data_editor(p_df, num_rows="dynamic", key="final_p_edit")
    if st.button("開始評估"):
        total = edited["金額"].sum()
        w_beta = 0
        p_list = []
        for _, row in edited.iterrows():
            _, i, _ = fetch_financial_data(row["代號"])
            if i:
                b = i.get('beta', 1.0)
                w_beta += b * (row["金額"] / total)
                p_list.append({"股票": row["代號"], "權重": row["金額"] / total})
        st.metric("組合加權 Beta 值", f"{w_beta:.2f}")
        st.plotly_chart(px.pie(p_list, values='權重', names='股票', hole=0.4, title="資產配比"))

with tab3:
    st.header("📖 模型說明")
    st.markdown("本系統採用 **OLS 線性回歸** 進行預測，並實作了 **毛玻璃介面** 與 **數據補全** 技術。")
