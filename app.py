import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. 核心數據引擎 (AI 預測與自動修正)
# ==========================================

@st.cache_data(ttl=3600)
def fetch_financial_data(ticker_name):
    """抓取數據並自動清洗代號 (如 BRK/B -> BRK-B)"""
    try:
        clean_ticker = ticker_name.upper().replace("/", "-").strip()
        ticker_obj = yf.Ticker(clean_ticker)
        # 抓取 2 年數據確保 200MA 完整且連貫
        history = ticker_obj.history(period="2y")
        try:
            info = ticker_obj.info
        except:
            info = {}
        if history.empty:
            return None, None, f"找不到代號 {clean_ticker}"
        return history, info, None
    except Exception as e:
        return None, None, str(e)

def calculate_indicators(df):
    """計算 RSI、10MA、50MA、200MA"""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # 三重專業均線
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    return df

def get_ai_prediction_model(df, days=7):
    """AI 線性回歸預測與動態信心區間"""
    df_pred = df.tail(120).reset_index()
    df_pred['Date_num'] = pd.to_datetime(df_pred['Date']).apply(lambda x: x.toordinal())
    
    X = df_pred[['Date_num']].values
    y = df_pred['Close'].values
    model = LinearRegression().fit(X, y)
    
    last_date_num = df_pred['Date_num'].max()
    future_dates_num = np.array([last_date_num + i for i in range(1, days + 1)]).reshape(-1, 1)
    future_preds = model.predict(future_dates_num)
    
    # 計算信心區間 (使用最近波動度)
    std_dev = df['Close'].tail(30).std() * 0.7
    
    last_date = df_pred['Date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    return future_dates, future_preds, std_dev

# ==========================================
# 2. 全局視覺美化 (解決字體與表格色差)
# ==========================================

st.set_page_config(page_title="AlphaCheck Elite 20.0", layout="wide")

# 加入深層 CSS：解決側邊欄色差、字體看不清、表格顏色生硬問題
st.markdown("""
    <style>
    /* 全局背景一致化 */
    .stApp, [data-testid="stSidebar"] { background-color: #0f172a !important; }
    
    /* 強制所有文字為清晰白灰色，解決黑色字體問題 */
    h1, h2, h3, p, span, label, .stMarkdown { color: #f1f5f9 !important; }
    
    /* 修正 Metrics 標籤與數值顏色 */
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 16px !important; }
    [data-testid="stMetricValue"] { color: #60a5fa !important; font-weight: bold !important; }

    /* 指令盒：專業毛玻璃與發光邊框 */
    .decision-card {
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 25px;
        border: 1px solid;
        backdrop-filter: blur(10px);
    }

    /* 解決第二頁表格與按鈕的人機感 */
    div[data-testid="stDataEditor"] {
        border: 1px solid #334155 !important;
        border-radius: 10px;
        background-color: #1e293b !important;
    }
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ AlphaCheck Elite: 專業投資決策終端")
st.caption(f"數位金融科技系 | 最終修復版 20.0 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# --- 側邊欄：質感美債中心 ---
with st.sidebar:
    st.markdown("### 🌍 市場監控中心")
    tnx_h, _, _ = fetch_financial_data("^TNX")
    if tnx_h is not None:
        cur_y = tnx_h['Close'].iloc[-1]
        st.metric("美債 10Y 殖利率", f"{cur_y:.2f}%", delta=f"{cur_y - tnx_h['Close'].iloc[-2]:.2f}%")
        # 側邊欄線條柔和化
        fig_side = px.line(tnx_h.tail(45), y='Close', template="plotly_dark")
        fig_side.update_traces(line_color='#60a5fa', line_width=2)
        fig_side.update_layout(height=130, margin=dict(l=0,r=0,t=0,b=0), xaxis_visible=False, yaxis_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_side, use_container_width=True, config={'displayModeBar': False})
    st.divider()
    st.info("💡 系統已同步 UI 質感並啟用 AI 決策引擎。")

tab1, tab2, tab3 = st.tabs(["🔍 AI 個股診斷", "🛡️ 投資組合風險", "📖 模型理論說明"])

# --- Tab 1: AI 診斷 (修正 MA 標註與字體) ---
with tab1:
    col_in, _ = st.columns([2, 2])
    raw_input = col_in.text_input("輸入美股代號 (如: BRK/B, VOO, NVDA)", "NVDA")
    
    if raw_input:
        target = raw_input.upper().replace("/", "-").strip()
        with st.spinner(f'AI 引擎正在掃描 {target} 市場數據...'):
            hist, info, err = fetch_financial_data(target)
            
            if err:
                st.error(f"分析失敗：{err}")
            else:
                hist = calculate_indicators(hist)
                f_dates, f_preds, std = get_ai_prediction_model(hist)
                
                # --- A. 決策盒：莫蘭迪配色與清晰字體 ---
                cur_p = hist['Close'].iloc[-1]
                target_p = f_preds[-1]
                expected_ret = ((target_p - cur_p) / cur_p) * 100
                
                if expected_ret > 2.0:
                    b_bg, b_border, b_text = "rgba(20, 83, 45, 0.4)", "#4ade80", "多頭趨勢 / Buy"
                elif expected_ret < -2.0:
                    b_bg, b_border, b_text = "rgba(127, 29, 29, 0.4)", "#f87171", "空頭預警 / Sell"
                else:
                    b_bg, b_border, b_text = "rgba(30, 41, 59, 0.6)", "#94a3b8", "盤整觀望 / Neutral"

                st.markdown(f"""
                    <div class="decision-card" style="background-color: {b_bg}; border-color: {b_border};">
                        <h3 style="margin:0; color: white !important; font-weight: 500;">🤖 AI 智能評級：{b_text}</h3>
                        <p style="margin-top:10px; font-size:18px; color: white !important;">
                            預估 7 日目標：<b>${target_p:.2f}</b> | 預期收益：<b>{expected_ret:+.2f}%</b>
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                # --- B. Plotly 圖表：修正標註文字為白色 ---
                plot_data = hist.tail(150)
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=plot_data.index, open=plot_data['Open'], high=plot_data['High'], low=plot_data['Low'], close=plot_data['Close'], name='歷史K線', increasing_line_color='#4ade80', decreasing_line_color='#f87171'))
                
                # 三重均線
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA10'], line=dict(color='#81d4fa', width=1), name='10MA (短)'))
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA50'], line=dict(color='#fbbf24', width=1.2), name='50MA (中)'))
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA200'], line=dict(color='#94a3b8', width=2), name='200MA (生命線)'))
                
                # AI 預測陰影與路徑
                fig.add_trace(go.Scatter(x=f_dates + f_dates[::-1], y=list(f_preds + std) + list(f_preds - std)[::-1], fill='toself', fillcolor='rgba(96, 165, 250, 0.1)', line_color='rgba(0,0,0,0)', name='AI 波動預期'))
                fig.add_trace(go.Scatter(x=f_dates, y=f_preds, line=dict(color='#60a5fa', dash='dot', width=2), name='AI 預測路徑'))
                
                fig.update_layout(
                    template="plotly_dark", height=600, xaxis_rangeslider_visible=False,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    # --- 重點修復：圖例文字顏色 ---
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#f1f5f9", size=12))
                )
                st.plotly_chart(fig, use_container_width=True)

                # --- C. 指標卡片 ---
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("即時股價", f"${cur_p:.2f}")
                c2.metric("RSI 強弱", f"{hist['RSI'].iloc[-1]:.1f}")
                c3.metric("預估 P/E", f"{info.get('forwardPE', 'N/A')}")
                c4.metric("市場 Beta", f"{info.get('beta', 'N/A')}")

# --- Tab 2: 組合風險 (視覺大補強) ---
with tab2:
    st.markdown("### 🛡️ 投資組合風險量化壓力測試")
    # 使用玻璃感容器包裝
    with st.container():
        col_table, col_tips = st.columns([3, 1])
        with col_table:
            p_df = pd.DataFrame([{"代號": "NVDA", "金額": 5000}, {"代號": "VOO", "金額": 5000}])
            edited = st.data_editor(p_df, num_rows="dynamic", key="v20_final_edit", use_container_width=True)
        with col_tips:
            st.markdown("""
            **操作指南**
            - 在左表輸入持有標的代號
            - 輸入投資總金額 (USD)
            - 系統自動計算加權 Beta
            """)
            run_btn = st.button("🚀 開始量化分析")
            
    if run_btn:
        total = edited["金額"].sum()
        w_beta = 0
        p_list = []
        for _, row in edited.iterrows():
            _, i, _ = fetch_financial_data(row["代號"])
            if i:
                b = i.get('beta', 1.0)
                w = row["金額"] / total
                w_beta += b * w
                p_list.append({"股票": row["代號"].upper(), "權重": w})
        
        ca, cb = st.columns(2)
        with ca:
            st.plotly_chart(px.pie(p_list, values='權重', names='股票', hole=0.4, title="資產權重配比", template="plotly_dark"))
        with cb:
            st.metric("組合加權 Beta 值", f"{w_beta:.2f}")
            risk_lv = "高 (積極型)" if w_beta > 1.3 else "中 (穩健型)" if w_beta >= 0.9 else "低 (防禦型)"
            st.markdown(f"""
                <div style="background-color: #1e293b; padding: 25px; border-radius: 12px; border: 1px solid #334155;">
                    <h4 style="margin:0; opacity: 0.8;">總體風險等級</h4>
                    <h1 style="color: #60a5fa; margin: 10px 0;">{risk_lv}</h1>
                    <p style="margin:0; opacity: 0.7;">此組合相對於 S&P 500 指數的市場敏感度為 {w_beta:.2f} 倍。</p>
                </div>
            """, unsafe_allow_html=True)

# --- Tab 3: 理論說明 ---
with tab3:
    st.header("📖 系統模型與參數說明")
    st.markdown("""
    本系統由 **數位金融科技系** 專案團隊開發，核心包含：
    1. **AI 預測引擎**：採用 `OLS Linear Regression` 針對 120 日收盤價進行趨勢擬合。
    2. **三重均線系統**：利用 10MA、50MA、200MA 進行多維度趨勢確認。
    3. **數據補全技術**：抓取 2 年歷史數據以確保長線指標（200MA）之連續性。
    4. **組合風險模型**：基於馬可維茲資產定價理論之 $\\beta$ 風險係數。
    """)
