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
# 1. 核心 AI 與數據引擎
# ==========================================

@st.cache_data(ttl=3600)
def fetch_financial_data(ticker_name):
    """抓取數據並自動修正代號 (如 BRK/B -> BRK-B)"""
    try:
        clean_ticker = ticker_name.upper().replace("/", "-").strip()
        ticker_obj = yf.Ticker(clean_ticker)
        history = ticker_obj.history(period="2y")
        info = ticker_obj.info if hasattr(ticker_obj, 'info') else {}
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
    # 三重均線系統
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    return df

def get_scenario_predictions(df, days=7):
    """
    進化版 AI：不再只是單一線條，而是 樂觀/中立/悲觀 三場景分析
    """
    df_p = df.tail(90).reset_index()
    df_p['Date_n'] = pd.to_datetime(df_p['Date']).apply(lambda x: x.toordinal())
    X = df_p[['Date_n']].values
    y = df_p['Close'].values
    
    # 使用多項式擬合捕捉轉折
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    
    last_d_n = df_p['Date_n'].max()
    future_n = np.array([last_d_n + i for i in range(1, days + 1)]).reshape(-1, 1)
    base_preds = model.predict(poly.transform(future_n))
    
    # 基於歷史波動度建立場景
    volatility = df['Close'].tail(30).std()
    bull_preds = base_preds + (volatility * 1.8)
    bear_preds = base_preds - (volatility * 1.8)
    
    last_d = df_p['Date'].max()
    future_d = [last_d + timedelta(days=i) for i in range(1, days + 1)]
    return future_d, bull_preds, base_preds, bear_preds

# ==========================================
# 2. 全局視覺美化 CSS
# ==========================================

st.set_page_config(page_title="AlphaCheck Elite", layout="wide")

st.markdown("""
    <style>
    /* 全局深海藍背景 */
    .stApp, [data-testid="stSidebar"] { background-color: #0f172a !important; }
    
    /* 強制文字顏色，解決黑色字體看不清的問題 */
    h1, h2, h3, p, span, label, .stMarkdown { color: #f1f5f9 !important; }
    
    /* Metrics 與 Alert 樣式 */
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 16px !important; }
    [data-testid="stMetricValue"] { color: #60a5fa !important; font-weight: bold !important; }
    .stAlert { background-color: #1e293b !important; border: 1px solid #334155 !important; color: white !important; }

    /* 表格樣式修正 */
    div[data-testid="stDataEditor"] { background-color: #1e293b !important; border: 1px solid #334155 !important; border-radius: 10px; }

    /* 現代感按鈕：發光邊框設計 */
    .stButton>button {
        background-color: transparent !important;
        color: #60a5fa !important;
        border: 2px solid #60a5fa !important;
        border-radius: 25px !important;
        padding: 10px 40px !important;
        font-weight: 700 !important;
        transition: 0.3s all ease-in-out !important;
        width: 100% !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        background-color: rgba(96, 165, 250, 0.1) !important;
        box-shadow: 0 0 20px rgba(96, 165, 250, 0.3) !important;
        color: #ffffff !important;
    }

    /* 專業決策卡片 */
    .report-card {
        padding: 25px; border-radius: 15px; margin-bottom: 25px;
        border: 1px solid #334155; backdrop-filter: blur(10px);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ AlphaCheck Elite: 專業投資決策終端")
st.caption(f"數位金融科技系專案 | 即時數據更新: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# --- 側邊欄：市場快訊 ---
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
    st.divider()
    st.info("💡 系統已啟用多項式場景模擬與加權風險評估模組。")

tab1, tab2, tab3 = st.tabs(["🔍 AI 市場深度診斷", "🛡️ 投資組合風險報告", "📖 模型理論說明"])

# --- Tab 1: AI 診斷 ---
with tab1:
    col_in, _ = st.columns([2, 2])
    raw_ticker = col_in.text_input("輸入美股代號 (如: BRK/B, VOO, NVDA)", "NVDA")
    
    if raw_ticker:
        target = raw_ticker.upper().replace("/", "-").strip()
        with st.spinner(f'AI 引擎正在計算 {target} 之多路徑走勢...'):
            hist, info, err = fetch_financial_data(target)
            
            if err:
                st.error(f"分析失敗：{err}")
            else:
                hist = calculate_indicators(hist)
                f_dates, bull, base, bear = get_scenario_predictions(hist)
                
                # A. 視覺化決策指令盒
                cur_p = hist['Close'].iloc[-1]
                target_p = base[-1]
                expected_ret = ((target_p - cur_p) / cur_p) * 100
                
                if expected_ret > 2.0: b_bg, b_border, b_text = "rgba(20, 83, 45, 0.4)", "#4ade80", "多頭趨勢 / Buy"
                elif expected_ret < -2.0: b_bg, b_border, b_text = "rgba(127, 29, 29, 0.4)", "#f87171", "空頭預警 / Sell"
                else: b_bg, b_border, b_text = "rgba(30, 41, 59, 0.6)", "#94a3b8", "盤整觀望 / Neutral"

                st.markdown(f"""
                    <div class="report-card" style="background-color: {b_bg}; border-color: {b_border};">
                        <h3 style="margin:0; color: white !important;">🤖 AI 投資指令：{b_text}</h3>
                        <p style="margin-top:10px; font-size:18px; color: white !important;">
                            核心預測目標: <b>${target_p:.2f}</b> | 期望收益: <b>{expected_ret:+.2f}%</b>
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                # B. 專業 Plotly 圖表 (強制修復標註字體顏色)
                plot_data = hist.tail(120)
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=plot_data.index, open=plot_data['Open'], high=plot_data['High'], low=plot_data['Low'], close=plot_data['Close'], name='歷史走勢', increasing_line_color='#4ade80', decreasing_line_color='#f87171'))
                
                # 三重均線
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA10'], line=dict(color='#81d4fa', width=1), name='10MA (短)'))
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA50'], line=dict(color='#fbbf24', width=1.2), name='50MA (中)'))
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA200'], line=dict(color='#94a3b8', width=2), name='200MA (生命線)'))
                
                # AI 場景路徑
                fig.add_trace(go.Scatter(x=f_dates, y=bull, line=dict(color='#4ade80', dash='dot', width=1), name='樂觀場景'))
                fig.add_trace(go.Scatter(x=f_dates, y=base, line=dict(color='#ffffff', dash='dash', width=3), name='核心路徑'))
                fig.add_trace(go.Scatter(x=f_dates, y=bear, line=dict(color='#f87171', dash='dot', width=1), name='悲觀場景'))
                
                fig.update_layout(
                    template="plotly_dark", height=600, xaxis_rangeslider_visible=False,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#f1f5f9", size=12))
                )
                st.plotly_chart(fig, use_container_width=True)

                # C. 底部指標卡片
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("即時股價", f"${cur_p:.2f}")
                c2.metric("RSI 指標", f"{hist['RSI'].iloc[-1]:.1f}")
                c3.metric("本益比 (PE)", f"{info.get('forwardPE', 'N/A')}")
                c4.metric("Beta 風險係數", f"{info.get('beta', 'N/A')}")

# --- Tab 2: 組合風險 (視覺大補強) ---
with tab2:
    st.markdown("### 🛡️ 投資組合風險量化報告")
    # 預設數據表格
    default_data = pd.DataFrame([{"代號": "NVDA", "金額": 5000}, {"代號": "VOO", "金額": 5000}])
    edited = st.data_editor(default_data, num_rows="dynamic", use_container_width=True, key="portfolio_editor")
    
    st.write("") # 留白
    if st.button("🚀 執行深度壓力測試"):
        with st.spinner('正在分析資產相關性與 Beta 貢獻度...'):
            total_val = edited["金額"].sum()
            results = []
            weighted_beta = 0
            
            for _, row in edited.iterrows():
                _, i, _ = fetch_financial_data(row["代號"])
                beta = i.get('beta', 1.0) if i else 1.0
                weight = row["金額"] / total_val
                weighted_beta += beta * weight
                results.append({"股票": row["代號"].upper(), "權重": weight, "Beta": beta, "貢獻": beta * weight})
            
            res_df = pd.DataFrame(results)
            
            # 視覺化報告卡
            st.divider()
            col_a, col_b = st.columns([2, 3])
            
            with col_a:
                fig_pie = px.pie(res_df, values='權重', names='股票', hole=0.4, title="資產配比", template="plotly_dark")
                fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col_b:
                risk_color = "#4ade80" if weighted_beta < 0.9 else "#fbbf24" if weighted_beta <= 1.3 else "#f87171"
                risk_lv = "低 (防禦型)" if weighted_beta < 0.9 else "中 (穩健型)" if weighted_beta <= 1.3 else "高 (積極型)"
                
                st.markdown(f"""
                    <div style="background-color: #1e293b; padding: 30px; border-radius: 15px; border-left: 10px solid {risk_color};">
                        <h4 style="margin:0; opacity:0.8; color: white !important;">組合加權 Beta 指數</h4>
                        <h1 style="color: {risk_color}; margin: 10px 0; font-size: 64px;">{weighted_beta:.2f}</h1>
                        <h3 style="margin:0; color: white !important;">風險等級：{risk_lv}</h3>
                        <p style="margin-top:15px; opacity:0.7; color: white !important;">
                            此組合相對於標普 500 的波動度約為 {weighted_beta:.2f} 倍。<br>
                            主要風險貢獻標的為：<b>{res_df.sort_values('貢獻', ascending=False)['股票'].iloc[0]}</b>
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                # 個股風險貢獻圖
                st.write("")
                fig_bar = px.bar(res_df, x='股票', y='貢獻', title="個股風險貢獻度分析", template="plotly_dark")
                fig_bar.update_traces(marker_color='#60a5fa')
                fig_bar.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
                st.plotly_chart(fig_bar, use_container_width=True)

# --- Tab 3: 理論說明 ---
with tab3:
    st.header("📖 系統設計與理論基礎")
    st.markdown("""
    本系統專為 **數位金融科技系** 課程設計，核心包含：
    
    1. **AI 場景模擬 (Scenario Analysis)**：
       採用二次多項式回歸擬合：$y = ax^2 + bx + c$
       並結合歷史波動率（$\sigma$）產出樂觀與悲觀邊界。
    
    2. **加權 Beta 模型**：
       $$\\beta_{portfolio} = \\sum_{i=1}^{n} w_i \\beta_i$$
       用於量化投資組合相對於市場的系統性風險。
       
    3. **三重均線系統**：
       結合 10MA (動能)、50MA (趨勢)、200MA (牛熊) 進行多維度判斷。
    """)
