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
    """抓取 2 年數據並處理代號自動修正"""
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
    """計算 RSI 與三重均線 (10, 50, 200)"""
    # RSI 計算
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

def get_ai_prediction_model(df, days=7):
    """AI 曲線擬合 + 信心陰影區間"""
    df_p = df.tail(90).reset_index()
    df_p['Date_n'] = pd.to_datetime(df_p['Date']).apply(lambda x: x.toordinal())
    X = df_p[['Date_n']].values
    y = df_p['Close'].values
    
    # 多項式回歸捕捉轉折
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    
    last_d_n = df_p['Date_n'].max()
    future_n = np.array([last_d_n + i for i in range(1, days + 1)]).reshape(-1, 1)
    preds = model.predict(poly.transform(future_n))
    
    # 信心區間隨時間擴張
    base_std = df['Close'].tail(30).std()
    intervals = [base_std * (1 + (i * 0.2)) for i in range(len(preds))]
    
    last_d = df_p['Date'].max()
    future_d = [last_d + timedelta(days=i) for i in range(1, days + 1)]
    return future_d, preds, intervals

# ==========================================
# 2. UI 視覺設計 (Midnight Navy & Glassmorphism)
# ==========================================

st.set_page_config(page_title="AlphaCheck Elite", layout="wide")

st.markdown("""
    <style>
    /* 全局背景色一致化 */
    .stApp, [data-testid="stSidebar"] { background-color: #0f172a !important; }
    h1, h2, h3, p, span, label, .stMarkdown { color: #f1f5f9 !important; }
    
    /* Metrics 與標籤清晰度 */
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 16px !important; }
    [data-testid="stMetricValue"] { color: #60a5fa !important; font-weight: bold !important; }

    /* 按鈕：質感發光線條設計 */
    .stButton>button {
        background-color: transparent !important;
        color: #60a5fa !important;
        border: 2px solid #60a5fa !important;
        border-radius: 25px !important;
        padding: 10px 40px !important;
        font-weight: 700 !important;
        width: 100% !important;
        transition: 0.3s all ease-in-out;
    }
    .stButton>button:hover {
        background-color: rgba(96, 165, 250, 0.1) !important;
        box-shadow: 0 0 20px rgba(96, 165, 250, 0.3) !important;
    }

    /* 專業決策卡片與表格包裝 */
    .decision-card {
        padding: 25px; border-radius: 12px; margin-bottom: 25px;
        border: 1px solid #334155; backdrop-filter: blur(10px);
    }
    div[data-testid="stDataEditor"] {
        border: 1px solid #334155 !important; border-radius: 10px;
        background-color: #1e293b !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ AlphaCheck Elite: 專業投資決策終端")
st.caption(f"數位金融科技系專案 | 即時數據更新: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# --- 側邊欄 ---
with st.sidebar:
    st.markdown("### 🌍 市場監控中心")
    tnx_h, _, _ = fetch_financial_data("^TNX")
    if tnx_h is not None:
        cur_y = tnx_h['Close'].iloc[-1]
        st.metric("美債 10Y 殖利率", f"{cur_y:.2f}%", delta=f"{cur_y - tnx_h['Close'].iloc[-2]:.2f}%")
        fig_side = px.line(tnx_h.tail(45), y='Close', template="plotly_dark")
        fig_side.update_traces(line_color='#60a5fa', line_width=2)
        fig_side.update_layout(height=130, margin=dict(l=0,r=0,t=0,b=0), xaxis_visible=False, yaxis_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_side, use_container_width=True, config={'displayModeBar': False})
    st.divider()
    st.info("💡 系統已啟用 AI 信心區間預測與多重指標監控模式。")

tab1, tab2, tab3 = st.tabs(["🔍 AI 市場深度診斷", "🛡️ 投資組合分析報告", "📖 模型理論說明"])

# --- Tab 1: AI 診斷 (三重均線 + RSI + 陰影預測) ---
with tab1:
    col_in, _ = st.columns([2, 2])
    raw_ticker = col_in.text_input("輸入美股代號 (如 BRK/B, VOO, NVDA)", "NVDA")
    
    if raw_ticker:
        target = raw_ticker.upper().replace("/", "-").strip()
        with st.spinner('正在分析市場趨勢與技術指標...'):
            hist, info, err = fetch_financial_data(target)
            if not err:
                hist = calculate_indicators(hist)
                f_dates, f_preds, f_intervals = get_ai_prediction_model(hist)
                
                # A. 決策卡片
                cur_p = hist['Close'].iloc[-1]
                target_p = f_preds[-1]
                expected_ret = ((target_p - cur_p) / cur_p) * 100
                
                if expected_ret > 2.0: b_bg, b_border, b_text = "rgba(20, 83, 45, 0.4)", "#4ade80", "多頭趨勢 / Buy"
                elif expected_ret < -2.0: b_bg, b_border, b_text = "rgba(127, 29, 29, 0.4)", "#f87171", "空頭預警 / Sell"
                else: b_bg, b_border, b_text = "rgba(30, 41, 59, 0.6)", "#94a3b8", "中立觀望 / Hold"

                st.markdown(f"""
                    <div class="decision-card" style="background-color: {b_bg}; border-color: {b_border};">
                        <h3 style="margin:0; color: white !important;">🤖 AI 智能評級：{b_text}</h3>
                        <p style="margin-top:8px; font-size:18px; color: white !important;">
                            預估 7 日目標：<b>${target_p:.2f}</b> | 期望收益：<b>{expected_ret:+.2f}%</b>
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                # B. Plotly 圖表 (三重均線 + 陰影預測)
                plot_data = hist.tail(150)
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=plot_data.index, open=plot_data['Open'], high=plot_data['High'], low=plot_data['Low'], close=plot_data['Close'], name='歷史走勢', increasing_line_color='#4ade80', decreasing_line_color='#f87171'))
                
                # 三重均線畫上去
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA10'], line=dict(color='#81d4fa', width=1), name='10MA (短)'))
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA50'], line=dict(color='#fbbf24', width=1.2), name='50MA (中)'))
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA200'], line=dict(color='#94a3b8', width=2), name='200MA (長)'))
                
                # AI 預測陰影 (信心區間)
                fig.add_trace(go.Scatter(
                    x=f_dates + f_dates[::-1],
                    y=[f_preds[i] + f_intervals[i] for i in range(len(f_preds))] + [f_preds[i] - f_intervals[i] for i in range(len(f_preds))][::-1],
                    fill='toself', fillcolor='rgba(96, 165, 250, 0.1)', line_color='rgba(0,0,0,0)', name='AI 波動預期'
                ))
                # AI 核心路徑虛線
                fig.add_trace(go.Scatter(x=f_dates, y=f_preds, line=dict(color='#ffffff', dash='dash', width=2), name='AI 預測路徑'))
                
                fig.update_layout(
                    template="plotly_dark", height=600, xaxis_rangeslider_visible=False,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#f1f5f9", size=12))
                )
                st.plotly_chart(fig, use_container_width=True)

                # C. 指標卡片 (含 RSI)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("即時股價", f"${cur_p:.2f}")
                c2.metric("RSI (相對強弱)", f"{hist['RSI'].iloc[-1]:.1f}")
                c3.metric("本益比 (P/E)", f"{info.get('forwardPE', 'N/A')}")
                c4.metric("市場 Beta", f"{info.get('beta', 'N/A')}")

# --- Tab 2: 投資組合分析 (含智能投顧評價) ---
with tab2:
    st.markdown("### 🛡️ 投資組合風險量化與 AI 評價")
    p_df = pd.DataFrame([{"代號": "VOO", "金額": 5000}, {"代號": "NVDA", "金額": 2000}])
    edited = st.data_editor(p_df, num_rows="dynamic", use_container_width=True, key="portfolio_editor")
    
    if st.button("🚀 執行 AI 智能評價分析"):
        total = edited["金額"].sum()
        results = []
        weighted_beta = 0
        weighted_rsi = 0
        
        for _, row in edited.iterrows():
            h, i, _ = fetch_financial_data(row["代號"])
            if h is not None:
                h = calculate_indicators(h)
                beta = i.get('beta', 1.0) if i else 1.0
                weight = row["金額"] / total
                weighted_beta += beta * weight
                weighted_rsi += h['RSI'].iloc[-1] * weight
                results.append({"股票": row["代號"].upper(), "權重": weight, "Beta": beta, "貢獻": beta * weight})
        
        res_df = pd.DataFrame(results)
        
        # AI 智能評價邏輯
        st.divider()
        risk_color = "#4ade80" if weighted_beta < 0.9 else "#fbbf24" if weighted_beta <= 1.3 else "#f87171"
        risk_lv = "防禦型" if weighted_beta < 0.9 else "穩健型" if weighted_beta <= 1.3 else "積極型"
        
        st.markdown(f"""
            <div style="background-color: #1e293b; padding: 30px; border-radius: 15px; border-left: 10px solid {risk_color};">
                <h2 style="color: {risk_color}; margin:0;">AI 組合評價：{risk_lv}資產配置</h2>
                <p style="margin-top:15px; font-size:18px; color: white !important;">
                    <b>加權 Beta 指數：</b>{weighted_beta:.2f}<br>
                    <b>組合平均 RSI：</b>{weighted_rsi:.1f}<br>
                    <b>核心驅動標的：</b>{res_df.sort_values('貢獻', ascending=False)['股票'].iloc[0]}
                </p>
                <p style="opacity:0.7; font-size:14px;">※ 評價基於市場敏感度與情緒指標綜合計算。</p>
            </div>
        """, unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(px.pie(res_df, values='權重', names='股票', hole=0.4, title="資產配比", template="plotly_dark"), use_container_width=True)
        with col_b:
            st.plotly_chart(px.bar(res_df, x='股票', y='貢獻', title="風險貢獻分析", template="plotly_dark").update_traces(marker_color='#60a5fa'), use_container_width=True)

with tab3:
    st.header("📖 模型理論基礎")
    st.markdown("""
    1. **多項式趨勢擬合**：利用二階回歸捕捉股價轉折，而非僵硬的直線。
    2. **信心帶 (Confidence Band)**：
       
       預測陰影會隨著時間向外擴張，體現出預測越遠、不確定性越高的金融特性。
    3. **RSI 指標**：衡量 14 日內平均漲跌幅度，評估市場超買或超賣狀態。
    """)
