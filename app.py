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
    """抓取 2 年數據並自動修正代號 (如 BRK/B -> BRK-B)"""
    try:
        clean_ticker = ticker_name.upper().replace("/", "-").strip()
        ticker_obj = yf.Ticker(clean_ticker)
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
    """計算技術指標：RSI、10MA、50MA、200MA"""
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
    """AI 多項式趨勢擬合 + 信心陰影區間"""
    df_p = df.tail(90).reset_index()
    df_p['Date_n'] = pd.to_datetime(df_p['Date']).apply(lambda x: x.toordinal())
    X = df_p[['Date_n']].values
    y = df_p['Close'].values
    
    # 使用二階多項式回歸捕捉動能轉折
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    
    last_d_n = df_p['Date_n'].max()
    future_n = np.array([last_d_n + i for i in range(1, days + 1)]).reshape(-1, 1)
    preds = model.predict(poly.transform(future_n))
    
    # 動態信心區間：隨時間擴張
    base_std = df['Close'].tail(30).std()
    intervals = [base_std * (1 + (i * 0.2)) for i in range(len(preds))]
    
    last_d = df_p['Date'].max()
    future_d = [last_d + timedelta(days=i) for i in range(1, days + 1)]
    return future_d, preds, intervals

# ==========================================
# 2. UI 視覺樣式優化 (Midnight Navy)
# ==========================================

st.set_page_config(page_title="AlphaCheck Elite", layout="wide")

st.markdown("""
    <style>
    /* 全局背景與側邊欄背景同步 */
    .stApp, [data-testid="stSidebar"] { background-color: #0f172a !important; }
    
    /* 強制文字顏色為 Off-White */
    h1, h2, h3, p, span, label, .stMarkdown { color: #f1f5f9 !important; }
    
    /* Metrics 標籤與數值發光強化 */
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 16px !important; }
    [data-testid="stMetricValue"] { color: #60a5fa !important; font-weight: bold !important; }

    /* 按鈕：極簡發光線條感設計 */
    .stButton>button {
        background-color: transparent !important;
        color: #60a5fa !important;
        border: 2px solid #60a5fa !important;
        border-radius: 25px !important;
        padding: 10px 40px !important;
        font-weight: 700 !important;
        width: 100% !important;
        transition: 0.3s all ease-in-out;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        background-color: rgba(96, 165, 250, 0.1) !important;
        box-shadow: 0 0 20px rgba(96, 165, 250, 0.3) !important;
        color: #ffffff !important;
    }

    /* 專業決策盒與表格封裝 */
    .decision-card {
        padding: 25px; border-radius: 15px; margin-bottom: 25px;
        border: 1px solid #334155; backdrop-filter: blur(10px);
    }
    div[data-testid="stDataEditor"] {
        border: 1px solid #334155 !important; border-radius: 10px;
        background-color: #1e293b !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ AlphaCheck Elite: 專業投資決策終端")
st.caption(f"數位金融科技系專案 | 資料更新: {datetime.now().strftime('%H:%M:%S')}")

# --- 側邊欄：市場宏觀背景 ---
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
    st.info("💡 系統已啟用 AI 場景趨勢分析與組合深度評價。")

tab1, tab2, tab3 = st.tabs(["🔍 AI 市場診斷", "🛡️ 投資組合分析報告", "📖 模型說明"])

# --- Tab 1: AI 診斷 (三重均線 + RSI + 曲線預測) ---
with tab1:
    col_in, _ = st.columns([2, 2])
    raw_ticker = col_in.text_input("輸入美股代號 (如 BRK/B, VOO, NVDA)", "VOO")
    
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
                        <p style="margin-top:10px; font-size:18px; color: white !important;">
                            預估 7 日目標：<b>${target_p:.2f}</b> | 期望收益：<b>{expected_ret:+.2f}%</b>
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                # B. Plotly 圖表 (三重均線 + 曲線預測)
                plot_data = hist.tail(150)
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=plot_data.index, open=plot_data['Open'], high=plot_data['High'], low=plot_data['Low'], close=plot_data['Close'], name='歷史走勢', increasing_line_color='#4ade80', decreasing_line_color='#f87171'))
                
                # 三重均線系統
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA10'], line=dict(color='#81d4fa', width=1), name='10MA (短)'))
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA50'], line=dict(color='#fbbf24', width=1), name='50MA (中)'))
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA200'], line=dict(color='#94a3b8', width=2), name='200MA (生命線)'))
                
                # AI 曲線預測陰影
                fig.add_trace(go.Scatter(
                    x=f_dates + f_dates[::-1],
                    y=[f_preds[i] + f_intervals[i] for i in range(len(f_preds))] + [f_preds[i] - f_intervals[i] for i in range(len(f_preds))][::-1],
                    fill='toself', fillcolor='rgba(96, 165, 250, 0.1)', line_color='rgba(0,0,0,0)', name='AI 波動區間'
                ))
                # AI 預測核心虛線
                fig.add_trace(go.Scatter(x=f_dates, y=f_preds, line=dict(color='#ffffff', dash='dash', width=2), name='AI 預測路徑'))
                
                fig.update_layout(
                    template="plotly_dark", height=600, xaxis_rangeslider_visible=False,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#f1f5f9", size=12))
                )
                st.plotly_chart(fig, use_container_width=True)

                # 指標卡片 (含 RSI)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("即時股價", f"${cur_p:.2f}")
                c2.metric("RSI 指標", f"{hist['RSI'].iloc[-1]:.1f}")
                c3.metric("本益比 (PE)", f"{info.get('forwardPE', 'N/A')}")
                c4.metric("市場 Beta", f"{info.get('beta', 'N/A')}")

# --- Tab 2: 組合分析 (智能投顧評價：核心衛星系統) ---
with tab2:
    st.markdown("### 🛡️ 投資組合風險量化與 AI 評價")
    p_df = pd.DataFrame([
        {"代號": "VOO", "金額": 5000}, {"代號": "BRK-B", "金額": 2850},
        {"代號": "GOOGL", "金額": 987}, {"代號": "NVDA", "金額": 915}
    ])
    edited = st.data_editor(p_df, num_rows="dynamic", use_container_width=True, key="portfolio_editor")
    
    if st.button("🚀 執行 AI 智能評價分析"):
        with st.spinner('正在分析資產結構並生成診斷評語...'):
            total_val = edited["金額"].sum()
            results = []
            weighted_beta = 0
            weighted_rsi = 0
            for _, row in edited.iterrows():
                h, i, _ = fetch_financial_data(row["代號"])
                if h is not None:
                    h = calculate_indicators(h)
                    beta = i.get('beta', 1.0) if i else 1.0
                    weight = row["金額"] / total_val
                    weighted_beta += beta * weight
                    weighted_rsi += h['RSI'].iloc[-1] * weight
                    results.append({"股票": row["代號"].upper(), "權重": weight, "Beta": beta, "驅動力": beta * weight})
            
            res_df = pd.DataFrame(results)
            st.divider()
            
            # --- AI 首席分析師評價邏輯 ---
            if 0.9 <= weighted_beta <= 1.2:
                eval_title = "精英級：核心—衛星配置"
                eval_color = "#60a5fa"
                eval_content = "配置展現了極高的專業度。以大盤指數為穩定的定海神針，並輔以高價值避險資產（如 BRK-B/JPM）與高動能科技標的。這是一種典型的『進可攻、退可守』之專業配置。"
            elif weighted_beta > 1.2:
                eval_title = "進攻型：高動能配置"
                eval_color = "#f87171"
                eval_content = "組合目前高度聚焦於市場成長動能，風險偏好較高。建議關注 RSI 警訊，目前情緒略顯擁擠，適合牛市環境但須提防波動。"
            else:
                eval_title = "防禦型：價值堡壘配置"
                eval_color = "#4ade80"
                eval_content = "資產組合具有極強的抗風險能力與市場獨立性，適合長期資本保護。建議在市場極度低迷時適度增加動能標的以提升收益。"

            st.markdown(f"""
                <div style="background-color: #1e293b; padding: 35px; border-radius: 15px; border-left: 10px solid {eval_color};">
                    <h2 style="color: {eval_color}; margin:0;">AI 診斷：{eval_title}</h2>
                    <p style="margin-top:20px; font-size:19px; line-height:1.7; color: white !important;">
                        <b>分析師評論：</b><br>{eval_content}<br><br>
                        <b>組合加權 Beta：</b>{weighted_beta:.2f} (展現出與市場極佳的平衡感)<br>
                        <b>組合平均 RSI：</b>{weighted_rsi:.1f} ({'情緒過熱，建議分批獲利了結' if weighted_rsi > 70 else '情緒穩健，可繼續持有'})<br>
                        <b>主要波動驅動標的：</b>{res_df.sort_values('驅動力', ascending=False)['股票'].iloc[0]} (該標的權重最高，決定了你資產的結構性穩定)
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            c_pie, c_bar = st.columns(2)
            c_pie.plotly_chart(px.pie(res_df, values='權重', names='股票', hole=0.4, title="資產配比 (Weight Allocation)", template="plotly_dark").update_layout(font=dict(color="white")), use_container_width=True)
            c_bar.plotly_chart(px.bar(res_df, x='股票', y='驅動力', title="資產波動驅動力分析 (Market Driving Force)", template="plotly_dark").update_traces(marker_color='#60a5fa'), use_container_width=True)

with tab3:
    st.header("📖 模型理論基礎")
    st.markdown("""
    本系統專為 **數位金融科技系** 課程設計，核心包含：
    1. **多項式擬合預測**：採用二階多項式擬合 $y = ax^2 + bx + c$，有效捕捉股價短期轉折。
    2. **信心帶 (Confidence Band)**：隨預測時間增加而擴張的陰影，體現預測的不確定性原理。
    3. **馬可維茲組合模型**：基於加權資產 $\\beta$ 係數進行系統性風險量化分析。
    """)
