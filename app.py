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
# 1. 核心引擎：數據處理與 AI 擬合
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
    # 三重均線系統
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    return df

def get_ai_prediction_model(df, days=7):
    df_p = df.tail(90).reset_index()
    df_p['Date_n'] = pd.to_datetime(df_p['Date']).apply(lambda x: x.toordinal())
    X = df_p[['Date_n']].values
    y = df_p['Close'].values
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    
    last_d_n = df_p['Date_n'].max()
    future_n = np.array([last_d_n + i for i in range(1, days + 1)]).reshape(-1, 1)
    preds = model.predict(poly.transform(future_n))
    
    base_std = df['Close'].tail(30).std()
    intervals = [base_std * (1 + (i * 0.2)) for i in range(len(preds))]
    
    last_d = df_p['Date'].max()
    future_d = [last_d + timedelta(days=i) for i in range(1, days + 1)]
    return future_d, preds, intervals

# ==========================================
# 2. UI 旗艦視覺設計
# ==========================================

st.set_page_config(page_title="AlphaCheck Elite", layout="wide")

st.markdown("""
    <style>
    .stApp, [data-testid="stSidebar"] { background-color: #0f172a !important; }
    h1, h2, h3, p, span, label, .stMarkdown { color: #f1f5f9 !important; }
    
    /* Metrics 標籤強化 */
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 16px !important; }
    [data-testid="stMetricValue"] { color: #60a5fa !important; font-weight: bold !important; }

    /* 按鈕：極簡發光線條感 */
    .stButton>button {
        background-color: transparent !important; color: #60a5fa !important;
        border: 2px solid #60a5fa !important; border-radius: 25px !important;
        width: 100% !important; font-weight: 700 !important;
        transition: 0.3s all;
    }
    .stButton>button:hover { 
        background-color: rgba(96, 165, 250, 0.1) !important;
        box-shadow: 0 0 20px rgba(96, 165, 250, 0.3) !important;
    }

    /* 專業決策卡片 */
    .report-card { padding: 30px; border-radius: 15px; border: 1px solid #334155; backdrop-filter: blur(10px); }
    div[data-testid="stDataEditor"] { border: 1px solid #334155 !important; border-radius: 10px; background-color: #1e293b !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ AlphaCheck Elite: 專業投資決策終端")

# --- 側邊欄 ---
with st.sidebar:
    st.markdown("### 🌍 市場監控中心")
    tnx_h, _, _ = fetch_financial_data("^TNX")
    if tnx_h is not None:
        st.metric("美債 10Y 殖利率", f"{tnx_h['Close'].iloc[-1]:.2f}%")
        fig_side = px.line(tnx_h.tail(45), y='Close', template="plotly_dark").update_traces(line_color='#60a5fa')
        fig_side.update_layout(height=130, margin=dict(l=0,r=0,t=0,b=0), xaxis_visible=False, yaxis_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_side, use_container_width=True)
    st.divider()
    st.info("💡 系統已啟用多向度風險偵測與動態語意評價系統。")

tab1, tab2, tab3 = st.tabs(["🔍 AI 市場診斷", "🛡️ 投資組合分析報告", "📖 模型理論說明"])

# --- Tab 1: AI 診斷 (均線 + RSI + 曲線陰影) ---
with tab1:
    col_in, _ = st.columns([2, 2])
    raw_ticker = col_in.text_input("輸入美股代號 (如 BRK/B, VOO, NVDA)", "NVDA")
    if raw_ticker:
        target = raw_ticker.upper().replace("/", "-").strip()
        hist, info, err = fetch_financial_data(target)
        if not err:
            hist = calculate_indicators(hist)
            f_dates, f_preds, f_intervals = get_ai_prediction_model(hist)
            cur_p = hist['Close'].iloc[-1]
            ret = ((f_preds[-1] - cur_p) / cur_p) * 100
            
            bg = "rgba(20, 83, 45, 0.4)" if ret > 2 else "rgba(127, 29, 29, 0.4)" if ret < -2 else "rgba(30, 41, 59, 0.6)"
            st.markdown(f"<div class='report-card' style='background-color:{bg}; border-color:white;'><h3>🤖 AI 評級</h3><p>預期變動：{ret:+.2f}%</p></div>", unsafe_allow_html=True)
            
            fig = go.Figure(data=[go.Candlestick(x=hist.tail(150).index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='走勢')])
            fig.add_trace(go.Scatter(x=hist.tail(150).index, y=hist['MA50'].tail(150), line=dict(color='#fbbf24', width=1), name='50MA'))
            fig.add_trace(go.Scatter(x=hist.tail(150).index, y=hist['MA200'].tail(150), line=dict(color='#94a3b8', width=2), name='200MA'))
            fig.add_trace(go.Scatter(x=f_dates+f_dates[::-1], y=[f_preds[i]+f_intervals[i] for i in range(len(f_preds))]+[f_preds[i]-f_intervals[i] for i in range(len(f_preds))][::-1], fill='toself', fillcolor='rgba(96, 165, 250, 0.1)', line_color='rgba(0,0,0,0)', name='AI 波動預期'))
            fig.add_trace(go.Scatter(x=f_dates, y=f_preds, line=dict(color='white', dash='dash', width=2), name='AI 路徑'))
            fig.update_layout(template="plotly_dark", height=550, xaxis_rangeslider_visible=False, legend=dict(font=dict(color="white")))
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("即時股價", f"${cur_p:.2f}")
            c2.metric("RSI (相對強弱)", f"{hist['RSI'].iloc[-1]:.1f}")
            c3.metric("本益比 (PE)", f"{info.get('forwardPE', 'N/A')}")
            c4.metric("市場 Beta", f"{info.get('beta', 'N/A')}")

# --- Tab 2: 真正「長腦袋」的分析報告 ---
with tab2:
    st.markdown("### 🛡️ 投資組合智能量化診斷報告")
    p_df = pd.DataFrame([
        {"代號": "QQQM", "金額": 5000}, {"代號": "QQQ", "金額": 5000},
        {"代號": "TSLA", "金額": 2000}, {"代號": "VOO", "金額": 3000}
    ])
    edited = st.data_editor(p_df, num_rows="dynamic", use_container_width=True, key="portfolio_table")
    
    if st.button("🚀 執行 AI 智能深度診斷"):
        with st.spinner('AI 正在偵測資產相關性與多樣性...'):
            total_val = edited["金額"].sum()
            results = []
            weighted_beta, weighted_rsi, tech_count = 0, 0, 0
            # 定義高成長科技清單
            tech_tickers = ['QQQ', 'QQQM', 'NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX', 'PLTR', 'ARKF', 'SOXX', 'SMH']

            for _, row in edited.iterrows():
                h, i, _ = fetch_financial_data(row["代號"])
                if h is not None:
                    h = calculate_indicators(h)
                    beta = i.get('beta', 1.0) if i else 1.0
                    weight = row["金額"] / total_val
                    weighted_beta += beta * weight
                    weighted_rsi += h['RSI'].iloc[-1] * weight
                    if row["代號"].upper() in tech_tickers: tech_count += 1
                    results.append({"股票": row["代號"].upper(), "權重": weight, "Beta": beta, "驅動力": beta * weight})
            
            res_df = pd.DataFrame(results)
            st.divider()
            
            # --- AI 診斷核心邏輯 (修正 Bug) ---
            top_stock = res_df.sort_values('權重', ascending=False)['股票'].iloc[0]
            tech_ratio = tech_count / len(res_df) if len(res_df) > 0 else 0
            has_redundancy = "QQQ" in res_df['股票'].values and "QQQM" in res_df['股票'].values

            if tech_ratio > 0.8 and weighted_beta > 1.1:
                eval_title = "高風險：極端成長型集中"
                eval_color = "#f87171"
                eval_content = f"警告：組合雖然具備結構，但實質上**極度向科技股傾斜**。這是一個典型的『進可攻、退不可守』配置。由於標的相關性過高，一旦科技產業修正，**{top_stock}** 與衛星標的將同步受挫。"
            elif has_redundancy:
                eval_title = "結構冗餘：標的重疊風險"
                eval_color = "#fbbf24"
                eval_content = "偵測到同時持有 QQQ 與 QQQM。這兩者追蹤同一指數，屬於冗餘配置，並未達到分散風險的效果。建議合併標的並引入其他產業（如金融、能源）以達成真正的『退可守』。"
            elif 0.9 <= weighted_beta <= 1.25 and tech_ratio < 0.6:
                eval_title = "精英級：均衡核心—衛星策略"
                eval_color = "#60a5fa"
                eval_content = f"配置展現了卓越的專業度。以 **{top_stock}** 為核心定海神針，並成功透過不同屬性的標的分散風險，是名副其實的『進可攻、退可守』配置。"
            else:
                eval_title = "穩健/防禦型資產配置"
                eval_color = "#4ade80"
                eval_content = f"組合抗風險能力強。以 **{top_stock}** 為主軸，能在市場動盪中提供出色的保護力，但在大牛市中增長幅度可能較慢。"

            st.markdown(f"""
                <div class="report-card" style="border-left: 10px solid {eval_color}; background-color: #1e293b;">
                    <h2 style="color: {eval_color}; margin:0;">AI 診斷：{eval_title}</h2>
                    <p style="margin-top:20px; font-size:18px; line-height:1.7; color: white !important;">
                        <b>分析師實話：</b><br>{eval_content}<br><br>
                        <b>組合加權 Beta：</b>{weighted_beta:.2f} (風險敏感度)<br>
                        <b>資產多樣性檢查：</b>{'偏低，標的過於雷同' if tech_ratio > 0.7 else '良好，產業分布健康'}<br>
                        <b>核心資產引擎：</b>{top_stock}
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            c_pie, c_bar = st.columns(2)
            c_pie.plotly_chart(px.pie(res_df, values='權重', names='股票', hole=0.4, title="資產權重配置", template="plotly_dark"), use_container_width=True)
            c_bar.plotly_chart(px.bar(res_df, x='股票', y='驅動力', title="波動驅動力分析 (Weight × Beta)", template="plotly_dark").update_traces(marker_color='#60a5fa'), use_container_width=True)

with tab3:
    st.header("📖 理論基礎與公式")
    st.markdown("""
    1. **多項式擬合預測**：採用二階多項式擬合 $y = ax^2 + bx + c$，捕捉短期趨勢。
    2. **馬可維茲組合風險**：加權 Beta $\\beta_{p} = \\sum w_i \\beta_i$ 用於量化系統性風險。
    3. **產業相關性偵測**：AI 會掃描標的屬性（如 Tech Concentration），評估實際的分散效果。
    """)
