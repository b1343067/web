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
    """計算 RSI 與三重均線 (10, 50, 200)"""
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
    """AI 曲線擬合 + 信心陰影區間"""
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
# 2. UI 視覺樣式優化
# ==========================================

st.set_page_config(page_title="AlphaCheck Elite", layout="wide")

st.markdown("""
    <style>
    .stApp, [data-testid="stSidebar"] { background-color: #0f172a !important; }
    h1, h2, h3, p, span, label, .stMarkdown { color: #f1f5f9 !important; }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 16px !important; }
    [data-testid="stMetricValue"] { color: #60a5fa !important; font-weight: bold !important; }
    .stButton>button {
        background-color: transparent !important; color: #60a5fa !important;
        border: 2px solid #60a5fa !important; border-radius: 25px !important;
        width: 100% !important; font-weight: 700 !important;
    }
    .stButton>button:hover { background-color: rgba(96, 165, 250, 0.1) !important; box-shadow: 0 0 20px rgba(96, 165, 250, 0.3) !important; }
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
        fig_side = px.line(tnx_h.tail(30), y='Close', template="plotly_dark").update_traces(line_color='#60a5fa')
        fig_side.update_layout(height=120, margin=dict(l=0,r=0,t=0,b=0), xaxis_visible=False, yaxis_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_side, use_container_width=True)

tab1, tab2, tab3 = st.tabs(["🔍 AI 市場深度診斷", "🛡️ 投資組合分析報告", "📖 模型理論說明"])

# --- Tab 1: AI 診斷 (RSI + 三重均線 + 曲線預測) ---
with tab1:
    col_in, _ = st.columns([2, 2])
    raw_ticker = col_in.text_input("輸入美股代號", "NVDA")
    if raw_ticker:
        target = raw_ticker.upper().replace("/", "-").strip()
        hist, info, err = fetch_financial_data(target)
        if not err:
            hist = calculate_indicators(hist)
            f_dates, f_preds, f_intervals = get_ai_prediction_model(hist)
            cur_p = hist['Close'].iloc[-1]
            ret = ((f_preds[-1] - cur_p) / cur_p) * 100
            
            bg = "rgba(20, 83, 45, 0.4)" if ret > 2 else "rgba(127, 29, 29, 0.4)" if ret < -2 else "rgba(30, 41, 59, 0.6)"
            st.markdown(f"<div class='report-card' style='background-color:{bg};'><h3>🤖 AI 評級</h3><p>預期變動：{ret:+.2f}%</p></div>", unsafe_allow_html=True)
            
            fig = go.Figure(data=[go.Candlestick(x=hist.tail(120).index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='走勢')])
            fig.add_trace(go.Scatter(x=hist.tail(120).index, y=hist['MA10'].tail(120), line=dict(color='#81d4fa', width=1), name='10MA'))
            fig.add_trace(go.Scatter(x=hist.tail(120).index, y=hist['MA50'].tail(120), line=dict(color='#fbbf24', width=1), name='50MA'))
            fig.add_trace(go.Scatter(x=hist.tail(120).index, y=hist['MA200'].tail(120), line=dict(color='#94a3b8', width=2), name='200MA'))
            fig.add_trace(go.Scatter(x=f_dates+f_dates[::-1], y=[f_preds[i]+f_intervals[i] for i in range(len(f_preds))]+[f_preds[i]-f_intervals[i] for i in range(len(f_preds))][::-1], fill='toself', fillcolor='rgba(96, 165, 250, 0.1)', line_color='rgba(0,0,0,0)', name='AI 波動預期'))
            fig.add_trace(go.Scatter(x=f_dates, y=f_preds, line=dict(color='white', dash='dash', width=2), name='AI 路徑'))
            fig.update_layout(template="plotly_dark", height=550, xaxis_rangeslider_visible=False, legend=dict(font=dict(color="white")))
            st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: 真正動態的分析報告 ---
with tab2:
    st.markdown("### 🛡️ 投資組合智能量化與 AI 評價")
    p_df = pd.DataFrame([{"代號": "QQQM", "金額": 5000}, {"代號": "QQQ", "金額": 5000}])
    edited = st.data_editor(p_df, num_rows="dynamic", use_container_width=True, key="p_table")
    
    if st.button("🚀 執行 AI 智能分析"):
        with st.spinner('正在分析資產結構...'):
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
                    results.append({"股票": row["代號"].upper(), "權重": weight, "Beta": beta, "驅動力": beta * weight})
            
            res_df = pd.DataFrame(results)
            st.divider()
            
            # --- 動態邏輯：判斷是分散還是集中 ---
            top_stock = res_df.sort_values('權重', ascending=False)['股票'].iloc[0]
            
            if weighted_beta > 1.25:
                eval_title = "進攻型：動能高度集中"
                eval_color = "#f87171"
                eval_content = f"組合呈現明顯的**單一風格偏好**。標的如 **{top_stock}** 具有高 Beta 特性，雖然在牛市獲利極快，但缺乏避險標的，回檔風險高。"
            elif 0.9 <= weighted_beta <= 1.25:
                # 這裡增加一個判斷：如果股票太少或 Beta 太接近，就不給「精英級」稱號
                if len(res_df) < 3:
                    eval_title = "平衡型：結構尚待完善"
                    eval_color = "#fbbf24"
                    eval_content = f"目前組合以 **{top_stock}** 為核心，雖 Beta 處於平衡區間，但資產標的過於雷同，建議引入低相關性的避險資產。"
                else:
                    eval_title = "精英級：核心—衛星策略"
                    eval_color = "#60a5fa"
                    eval_content = f"配置展現了專業深度。以 **{top_stock}** 為核心定海神針，並成功透過不同屬性的標的分散系統性風險。"
            else:
                eval_title = "防禦型：價值堡壘配置"
                eval_color = "#4ade80"
                eval_content = f"資產抗風險能力極強。以 **{top_stock}** 為主軸，能在市場動盪中提供極佳保護力。"

            st.markdown(f"""
                <div class="report-card" style="border-left: 10px solid {eval_color}; background-color: #1e293b;">
                    <h2 style="color: {eval_color}; margin:0;">AI 診斷：{eval_title}</h2>
                    <p style="margin-top:20px; font-size:18px; color: white !important;">
                        <b>首席分析師診斷：</b><br>{eval_content}<br><br>
                        <b>組合加權 Beta：</b>{weighted_beta:.2f}<br>
                        <b>組合平均 RSI：</b>{weighted_rsi:.1f} ({'短期情緒過熱' if weighted_rsi > 70 else '情緒健康'})<br>
                        <b>波動核心：</b>{top_stock}
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            c_pie, c_bar = st.columns(2)
            c_pie.plotly_chart(px.pie(res_df, values='權重', names='股票', hole=0.4, title="資產權重配置", template="plotly_dark"), use_container_width=True)
            c_bar.plotly_chart(px.bar(res_df, x='股票', y='驅動力', title="波動驅動力 (Weight × Beta)", template="plotly_dark").update_traces(marker_color='#60a5fa'), use_container_width=True)

with tab3:
    st.header("📖 模型理論")
    st.markdown("本系統採用馬可維茲組合風險模型與動態多項式趨勢擬合...")
