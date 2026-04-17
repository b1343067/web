import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. 核心數據處理模組 (AI 預測與技術指標)
# ==========================================

@st.cache_data(ttl=3600)
def fetch_financial_data(ticker_name):
    """抓取 2 年數據以確保均線完整，並過濾不可序列化物件"""
    try:
        # 代號自動清洗：處理 BRK/B 或小寫輸入
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
    """計算 RSI、10MA、50MA、200MA"""
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
    """AI 線性回歸預測與動態波動區間"""
    df_pred = df.tail(120).reset_index() # 使用最近半年數據擬合
    df_pred['Date_num'] = pd.to_datetime(df_pred['Date']).apply(lambda x: x.toordinal())
    
    X = df_pred[['Date_num']].values
    y = df_pred['Close'].values
    model = LinearRegression().fit(X, y)
    
    last_date_num = df_pred['Date_num'].max()
    future_dates_num = np.array([last_date_num + i for i in range(1, days + 1)]).reshape(-1, 1)
    future_preds = model.predict(future_dates_num)
    
    # 信心區間：利用最近 30 天波動度
    std_dev = df['Close'].tail(30).std() * 0.8
    
    last_date = df_pred['Date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    return future_dates, future_preds, std_dev

# ==========================================
# 2. UI 專業介面設計 (毛玻璃與莫蘭迪配色)
# ==========================================

st.set_page_config(page_title="AlphaCheck Elite Pro", layout="wide")

# 核心美化 CSS：引入深海藍主題與柔和邊框
st.markdown("""
    <style>
    .stApp { background-color: #0f172a; }
    .decision-card {
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid;
        backdrop-filter: blur(10px);
    }
    .stMetric { background-color: #1e293b !important; border: 1px solid #334155 !important; border-radius: 10px; }
    h1, h2, h3, p, span, .stMarkdown { color: #f1f5f9 !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ AlphaCheck Elite: 專業投資決策終端")
st.caption(f"數位金融科技系專案 | 版本 15.0 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# --- 側邊欄：全球市場背景 ---
st.sidebar.title("📊 市場監控")
with st.sidebar:
    st.markdown("### 🌍 全球宏觀指標")
    tnx_h, _, _ = fetch_financial_data("^TNX")
    if tnx_h is not None:
        cur_y = tnx_h['Close'].iloc[-1]
        st.metric("美債 10Y 殖利率", f"{cur_y:.2f}%", delta=f"{cur_y - tnx_h['Close'].iloc[-2]:.2f}%")
        # 繪製美債柔和小圖
        fig_side = px.line(tnx_h.tail(30), y='Close', template="plotly_dark")
        fig_side.update_traces(line_color='#60a5fa', line_width=2)
        fig_side.update_layout(height=120, margin=dict(l=0,r=0,t=0,b=0), xaxis_visible=False, yaxis_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_side, use_container_width=True, config={'displayModeBar': False})
    st.divider()
    st.info("💡 系統已啟用毛玻璃介面與 AI 信心區間預測模式。")

# --- 定義分頁 ---
tab1, tab2, tab3 = st.tabs(["🎯 AI 深度診斷", "🛡️ 投資組合風險", "📖 理論與模型說明"])

# --- Tab 1: AI 診斷主介面 ---
with tab1:
    col_in, _ = st.columns([2, 2])
    raw_ticker = col_in.text_input("輸入美股代號 (支援斜線格式，如 BRK/B)", "NVDA")
    
    if raw_ticker:
        target = raw_ticker.upper().replace("/", "-").strip()
        with st.spinner(f'AI 引擎正在掃描 {target} 市場數據...'):
            hist, info, err = fetch_financial_data(target)
            
            if err:
                st.error(f"分析失敗：{err}")
            else:
                # 數據運算
                hist = calculate_indicators(hist)
                f_dates, f_preds, std = get_ai_prediction_model(hist)
                
                # --- A. 視覺化決策指令盒 (柔和莫蘭迪配色) ---
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
                        <h3 style="margin:0; font-weight: 500;">🤖 AI 智能評級：{b_text}</h3>
                        <p style="margin-top:8px; font-size:18px;">
                            預估 7 日目標：<b>${target_p:.2f}</b> | 期望收益：<b>{expected_ret:+.2f}%</b>
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                # --- B. 專業 Plotly 圖表 ---
                plot_data = hist.tail(120)
                fig = go.Figure()
                # 歷史K線
                fig.add_trace(go.Candlestick(x=plot_data.index, open=plot_data['Open'], high=plot_data['High'], low=plot_data['Low'], close=plot_data['Close'], name='歷史走勢', increasing_line_color='#4ade80', decreasing_line_color='#f87171'))
                # 技術指標
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA200'], line=dict(color='#64748b', width=2), name='200MA (生命線)'))
                # AI 預測陰影區
                fig.add_trace(go.Scatter(x=f_dates + f_dates[::-1], y=list(f_preds + std) + list(f_preds - std)[::-1], fill='toself', fillcolor='rgba(96, 165, 250, 0.1)', line_color='rgba(0,0,0,0)', name='AI 波動預期'))
                # AI 預測路徑
                fig.add_trace(go.Scatter(x=f_dates, y=f_preds, line=dict(color='#60a5fa', dash='dot', width=2), name='AI 預測路徑'))
                
                fig.update_layout(template="plotly_dark", height=550, xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)

                # --- C. 指標卡片 ---
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("即時股價", f"${cur_p:.2f}")
                c2.metric("RSI 指標", f"{hist['RSI'].iloc[-1]:.1f}")
                c3.metric("預估 P/E", f"{info.get('forwardPE', 'N/A')}")
                c4.metric("市場 Beta", f"{info.get('beta', 'N/A')}")

# --- Tab 2: 投資組合風險 ---
with tab2:
    st.header("🛡️ 組合風險量化壓力測試")
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
                w = row["金額"] / total
                w_beta += b * w
                p_list.append({"股票": row["代號"], "權重": w})
        
        ca, cb = st.columns(2)
        with ca:
            st.plotly_chart(px.pie(p_list, values='權重', names='股票', hole=0.4, title="資產配比"))
        with cb:
            st.metric("組合加權 Beta 值", f"{w_beta:.2f}")
            risk_lv = "高 (積極型)" if w_beta > 1.3 else "中 (穩健型)" if w_beta >= 0.9 else "低 (防禦型)"
            st.markdown(f"### 總體風險等級：**{risk_lv}**")

# --- Tab 3: 理論說明 ---
with tab3:
    st.header("📖 系統設計與理論基礎")
    st.markdown("""
    本系統基於數位金融科技概念開發，整合以下技術模組：
    
    1. **AI 預測引擎**：
       採用 `scikit-learn` 線性回歸模型，對歷史收盤價進行最小平方法 (OLS) 之趨勢擬合。
       
    2. **技術指標 (Technical Analysis)**：
       * **RSI (相對強弱指標)**：$RSI = 100 - \\frac{100}{1 + RS}$，用於衡量市場過熱或超賣。
       * **移動平均線 (MA)**：採用 200 日均線作為長線支撐壓力參考。
       
    3. **Lookback Period 數據補全**：
       系統自動抓取 730 天 (2 年) 數據，確保 200MA 在圖表首日即可完整呈現，消除數據斷裂問題。
       
    4. **毛玻璃 UI 視覺語言**：
       採用低飽和度配色與背景模糊 (Backdrop Blur) 技術，降低視覺疲勞並提升系統專業質感。
    """)
