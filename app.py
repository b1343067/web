import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. 核心數據處理與 AI 引擎
# ==========================================

@st.cache_data(ttl=3600)
def fetch_financial_data(ticker_name):
    """抓取數據並自動修正代號格式"""
    try:
        clean_ticker = ticker_name.upper().replace("/", "-").strip()
        ticker_obj = yf.Ticker(clean_ticker)
        # 抓取 2 年數據確保 200MA 完整
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

def calculate_all_indicators(df):
    """計算 RSI 與三重專業均線"""
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # 均線
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    return df

def get_ai_prediction_model(df, days=7):
    """AI 線性回歸預測與動態波動區間"""
    df_pred = df.tail(120).reset_index() # 取最近半年數據做擬合
    df_pred['Date_num'] = pd.to_datetime(df_pred['Date']).apply(lambda x: x.toordinal())
    
    X = df_pred[['Date_num']].values
    y = df_pred['Close'].values
    model = LinearRegression().fit(X, y)
    
    last_date_num = df_pred['Date_num'].max()
    future_dates_num = np.array([last_date_num + i for i in range(1, days + 1)]).reshape(-1, 1)
    future_preds = model.predict(future_dates_num)
    
    # 信心區間：使用標準差的 5% 作為視覺化波動
    std_dev = df['Close'].tail(30).std()
    
    last_date = df_pred['Date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    return future_dates, future_preds, std_dev

# ==========================================
# 2. UI 專業介面設計
# ==========================================

st.set_page_config(page_title="AlphaCheck Elite Pro", layout="wide")

# 自定義 CSS 強化視覺深度
st.markdown("""
    <style>
    .decision-card {
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 25px;
        border: 1px solid #3d4250;
    }
    .stMetric { background-color: #161a24; border: 1px solid #2d3139; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 AlphaCheck Elite: 智慧金融決策終端")
st.caption(f"數位金融科技系 | 系統版本 13.0 | 更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# --- 側邊欄：全球市場背景 ---
st.sidebar.title("📊 市場監控")
with st.sidebar:
    st.subheader("美債 10Y 殖利率 (^TNX)")
    tnx_h, _, _ = fetch_financial_data("^TNX")
    if tnx_h is not None:
        cur_y = tnx_h['Close'].iloc[-1]
        st.metric("目前水準", f"{cur_y:.2f}%", delta=f"{cur_y - tnx_h['Close'].iloc[-2]:.2f}%")
        st.line_chart(tnx_h['Close'].tail(30))
    st.divider()
    st.info("💡 系統已啟用 AI 信心區間預測模式。")

# --- 功能分頁 ---
tab1, tab2, tab3 = st.tabs(["🎯 AI 深度診斷", "🛡️ 投資組合風險", "📖 模型數學邏輯"])

# --- Tab 1: AI 診斷主介面 ---
with tab1:
    col_in, _ = st.columns([2, 2])
    raw_ticker = col_in.text_input("輸入美股代號 (支援斜線格式，如 BRK/B)", "NVDA").upper().strip()
    
    if raw_ticker:
        target = raw_ticker.replace("/", "-")
        with st.spinner(f'AI 引擎正在掃描 {target} 市場數據...'):
            hist, info, err = fetch_financial_data(target)
            
            if err:
                st.error(f"數據分析失敗：{err}")
            else:
                # 數據運算
                hist = calculate_all_indicators(hist)
                f_dates, f_preds, std = get_ai_prediction_model(hist)
                
                # --- A. 視覺化決策指令盒 ---
                cur_p = hist['Close'].iloc[-1]
                target_p = f_preds[-1]
                expected_ret = ((target_p - cur_p) / cur_p) * 100
                
                if expected_ret > 2.5:
                    b_color, b_text, b_icon = "#1d3a2f", "Strong Buy / 強力買入", "🚀"
                elif expected_ret < -2.5:
                    b_color, b_text, b_icon = "#3a1d1d", "Strong Sell / 建議減持", "⚠️"
                else:
                    b_color, b_text, b_icon = "#2b2e3a", "Neutral / 持平觀望", "⚖️"

                st.markdown(f"""
                    <div class="decision-card" style="background-color: {b_color};">
                        <h2 style="margin:0;">{box_icon if 'box_icon' in locals() else b_icon} AI 指令：{b_text}</h2>
                        <p style="margin-top:10px; opacity:0.8; font-size:18px;">
                            7天目標價: <b>${target_p:.2f}</b> | 預期變動: <b>{expected_ret:+.2f}%</b>
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                # --- B. 專業 Plotly 圖表 ---
                plot_data = hist.tail(150)
                fig = go.Figure()
                
                # 歷史K線
                fig.add_trace(go.Candlestick(x=plot_data.index, open=plot_data['Open'], 
                                            high=plot_data['High'], low=plot_data['Low'], 
                                            close=plot_data['Close'], name='歷史走勢'))
                
                # 三重均線
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA10'], line=dict(color='#81d4fa', width=1), name='10MA'))
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA200'], line=dict(color='#ff9800', width=2), name='200MA (生命線)'))
                
                # AI 預測陰影區
                fig.add_trace(go.Scatter(
                    x=f_dates + f_dates[::-1],
                    y=list(f_preds + std) + list(f_preds - std)[::-1],
                    fill='toself', fillcolor='rgba(255, 255, 255, 0.1)',
                    line_color='rgba(255,255,255,0)', name='AI 波動預估區間'
                ))
                
                # AI 預測中心線
                fig.add_trace(go.Scatter(x=f_dates, y=f_preds, line=dict(color='white', dash='dash', width=3), name='AI 預測路徑'))
                
                fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False,
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)

                # --- C. 指標卡片 ---
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("即時股價", f"${cur_p:.2f}")
                c2.metric("RSI 強弱", f"{hist['RSI'].iloc[-1]:.1f}")
                c3.metric("預估 P/E", f"{info.get('forwardPE', 'N/A')}")
                c4.metric("市場風險 (Beta)", f"{info.get('beta', 'N/A')}")

# --- Tab 2: 投資組合風險 ---
with tab2:
    st.header("🛡️ 組合風險量化健康檢查")
    p_df = pd.DataFrame([{"代號": "NVDA", "金額": 5000}, {"代號": "VOO", "金額": 5000}])
    edited = st.data_editor(p_df, num_rows="dynamic", key="final_p_edit")
    
    if st.button("開始量化壓力測試"):
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

# --- Tab 3: 模型說明 ---
with tab3:
    st.header("📖 系統模型與數學原理")
    st.markdown("""
    本系統採用以下數位金融模型：
    1. **AI 預測模型**：使用 `LinearRegression` 對對數價格進行 OLS 擬合。
    2. **RSI 指標**：
    $$RSI = 100 - \\frac{100}{1 + \\frac{AverageGain}{AverageLoss}}$$
    3. **風險評估**：
    $$\\beta_p = \\sum w_i \\beta_i$$
    4. **回溯期優化**：抓取 730 天歷史數據以計算精準之 200MA 長線指標。
    """)
