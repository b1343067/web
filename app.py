import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression # AI 預測引擎

# ==========================================
# 1. 核心邏輯與 AI 引擎 (Functions)
# ==========================================

@st.cache_data(ttl=3600)
def fetch_financial_data(ticker_name):
    """抓取 2 年數據確保均線完整，並過濾不可序列化物件"""
    try:
        # 自動清洗代號 (BRK/B -> BRK-B)
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
    """計算 RSI、10MA、50MA、200MA"""
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
    return df.tail(252) # 只回傳最近一年給圖表，但均線是完整的

def ai_trend_prediction(df, days=5):
    """利用線性回歸 (AI) 預測未來 5 天走勢"""
    df_pred = df.reset_index()
    df_pred['Date_num'] = pd.to_datetime(df_pred['Date']).apply(lambda x: x.toordinal())
    
    X = df_pred[['Date_num']].values
    y = df_pred['Close'].values
    
    model = LinearRegression().fit(X, y)
    
    last_date_num = df_pred['Date_num'].max()
    future_dates_num = np.array([last_date_num + i for i in range(1, days + 1)]).reshape(-1, 1)
    future_preds = model.predict(future_dates_num)
    
    last_date = df_pred['Date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    return future_dates, future_preds

# ==========================================
# 2. 網頁 UI 佈局設計
# ==========================================

st.set_page_config(page_title="AlphaCheck Elite 究極版", layout="wide")
st.title("🏛️ AlphaCheck Elite: 智慧金融與 AI 趨勢分析系統")
st.caption(f"數位金融科技系專案 | 最後更新: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# --- 側邊欄：全球監控 ---
st.sidebar.title("📊 市場監控中心")
with st.sidebar:
    tnx_h, _, _ = fetch_financial_data("^TNX")
    if tnx_h is not None:
        cur_y = tnx_h['Close'].iloc[-1]
        st.metric("美債 10Y 殖利率", f"{cur_y:.2f}%", delta=f"{cur_y - tnx_h['Close'].iloc[-2]:.2f}%")
        st.line_chart(tnx_h['Close'].tail(60))
    st.divider()
    st.success("✅ 系統已整合 AI 預測模型")

# --- 功能分頁 ---
tab1, tab2, tab3 = st.tabs(["🔍 AI 個股深度診斷", "🛡️ 投資組合風險量化", "📖 系統邏輯說明"])

# --- Tab 1: 個股診斷 ---
with tab1:
    col_in, _ = st.columns([2, 2])
    raw_ticker = col_in.text_input("請輸入美股代號 (支援 BRK/B 格式)", "NVDA")
    
    if raw_ticker:
        target = raw_ticker.upper().replace("/", "-").strip()
        with st.spinner(f'AI 引擎運算中...'):
            hist, info, err = fetch_financial_data(target)
            
            if err:
                st.error(f"分析失敗：{err}")
            else:
                hist = calculate_indicators(hist)
                f_dates, f_preds = ai_trend_prediction(hist)
                
                # --- 專業繪圖區 ---
                fig = go.Figure()
                # K線
                fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'],
                                            low=hist['Low'], close=hist['Close'], name='歷史K線'))
                # 均線
                fig.add_trace(go.Scatter(x=hist.index, y=hist['MA10'], line=dict(color='cyan', width=1), name='10MA'))
                fig.add_trace(go.Scatter(x=hist.index, y=hist['MA50'], line=dict(color='magenta', width=1), name='50MA'))
                fig.add_trace(go.Scatter(x=hist.index, y=hist['MA200'], line=dict(color='orange', width=2), name='200MA'))
                # AI 預測線 (橘色虛線)
                fig.add_trace(go.Scatter(x=f_dates, y=f_preds, line=dict(color='yellow', dash='dash', width=3), name='AI 預測趨勢'))
                
                fig.update_layout(title=f"{target} 多重指標與 AI 未來預估", template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

                # 數據面板
                c1, c2, c3, c4 = st.columns(4)
                cur_p = hist['Close'].iloc[-1]
                rsi_v = hist['RSI'].iloc[-1]
                c1.metric("目前價格", f"${cur_p:.2f}")
                c2.metric("RSI (14D)", f"{rsi_v:.1f}")
                c3.metric("本益比 (PE)", info.get('forwardPE', 'N/A'))
                c4.metric("AI 預測 (5D)", "看漲" if f_preds[-1] > cur_p else "看跌")

                # 綜合評分
                score = 0
                if cur_p > hist['MA200'].iloc[-1]: score += 40
                if 30 <= rsi_v <= 65: score += 30
                if f_preds[-1] > cur_p: score += 30 # AI 預測向上就加分
                
                s_color = "green" if score >= 70 else "orange" if score >= 40 else "red"
                st.markdown(f"### 🚩 系統綜合評分：<span style='color:{s_color}'>{score} 分</span>", unsafe_allow_html=True)

# --- Tab 2: 投資組合 ---
with tab2:
    st.header("🛡️ 組合風險量化分析")
    portfolio_df = pd.DataFrame([{"代號": "NVDA", "金額": 5000}, {"代號": "VOO", "金額": 5000}])
    edited = st.data_editor(portfolio_df, num_rows="dynamic", key="final_p_edit")
    
    if st.button("運行壓力測試"):
        total_amt = edited["金額"].sum()
        w_beta = 0
        p_list = []
        for _, row in edited.iterrows():
            _, i, _ = fetch_financial_data(row["代號"])
            if i:
                b = i.get('beta', 1.0)
                w = row["金額"] / total_amt
                w_beta += b * w
                p_list.append({"股票": row["代號"], "權重": w})
        
        ca, cb = st.columns(2)
        with ca:
            st.plotly_chart(px.pie(p_results:=p_list, values='權重', names='股票', hole=0.4, title="資產比例"))
        with cb:
            st.metric("組合加權 Beta 值", f"{w_beta:.2f}")
            risk_desc = "高風險" if w_beta > 1.3 else "中風險" if w_beta >= 0.9 else "低風險"
            st.write(f"### 總體風險等級：**{risk_desc}**")

# --- Tab 3: 理論說明 ---
with tab3:
    st.header("📖 系統設計理論")
    st.markdown("""
    本系統基於數位金融科技概念開發，整合以下模組：
    1. **AI 預測引擎**：採用 `scikit-learn` 線性回歸模型，對股價進行最小平方法 (OLS) 趨勢擬合。
    

[Image of linear regression graph]

    2. **三重均線系統**：結合短、中、長線均線，並透過 2 年數據回溯確保指標連續性。
    3. **風險量化**：計算資產組合之加權 Beta 值，評估相對於標普 500 的市場風險。
    """)
