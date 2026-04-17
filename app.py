import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ==========================================
# 1. 核心數據模組 (含自動修正與 2 年長回溯期)
# ==========================================

@st.cache_data(ttl=3600)
def fetch_financial_data(ticker_name):
    """
    抓取 2 年數據以確保 200MA 完整，並只回傳可快取的資料類型
    """
    try:
        # 自動修正代號格式 (如 BRK/B -> BRK-B)
        clean_ticker = ticker_name.upper().replace("/", "-").strip()
        ticker_obj = yf.Ticker(clean_ticker)
        
        # 抓取 2 年數據 (確保 200MA 不會斷掉)
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

def calculate_advanced_indicators(df):
    """計算 RSI 與三重均線 (10, 50, 200)"""
    # RSI 計算
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 三重均線計算
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # 只取最近一年的資料來畫圖，但前面的數據已用來算好 200MA 了
    return df.tail(252)

# ==========================================
# 2. 網頁 UI 佈局
# ==========================================

st.set_page_config(page_title="AlphaCheck Elite | 數位金融終端", layout="wide")

# 主標題
st.title("🏛️ AlphaCheck Elite: 智慧型金融決策終端")
st.caption(f"系統狀態：正常 | 數據更新時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- 側邊欄：市場監控中心 (修正縮排錯誤的地方) ---
st.sidebar.title("📊 市場監控中心")
with st.sidebar:
    st.subheader("美債 10Y 殖利率 (^TNX)")
    tnx_h, _, _ = fetch_financial_data("^TNX")
    if tnx_h is not None:
        cur_y = tnx_h['Close'].iloc[-1]
        st.metric("目前殖利率", f"{cur_y:.2f}%", delta=f"{cur_y - tnx_h['Close'].iloc[-2]:.2f}%")
        st.line_chart(tnx_h['Close'].tail(60))
    st.divider()
    st.info("💡 系統已啟用 Data Sanitization (代號自動清洗) 與 Cache Protection。")

# --- 定義分頁 ---
tab1, tab2, tab3 = st.tabs(["🔍 深度個股掃描", "🛡️ 投資組合風險", "📖 系統分析邏輯"])

# --- Tab 1: 個股診斷 ---
with tab1:
    col_in, _ = st.columns([2, 2])
    raw_input = col_in.text_input("輸入美股代號 (支援斜線自動修正，如: BRK/B, VOO, NVDA)", "NVDA")
    
    if raw_input:
        target = raw_input.upper().replace("/", "-").strip()
        with st.spinner(f'正在深度掃描 {target}...'):
            hist, info, err = fetch_financial_data(target)
            
            if err:
                if "找不到" in err:
                    st.error(f"❌ {err}，請確認代號格式是否正確。")
                else:
                    st.error(f"🚨 數據源 API 暫時受限，請幾分鐘後再試。")
            else:
                # 計算指標並裁切至最近一年
                hist = calculate_advanced_indicators(hist)
                
                # A. 專業多重均線 K 線圖
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'],
                                            low=hist['Low'], close=hist['Close'], name='K線'))
                
                fig.add_trace(go.Scatter(x=hist.index, y=hist['MA10'], line=dict(color='cyan', width=1), name='10MA'))
                fig.add_trace(go.Scatter(x=hist.index, y=hist['MA50'], line=dict(color='magenta', width=1), name='50MA'))
                fig.add_trace(go.Scatter(x=hist.index, y=hist['MA200'], line=dict(color='orange', width=2), name='200MA'))
                
                fig.update_layout(title=f"{target} 三重均線分析 (已補足 200MA 背景數據)", 
                                  template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

                # B. 數據指標卡
                c1, c2, c3, c4 = st.columns(4)
                rsi_v = hist['RSI'].iloc[-1]
                cur_p = hist['Close'].iloc[-1]
                ma200_v = hist['MA200'].iloc[-1]
                
                c1.metric("目前股價", f"${cur_p:.2f}")
                c2.metric("RSI (14D)", f"{rsi_v:.1f}")
                c3.metric("本益比 (PE)", info.get('forwardPE', 'N/A'))
                c4.metric("Beta (風險係數)", info.get('beta', 'N/A'))

                # C. 整合評分
                st.subheader("🎯 系統綜合評估")
                score = 0
                reasons = []

                if cur_p > ma200_v:
                    score += 40
                    reasons.append("✅ 股價位於 200 日長線均線上方。")
                
                if 30 <= rsi_v <= 65:
                    score += 30
                    reasons.append("✅ RSI 指標處於健康區間。")
                elif rsi_v < 30:
                    score += 35
                    reasons.append("💎 RSI 進入超賣區，具備潛在價值。")

                if (info.get('forwardPE', 100) < 55) or (info.get('quoteType') == 'ETF'):
                    score += 30
                    reasons.append("✅ 估值合理或屬 ETF 工具。")

                score = min(score, 100)
                s_color = "green" if score >= 70 else "orange" if score >= 40 else "red"
                st.markdown(f"### 評分結果：<span style='color:{s_color}'>{score} 分</span>", unsafe_allow_html=True)
                for r in reasons: st.write(r)

# --- Tab 2: 組合風險 ---
with tab2:
    st.header("🛡️ 投資組合風險量化分析")
    p_df = pd.DataFrame([{"代號": "NVDA", "金額": 5000}, {"代號": "VOO", "金額": 5000}])
    edited = st.data_editor(p_df, num_rows="dynamic", key="p_edit_v10")
    
    if st.button("運行組合測試"):
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
            st.plotly_chart(px.pie(p_list, values='權重', names='股票', hole=0.4, title="資產比例"))
        with cb:
            st.metric("組合加權 Beta 值", f"{w_beta:.2f}")
            risk_lv = "大" if w_beta > 1.3 else "中" if w_beta >= 0.9 else "小"
            st.write(f"### 總體風險等級：**{risk_lv}**")

# --- Tab 3: 理論說明 ---
with tab3:
    st.header("📖 系統分析邏輯說明")
    st.markdown("""
    1. **三重均線系統**: 利用 10MA (短)、50MA (中) 與 200MA (長) 進行趨勢過濾。
    2. **補全數據**: 抓取 2 年數據以確保 200 日移動平均線在圖表首日即可呈現。
    3. **RSI 情緒指標**: 評估股價是否過度偏離。
    4. **組合 Beta**: 計算組合相對於標普 500 的波動度。
    """)
