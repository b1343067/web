import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- 1. 網頁基本設定 ---
st.set_page_config(page_title="AlphaCheck Pro | 數位金融導航", layout="wide")

# --- 2. 優化：資料快取與技術指標函式 (放在最上面，確保程式整潔) ---
@st.cache_data(ttl=3600)  # 快取一小時，防止 API 被限流
def get_clean_data(ticker_symbol):
    try:
        s = yf.Ticker(ticker_symbol)
        h = s.history(period="1y")
        i = s.info
        if h.empty: return None, None, None, "查無數據"
        return s, h, i, None
    except Exception as e:
        return None, None, None, str(e)

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- 3. 主頁面標題 ---
st.title("🛡️ AlphaCheck Pro: 全方位投資決策系統")
st.caption(f"數據最後更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- 4. 頂部區塊：美債 10Y 監控 (這部分在你的貼文中不見了，我幫你補回來) ---
with st.container():
    st.subheader("🇺🇸 全球定價之錨：10 年期美債殖利率")
    _, tnx_h, _, tnx_err = get_clean_data("^TNX")
    if not tnx_err:
        cur_y = tnx_h['Close'].iloc[-1]
        y_high = tnx_h['Close'].max()
        c1, c2, c3 = st.columns(3)
        c1.metric("目前殖利率", f"{cur_y:.2f}%")
        c2.metric("近一年最高點", f"{y_high:.2f}%")
        c3.metric("距高點回落", f"-{y_high - cur_y:.2f}%")
        fig_tnx = px.line(tnx_h, y='Close', title="美債 10Y 走勢 (一年)", template="plotly_dark")
        fig_tnx.update_traces(line_color='#FF4B4B')
        st.plotly_chart(fig_tnx, use_container_width=True)
st.divider()

# --- 5. 功能分頁 ---
tab1, tab2 = st.tabs(["🔍 個股診斷 (RSI+均線)", "🛡️ 投資組合風險評估"])

# --- 第一頁：個股診斷 ---
with tab1:
    st.header("個股買賣決策系統")
    target = st.text_input("輸入美股代號 (如: VOO, NVDA, TSLA)", "NVDA").upper()

    if target:
        with st.spinner(f'正在深度分析 {target}...'):
            s_obj, s_h, s_i, err = get_clean_data(target)
            
            if err:
                st.error(f"分析失敗：{err} (請等幾分鐘再試)")
            else:
                # 計算指標
                curr_p = s_h['Close'].iloc[-1]
                ma200 = s_h['Close'].rolling(200).mean().iloc[-1]
                ma20 = s_h['Close'].rolling(20).mean().iloc[-1]
                s_h['RSI'] = calculate_rsi(s_h['Close'])
                rsi = s_h['RSI'].iloc[-1]
                pe = s_i.get('forwardPE', 0)
                is_etf = s_i.get('quoteType') == 'ETF'

                # 股價圖
                fig_p = go.Figure()
                fig_p.add_trace(go.Scatter(x=s_h.index, y=s_h['Close'], name='股價'))
                fig_p.add_trace(go.Scatter(x=s_h.index, y=s_h['Close'].rolling(200).mean(), name='200MA'))
                fig_p.update_layout(title=f"{target} 趨勢圖", template="plotly_dark")
                st.plotly_chart(fig_p, use_container_width=True)

                # 評分邏輯 (整合版)
                score = 0
                if curr_p > ma200: score += 40
                if curr_p > ma20: score += 10
                if 30 <= rsi <= 70: score += 10
                elif rsi < 30: score += 20  # 超跌加分
                
                if (pe > 0 and pe < 60) or is_etf: score += 20
                if s_i.get('recommendationKey') == 'buy' or is_etf: score += 20
                score = min(score, 100)

                # 顯示結果
                r1, r2 = st.columns([1, 2])
                with r1:
                    st.metric("綜合評分", f"{score} / 100")
                    if score >= 70: st.success("🚀 建議：強力買入")
                    elif score >= 40: st.warning("⚖️ 建議：中立觀望")
                    else: st.error("⚠️ 建議：風險較高")
                with r2:
                    st.write(f"**詳細指標數據：**")
                    st.write(f"RSI: {rsi:.2f} | P/E: {pe:.2f} | 200MA: {ma200:.2f}")
                    if is_etf: st.info("ℹ️ 偵測為 ETF，已優化權重。")

# --- 第二頁：組合風險 ---
with tab2:
    st.header("投資組合風險健康檢查")
    df_init = pd.DataFrame([{"股票代號": "NVDA", "持有金額 (USD)": 5000}, {"股票代號": "VOO", "持有金額 (USD)": 5000}])
    edit_df = st.data_editor(df_init, num_rows="dynamic", key="portfolio_editor")

    if st.button("開始評估組合風險"):
        total_v = edit_df["持有金額 (USD)"].sum()
        p_beta = 0
        p_results = []
        
        for idx, row in edit_df.iterrows():
            t = row["股票代號"].upper()
            amt = row["持有金額 (USD)"]
            obj, _, i, _ = get_clean_data(t)
            if i:
                b = i.get('beta', 1.0)
                w = amt / total_v
                p_beta += b * w
                p_results.append({"股票": t, "比例": w, "Beta": b})
        
        if p_results:
            ca, cb = st.columns(2)
            with ca:
                st.plotly_chart(px.pie(p_results, values='比例', names='股票', hole=0.4, title="資產配置"))
            with cb:
                st.metric("組合加權 Beta 值", f"{p_beta:.2f}")
                risk_lv = "大" if p_beta > 1.4 else ("中" if p_beta >= 0.9 else "小")
                st.markdown(f"### 總體風險比率：**{risk_lv}**")
                st.dataframe(pd.DataFrame(p_results))
