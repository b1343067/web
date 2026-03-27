import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 網頁基本設定
st.set_page_config(page_title="數位金融：全方位投資導航", layout="wide")
st.title("🛡️ AlphaCheck 6.0: 投資決策與風險評估系統")

# --- 市場大盤指標：美債 10Y 深度分析 ---
st.subheader("🇺🇸 全球定價之錨：10 年期美債殖利率動態")
try:
    tnx = yf.Ticker("^TNX")
    hist_1y = tnx.history(period="1y")
    
    current_yield = hist_1y['Close'].iloc[-1]
    recent_high = hist_1y['Close'].max()
    gap = recent_high - current_yield

    # 第一列：數據卡片
    col1, col2, col3 = st.columns(3)
    col1.metric("目前殖利率", f"{current_yield:.2f}%")
    col2.metric("近一年最高點", f"{recent_high:.2f}%")
    col3.metric("距高點回落", f"-{gap:.2f}%", delta_color="normal")

    # 第二列：殖利率走勢圖 (這會讓網頁看起來很豐富！)
    fig_tnx = px.line(hist_1y, y='Close', title="美債 10Y 殖利率一年走勢圖", labels={'Close': 'Yield (%)', 'Date': '日期'})
    fig_tnx.update_traces(line_color='#FF4B4B') # 使用顯眼的紅色
    st.plotly_chart(fig_tnx, use_container_width=True)
except:
    st.write("暫時無法獲取美債即時數據")

st.divider()

# 建立分頁
tab1, tab2 = st.tabs(["🔍 個股診斷 (該不該買)", "🛡️ 投資組合風險評估"])

# --- 第一頁：個股查詢系統 ---
with tab1:
    st.header("個股買賣決策系統")
    ticker = st.text_input("輸入美股代號 (如: NVDA, PLTR, TSLA)", "PLTR")

    if ticker:
        stock = yf.Ticker(ticker)
        try:
            info = stock.info
            hist = stock.history(period="1y")
            
            # 建立圖表：股價與均線對比
            fig_stock = go.Figure()
            fig_stock.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='股價'))
            fig_stock.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(200).mean(), name='200日均線(長線)'))
            fig_stock.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(20).mean(), name='20日均線(短線)'))
            fig_stock.update_layout(title=f"{ticker} 價格與趨勢分析", xaxis_title="日期", yaxis_title="股價 (USD)")
            st.plotly_chart(fig_stock, use_container_width=True)

           # --- 優化後的決策邏輯 ---
score = 0
total_possible_score = 100 # 總分上限

# 1. 均線邏輯 (這對 ETF 依然有效)
if current_price > ma200: score += 40
if current_price > ma20: score += 20

# 2. 處理 P/E (如果抓不到數據就不扣分，或改用其他權重)
if pe_ratio > 0:
    if pe_ratio < 60: score += 20
else:
    # 如果是 ETF 抓不到 P/E，我們把這 20 分權重分配給均線或直接給基礎分
    score += 10 

# 3. 處理推薦評級
if info.get('recommendationKey') == 'buy':
    score += 20
elif info.get('quoteType') == 'ETF':
    # 如果是 ETF，通常沒有推薦評級，我們給予 10 分的基礎信任分
    score += 10

# --- 第二頁：投資組合風險 ---
with tab2:
    st.header("組合風險健康檢查")
    df_input = pd.DataFrame([
        {"股票代號": "NVDA", "持有金額 (USD)": 5000},
        {"股票代號": "KO", "持有金額 (USD)": 5000},
    ])
    edited_df = st.data_editor(df_input, num_rows="dynamic", key="portfolio_editor")

    if st.button("開始評估組合風險"):
        total_val = edited_df["持有金額 (USD)"].sum()
        results = []
        portfolio_beta = 0
        
        for index, row in edited_df.iterrows():
            t = row["股票代號"]
            amt = row["持有金額 (USD)"]
            try:
                b = yf.Ticker(t).info.get('beta', 1.0)
                weight = amt / total_val
                portfolio_beta += b * weight
                results.append({"股票": t, "比例": weight, "Beta": b})
            except:
                st.write(f"⚠️ 無法讀取 {t}")

        if results:
            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(px.pie(results, values='比例', names='股票', hole=0.4, title="資產比例分佈"))
            with col_b:
                st.metric("組合加權 Beta 值", f"{portfolio_beta:.2f}")
                # 風險描述
                risk_level = "大" if portfolio_beta > 1.4 else ("中" if portfolio_beta >= 0.9 else "小")
                st.write(f"### 總體風險比率：**{risk_level}**")
                st.dataframe(pd.DataFrame(results)) # 顯示詳細列表增加豐富感
