import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 網頁基本設定
st.set_page_config(page_title="數位金融：全方位投資導航", layout="wide")
st.title("📈 AlphaCheck 3.0: 投資決策與風險評估系統")

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
            
            # 計算指標
            current_price = hist['Close'].iloc[-1]
            ma200 = hist['Close'].rolling(window=200).mean().iloc[-1]
            pe_ratio = info.get('forwardPE', 0)
            
            # 決策邏輯
            score = 0
            if current_price > ma200: score += 40
            if 0 < pe_ratio < 50: score += 30
            if info.get('recommendationKey') == 'buy': score += 30

            # 顯示儀表板
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = score,
                title = {'text': f"{ticker} 綜合決策分數"},
                gauge = {'axis': {'range': [0, 100]},
                         'bar': {'color': "royalblue"},
                         'steps': [
                             {'range': [0, 50], 'color': "lightcoral"},
                             {'range': [50, 80], 'color': "wheat"},
                             {'range': [80, 100], 'color': "lightgreen"}]}))
            st.plotly_chart(fig)
            
            if score >= 70: st.success("✅ 決策建議：【值得關注 / 買入】")
            elif score >= 40: st.warning("⚠️ 決策建議：【中立 / 觀望】")
            else: st.error("❌ 決策建議：【風險較高 / 避開】")
        except:
            st.error("無法抓取數據，請確認代號是否正確")

# --- 第二頁：投資組合風險 ---
with tab2:
    st.header("組合風險健康檢查")
    st.write("請輸入您的持倉比例，系統將自動計算加權風險 (Beta)。")
    
    df_input = pd.DataFrame([
        {"股票代號": "NVDA", "持有金額 (USD)": 5000},
        {"股票代號": "QQQM", "持有金額 (USD)": 3000},
    ])
    edited_df = st.data_editor(df_input, num_rows="dynamic")

    if st.button("開始評估組合風險"):
        total_val = edited_df["持有金額 (USD)"].sum()
        portfolio_beta = 0
        results = []
        
        for index, row in edited_df.iterrows():
            t = row["股票代號"]
            amt = row["持有金額 (USD)"]
            try:
                b = yf.Ticker(t).info.get('beta', 1.0)
                weight = amt / total_val
                portfolio_beta += b * weight
                results.append({"股票": t, "比例": weight, "Beta": b})
            except:
                st.write(f"⚠️ 無法讀取 {t} 的數據")

        if results:
            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(px.pie(results, values='比例', names='股票', hole=0.4))
            with col_b:
                st.metric("組合加權 Beta 值", f"{portfolio_beta:.2f}")
                if portfolio_beta < 0.9: st.info("評估結果：【風險：小】")
                elif portfolio_beta <= 1.4: st.warning("評估結果：【風險：中】")
                else: st.error("評估結果：【風險：大】")
