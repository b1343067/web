import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 網頁基本設定
st.set_page_config(page_title="數位金融：全方位投資導航", layout="wide")
st.title("🛡️ AlphaCheck 7.0: 投資決策與風險評估系統")


tab1, tab2 = st.tabs(["🔍 個股診斷 (該不該買)", "🛡️ 投資組合風險評估"])

# --- 第一頁：個股查詢系統 (已修正 VOO 評分) ---
with tab1:
    st.header("個股買賣決策系統")
    ticker = st.text_input("輸入美股代號 (如: VOO, NVDA, PLTR)", "VOO")

    if ticker:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1y")
            
            # 建立圖表
            fig_stock = go.Figure()
            fig_stock.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='股價'))
            fig_stock.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(200).mean(), name='200MA(長線)'))
            fig_stock.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(20).mean(), name='20MA(短線)'))
            st.plotly_chart(fig_stock, use_container_width=True)

            # --- 核心評分邏輯 (針對 ETF 優化) ---
            current_price = hist['Close'].iloc[-1]
            ma200 = hist['Close'].rolling(200).mean().iloc[-1]
            ma20 = hist['Close'].rolling(20).mean().iloc[-1]
            pe = info.get('forwardPE', 0)
            is_etf = info.get('quoteType') == 'ETF'

            score = 0
            # 1. 趨勢分 (60分)
            if current_price > ma200: score += 40
            if current_price > ma20: score += 20
            
            # 2. 估值分 (20分)
            if pe > 0 and pe < 60:
                score += 20
            elif is_etf: # ETF 通常抓不到 PE，給予基礎分
                score += 15
            
            # 3. 機構評分 (20分)
            if info.get('recommendationKey') == 'buy':
                score += 20
            elif is_etf: # ETF 沒有分析師評級，給予基礎分
                score += 15

            # 總分上限限制為 100
            score = min(score, 100)

            # 顯示結果
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.metric("綜合評分", f"{score} / 100")
                if score >= 70: st.success("🚀 建議：強力買入 / 趨勢極佳")
                elif score >= 40: st.warning("⚖️ 建議：中立觀望 / 等待拉回")
                else: st.error("⚠️ 建議：風險較高 / 暫不建議")
            with res_col2:
                st.write(f"目前 P/E: {pe:.2f} | 200MA: {ma200:.2f} | 20MA: {ma20:.2f}")
                if is_etf: st.info("ℹ️ 偵測到此代號為 ETF，已自動調整評分權重。")
        
        except Exception as e:
            st.error(f"分析失敗：請確認代號是否正確。錯誤訊息: {e}")
            st.divider()

            # 建立分頁

# --- 第二頁：投資組合風險 ---
with tab2:
    st.header("組合風險健康檢查")
    df_input = pd.DataFrame([{"股票代號": "NVDA", "持有金額 (USD)": 5000}, {"股票代號": "KO", "持有金額 (USD)": 5000}])
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
                st.plotly_chart(px.pie(results, values='比例', names='股票', hole=0.4))
            with col_b:
                st.metric("組合加權 Beta 值", f"{portfolio_beta:.2f}")
                risk_level = "大" if portfolio_beta > 1.4 else ("中" if portfolio_beta >= 0.9 else "小")
                st.write(f"### 總體風險比率：**{risk_level}**")
                # --- 市場大盤指標：美債 10Y 深度分析 ---




