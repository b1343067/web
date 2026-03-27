import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="FinTech 投資組合風險評估", layout="wide")
st.title("🛡️ 投資組合風險健康檢查")

# 1. 建立輸入區 (像 Excel 一樣好用)
st.subheader("第一步：輸入你的投資組合")
df_input = pd.DataFrame(
    [
        {"股票代號": "NVDA", "持有金額 (USD)": 5000},
        {"股票代號": "QQQM", "持有金額 (USD)": 3000},
        {"股票代號": "PLTR", "持有金額 (USD)": 2000},
    ]
)
edited_df = st.data_editor(df_input, num_rows="dynamic")

# 2. 計算邏輯
if st.button("開始計算風險評估"):
    total_value = edited_df["持有金額 (USD)"].sum()
    portfolio_beta = 0
    valid_data = True
    
    with st.spinner('正在從 Yahoo Finance 抓取即時數據...'):
        results = []
        for index, row in edited_df.iterrows():
            ticker = row["股票代號"]
            amount = row["持有金額 (USD)"]
            weight = amount / total_value
            
            try:
                stock_info = yf.Ticker(ticker).info
                beta = stock_info.get('beta', 1.0) # 如果抓不到 Beta 預設給 1.0
                portfolio_beta += beta * weight
                results.append({"股票": ticker, "比例": weight, "Beta": beta})
            except:
                st.error(f"找不到代號: {ticker}")
                valid_data = False

    if valid_data:
        # --- 顯示風險結論 ---
        st.divider()
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 投資組合分布")
            fig = px.pie(results, values='比例', names='股票', hole=0.4)
            st.plotly_chart(fig)

        with col2:
            st.subheader("⚖️ 綜合風險等級")
            # 判斷風險大小
            if portfolio_beta < 0.9:
                risk_txt, color = "低風險 (風險：小)", "blue"
            elif portfolio_beta <= 1.4:
                risk_txt, color = "中風險 (風險：中)", "orange"
            else:
                risk_txt, color = "高風險 (風險：大)", "red"
            
            st.metric("組合加權 Beta 值", f"{portfolio_beta:.2f}")
            st.markdown(f"### 評估結果：:{color}[{risk_txt}]")
            st.info(f"這代表當大盤上漲 1% 時，你的組合預期會變動 {portfolio_beta:.2f}%")

        # 3. 給予專業建議
        st.subheader("💡 理財建議")
        if color == "red":
            st.write("您的組合攻擊力強，但遇到崩盤時跌幅會很大。建議增加一些低 Beta 的標的 (如債券 ETF 或電信股) 來分散風險。")
        elif color == "blue":
            st.write("您的組合非常穩健，適合長期退休金規劃，但在大牛市時獲利可能跟不上科技股。")
        else:
            st.write("您的組合與市場步調一致，屬於穩健中求成長的配置。")
