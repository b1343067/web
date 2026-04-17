# --- Tab 1: 個股診斷 (加入代號自動修正) ---
with tab1:
    # 自動將 / 轉換成 -，並移除空格
    raw_input = st.text_input("輸入代號 (如: VOO, NVDA, BRK-B)", "BRK-B")
    target = raw_input.upper().replace("/", "-").strip()
    
    if target:
        with st.spinner(f'正在分析 {target}...'):
            hist, info, err = fetch_financial_data(target)
            
            if err:
                # 這裡修正了：如果是代號打錯，就顯示代號錯誤；否則顯示 API 受限
                if "無此標的" in err:
                    st.error(f"❌ 找不到代號 '{target}'，請確認格式（例如：BRK-B）。")
                else:
                    st.error(f"🚨 數據源暫時連線失敗，請稍候再試。")
            else:
                hist = calculate_indicators(hist)
                
                # --- 以下維持原有的圖表與評分邏輯 ---
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'],
                                            low=hist['Low'], close=hist['Close'], name='K線'))
                fig.add_trace(go.Scatter(x=hist.index, y=hist['MA200'], line=dict(color='orange'), name='200MA'))
                fig.update_layout(title=f"{target} 歷史走勢", template="plotly_dark", height=500)
                st.plotly_chart(fig, use_container_width=True)

                rsi_val = hist['RSI'].iloc[-1]
                cur_p = hist['Close'].iloc[-1]
                ma200_v = hist['MA200'].iloc[-1]
                
                c1, c2, c3 = st.columns(3)
                c1.metric("目前股價", f"${cur_p:.2f}")
                c2.metric("RSI (14D)", f"{rsi_val:.1f}")
                c3.metric("P/E (估值)", info.get('forwardPE', 'N/A'))

                score = 0
                if cur_p > ma200_v: score += 40
                if 30 <= rsi_val <= 65: score += 30
                if (info.get('forwardPE', 100) < 50) or (info.get('quoteType') == 'ETF'): score += 30

                st.markdown(f"### 系統綜合評分：{score} 分")
