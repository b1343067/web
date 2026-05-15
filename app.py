import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ==========================================
# 1. 核心計算引擎
# ==========================================

@st.cache_data(ttl=3600)
def fetch_financial_data(ticker_name):
    try:
        clean_ticker = ticker_name.upper().replace("/", "-").strip()
        ticker_obj = yf.Ticker(clean_ticker)
        history = ticker_obj.history(period="2y")
        info = ticker_obj.info if hasattr(ticker_obj, 'info') else {}
        if history.empty: return None, None, f"找不到代號 {clean_ticker}"
        return history, info, None
    except Exception as e:
        return None, None, str(e)

def calculate_indicators(df):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    return df

def get_ai_prediction_model(df, days=7):
    df_p = df.tail(90).reset_index()
    df_p['Date_n'] = pd.to_datetime(df_p['Date']).apply(lambda x: x.toordinal())
    X = df_p[['Date_n']].values
    y = df_p['Close'].values
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    last_d_n = df_p['Date_n'].max()
    future_n = np.array([last_d_n + i for i in range(1, days + 1)]).reshape(-1, 1)
    preds = model.predict(poly.transform(future_n))
    base_std = df['Close'].tail(30).std()
    intervals = [base_std * (1 + (i * 0.2)) for i in range(len(preds))]
    last_d = df_p['Date'].max()
    future_d = [last_d + timedelta(days=i) for i in range(1, days + 1)]
    return future_d, preds, intervals

# 計算最大回撤 (MDD)
def calculate_mdd(cumulative_returns):
    roll_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - roll_max) / roll_max
    return drawdown.min()

# ==========================================
# 2. UI 視覺設計 (完美相容 config.toml)
# ==========================================

st.set_page_config(page_title="AlphaCheck Elite", layout="wide")

st.markdown("""
    <style>
    h1, h2, h3, p, label { color: #f1f5f9 !important; }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 15px !important; }
    [data-testid="stMetricValue"] { color: #f8fafc !important; font-weight: bold !important; }

    .stButton>button {
        background-color: transparent !important; color: #60a5fa !important;
        border: 2px solid #60a5fa !important; border-radius: 25px !important;
        padding: 10px 40px !important; font-weight: 700 !important; width: 100% !important;
        transition: 0.3s all ease-in-out; text-transform: uppercase; letter-spacing: 1px;
    }
    .stButton>button:hover {
        background-color: rgba(96, 165, 250, 0.1) !important;
        box-shadow: 0 0 20px rgba(96, 165, 250, 0.3) !important; color: #ffffff !important;
    }

    .report-card { padding: 30px; border-radius: 15px; margin-bottom: 25px; border: 1px solid #334155; backdrop-filter: blur(10px); }
    .prescription-box { padding: 15px; border-radius: 10px; background-color: rgba(96, 165, 250, 0.1); border-left: 5px solid #60a5fa; margin-top: 15px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ AlphaCheck Elite: 專業投資決策終端")

# --- 側邊欄 ---
with st.sidebar:
    st.markdown("### 🌍 市場監控中心")
    tnx_h, _, _ = fetch_financial_data("^TNX")
    spy_h, _, _ = fetch_financial_data("SPY")
    rf_rate = 0.04
    spy_ret = 0.10
    
    if tnx_h is not None:
        rf_rate = tnx_h['Close'].iloc[-1] / 100
        st.metric("美債 10Y (無風險利率 Rf)", f"{rf_rate*100:.2f}%")
    if spy_h is not None:
        spy_ret = spy_h['Close'].pct_change(252).iloc[-1]
        st.metric("S&P 500 年化報酬 (Rm)", f"{spy_ret*100:.2f}%")
        fig_side = px.line(spy_h.tail(45), y='Close', template="plotly_dark").update_traces(line_color='#60a5fa')
        fig_side.update_layout(height=130, margin=dict(l=0,r=0,t=0,b=0), xaxis_visible=False, yaxis_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_side, use_container_width=True, config={'displayModeBar': False})
    st.info("💡 系統已啟用: 即時 MTM 損益 / CAPM 評價 / 熱力圖 / AI 處方。")

tab1, tab2, tab3 = st.tabs(["🔍 AI 市場診斷", "🛡️ 投資組合深度績效與診斷", "📖 模型說明"])

# --- Tab 1: AI 診斷 (保持不變) ---
with tab1:
    col_in, _ = st.columns([2, 2])
    raw_ticker = col_in.text_input("輸入美股代號 (如 BRK/B, VOO, NVDA)", "VOO")
    
    if raw_ticker:
        target = raw_ticker.upper().replace("/", "-").strip()
        with st.spinner('正在掃描技術指標...'):
            hist, info, err = fetch_financial_data(target)
            if not err:
                hist = calculate_indicators(hist)
                f_dates, f_preds, f_intervals = get_ai_prediction_model(hist)
                
                cur_p = hist['Close'].iloc[-1]
                target_p = f_preds[-1]
                expected_ret = ((target_p - cur_p) / cur_p) * 100
                
                bg, border, txt = ("rgba(20, 83, 45, 0.4)", "#4ade80", "多頭趨勢 / Buy") if expected_ret > 2.0 else ("rgba(127, 29, 29, 0.4)", "#f87171", "空頭預警 / Sell") if expected_ret < -2.0 else ("rgba(30, 41, 59, 0.6)", "#94a3b8", "中立觀望 / Hold")

                st.markdown(f"<div class='report-card' style='background-color: {bg}; border-color: {border};'><h3 style='margin:0; color: white !important;'>🤖 AI 智能評級：{txt}</h3><p style='margin-top:10px; font-size:18px; color: white !important;'>預估 7 日目標：<b>${target_p:.2f}</b> | 期望收益：<b>{expected_ret:+.2f}%</b></p></div>", unsafe_allow_html=True)

                plot_data = hist.tail(150)
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=plot_data.index, open=plot_data['Open'], high=plot_data['High'], low=plot_data['Low'], close=plot_data['Close'], name='歷史走勢', increasing_line_color='#4ade80', decreasing_line_color='#f87171'))
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA10'], line=dict(color='#81d4fa', width=1), name='10MA (短)'))
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA50'], line=dict(color='#fbbf24', width=1.2), name='50MA (中)'))
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA200'], line=dict(color='#94a3b8', width=2), name='200MA (生命線)'))
                fig.add_trace(go.Scatter(x=f_dates + f_dates[::-1], y=[f_preds[i] + f_intervals[i] for i in range(len(f_preds))] + [f_preds[i] - f_intervals[i] for i in range(len(f_preds))][::-1], fill='toself', fillcolor='rgba(96, 165, 250, 0.1)', line_color='rgba(0,0,0,0)', name='AI 波動預期'))
                fig.add_trace(go.Scatter(x=f_dates, y=f_preds, line=dict(color='#ffffff', dash='dash', width=2), name='AI 核心路徑'))
                fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("即時股價", f"${cur_p:.2f}")
                c2.metric("RSI 指標 (14D)", f"{hist['RSI'].iloc[-1]:.1f}")
                c3.metric("本益比 (PE)", f"{info.get('forwardPE', 'N/A')}")
                c4.metric("市場風險 Beta", f"{info.get('beta', 'N/A')}")

# --- Tab 2: 究極完全體 (損益 + CAPM + 熱力圖 + MDD + AI處方) ---
with tab2:
    def color_pnl_cells(val):
        color = '#4ade80' if val >= 0 else '#f87171'
        return f'color: {color} !important; font-weight: bold;'

    st.markdown("### 💰 輸入持倉資訊 (持股數與平均成本)")
    p_df = pd.DataFrame([
        {"代號": "VOO",  "持有股數": 20, "平均成本": 450.0},
        {"代號": "QQQM", "持有股數": 20, "平均成本": 170.0},
        {"代號": "NVDA", "持有股數": 15, "平均成本": 115.0}
    ])
    edited = st.data_editor(p_df, num_rows="dynamic", use_container_width=True)
    
    if st.button("🚀 即時結算與 AI 量化診斷"):
        with st.spinner('AI 正在計算即時損益、生成相關性矩陣與最大回撤...'):
            assets_data = []
            hist_dict = {} # 用來存儲歷史價格計算矩陣
            total_invested_cost = 0
            total_current_value = 0
            tech_count = 0
            tech_tickers = ['QQQ', 'QQQM', 'NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX', 'PLTR', 'ARKF', 'SMH', 'SOXX']

            for _, row in edited.iterrows():
                ticker = row["代號"].upper()
                h, i, _ = fetch_financial_data(ticker)
                if h is not None:
                    h = calculate_indicators(h)
                    cur_price = h['Close'].iloc[-1]
                    shares = row["持有股數"]
                    avg_cost = row["平均成本"]
                    
                    asset_cost = shares * avg_cost
                    asset_val = shares * cur_price
                    pnl = asset_val - asset_cost
                    pnl_pct = (pnl / asset_cost)*100 if asset_cost > 0 else 0
                    
                    total_invested_cost += asset_cost
                    total_current_value += asset_val
                    if ticker in tech_tickers: tech_count += 1
                        
                    assets_data.append({
                        "股票": ticker, "即時現價": cur_price, "總成本": asset_cost, "目前市值": asset_val, 
                        "未實現損益": pnl, "報酬率(%)": pnl_pct, "Beta": i.get('beta', 1.0), "RSI": h['RSI'].iloc[-1]
                    })
                    hist_dict[ticker] = h['Close'] # 存入字典供後續矩陣計算

            res_df = pd.DataFrame(assets_data)
            
            # --- 儀表板與 HTML 表格 ---
            total_pnl = total_current_value - total_invested_cost
            total_pnl_pct = (total_pnl / total_invested_cost)*100 if total_invested_cost > 0 else 0
            pnl_color = "#4ade80" if total_pnl >= 0 else "#f87171"

            st.markdown(f"### 📈 即時損益與風險控制儀表板")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("投入總成本", f"${total_invested_cost:,.2f}")
            m2.metric("目前總市值", f"${total_current_value:,.2f}")
            m3.markdown(f"<div style='padding: 15px; border-radius: 10px; border: 1px solid #334155; background-color: #1e293b; text-align: center;'><span style='color: #94a3b8; font-size: 14px;'>未實現總損益</span><br><span style='color: {pnl_color} !important; font-size: 32px; font-weight: bold;'>${total_pnl:+,.2f}</span></div>", unsafe_allow_html=True)
            m4.markdown(f"<div style='padding: 15px; border-radius: 10px; border: 1px solid #334155; background-color: #1e293b; text-align: center;'><span style='color: #94a3b8; font-size: 14px;'>總體報酬率</span><br><span style='color: {pnl_color} !important; font-size: 32px; font-weight: bold;'>{total_pnl_pct:+.2f}%</span></div>", unsafe_allow_html=True)
            
            styled_table = res_df[['股票', '即時現價', '總成本', '目前市值', '未實現損益', '報酬率(%)']].style.format({
                '即時現價': '${:.2f}', '總成本': '${:,.2f}', '目前市值': '${:,.2f}', '未實現損益': '${:+,.2f}', '報酬率(%)': '{:+.2f}%'
            }).map(color_pnl_cells, subset=['未實現損益', '報酬率(%)']).set_table_styles([
                {'selector': 'table', 'props': [('width', '100%'), ('border-collapse', 'collapse'), ('background-color', '#1e293b'), ('border-radius', '10px'), ('overflow', 'hidden'), ('margin-top', '20px'), ('margin-bottom', '20px')]},
                {'selector': 'th', 'props': [('background-color', '#0f172a'), ('color', '#94a3b8'), ('font-weight', 'bold'), ('padding', '15px'), ('text-align', 'center'), ('border', '1px solid #334155')]},
                {'selector': 'td', 'props': [('color', '#f1f5f9'), ('padding', '12px'), ('text-align', 'center'), ('border', '1px solid #334155')]},
                {'selector': 'tr:hover td', 'props': [('background-color', '#2c3e50')]}
            ]).hide(axis="index").to_html()
            
            st.markdown(styled_table, unsafe_allow_html=True)

            # --- 進階計算：權重、相關性矩陣、最大回撤 (MDD)、CAPM ---
            hist_df = pd.DataFrame(hist_dict).dropna() # 對齊歷史數據
            ret_df = hist_df.pct_change().dropna() # 計算每日回報率
            corr_matrix = ret_df.corr() # 產生相關性矩陣

            weights = []
            weighted_beta, weighted_rsi, portfolio_return = 0, 0, 0
            
            for idx, row in res_df.iterrows():
                weight = row["目前市值"] / total_current_value if total_current_value > 0 else 0
                weights.append(weight)
                res_df.at[idx, '權重'] = weight
                res_df.at[idx, '驅動力'] = row["Beta"] * weight
                weighted_beta += row["Beta"] * weight
                weighted_rsi += row["RSI"] * weight
                
                # 計算個股年化報酬率供 CAPM 使用
                ann_ret = hist_df[row['股票']].pct_change(252).iloc[-1]
                portfolio_return += ann_ret * weight

            # 計算組合歷史曲線與最大回撤 (MDD)
            port_daily_returns = ret_df.dot(weights)
            cumulative_returns = (1 + port_daily_returns).cumprod()
            port_mdd = calculate_mdd(cumulative_returns)
            
            # 計算 CAPM 指標
            jensen_alpha = portfolio_return - (rf_rate + weighted_beta * (spy_ret - rf_rate))

            st.divider()

            # --- AI 智能診斷與開處方 (Rebalancing) ---
            top_stock = res_df.sort_values('權重', ascending=False)['股票'].iloc[0]
            tech_ratio = tech_count / len(res_df) if len(res_df) > 0 else 0
            has_redundancy = "QQQ" in res_df['股票'].values and "QQQM" in res_df['股票'].values

            if tech_ratio > 0.8 and weighted_beta > 1.15:
                eval_title = "高風險：極端成長型集中"
                eval_color = "#f87171"
                eval_content = f"組合極度向科技股傾斜。受 **{top_stock}** 等標的影響，在極端市況下抗跌能力極弱。"
                prescription = f"建議減碼 {top_stock} 等科技標的 15-20%，並將資金轉入低相關性避險資產（如 TLT 美債、BRK-B），以有效降低目前達 {abs(port_mdd*100):.1f}% 的最大回撤風險。"
            elif has_redundancy:
                eval_title = "結構冗餘：標的重疊風險"
                eval_color = "#fbbf24"
                eval_content = "偵測到同時持有 QQQ 與 QQQM，兩者追蹤同一指數，屬於無效的分散風險配置。"
                prescription = "建議將 QQQ 完全轉換為 QQQM（內扣費用較低），釋出的資金部位佈局非科技板塊（如 XLV 醫療、XLP 必需消費）來達到真正的『退可守』。"
            elif 0.9 <= weighted_beta <= 1.25 and tech_ratio < 0.6:
                eval_title = "精英級：均衡核心—衛星配置"
                eval_color = "#60a5fa"
                eval_content = f"配置展現了極高的專業度。以 **{top_stock}** 為核心定海神針，是名副其實的『進可攻、退可守』。"
                prescription = f"目前結構極佳（Alpha: {jensen_alpha*100:+.2f}%）。建議維持現有權重比例，若遇大盤回檔可優先逢低加碼核心標的 {top_stock}。"
            else:
                eval_title = "穩健/防禦型配置"
                eval_color = "#4ade80"
                eval_content = f"組合抗波動能力強。以 **{top_stock}** 等穩健標的為主軸，能提供極佳的資產保護力。"
                prescription = "防禦力極強但可能錯失牛市超額利潤。建議可提撥 5-10% 資金佈局高動能衛星標的（如 NVDA 等高 Beta 股）以提升整體的 Alpha 表現。"

            st.markdown(f"""
                <div class="report-card" style="border-left: 10px solid {eval_color}; background-color: #1e293b;">
                    <h2 style="color: {eval_color}; margin:0;">AI 綜合診斷：{eval_title}</h2>
                    <p style="margin-top:20px; font-size:18px; line-height:1.7; color: white !important;">
                        <b>分析師實話：</b><br>{eval_content}<br><br>
                        <b>組合加權 Beta：</b>{weighted_beta:.2f} (市場敏感度)<br>
                        <b>Jensen's Alpha ($\alpha$)：</b><span style='color: {"#4ade80" if jensen_alpha > 0 else "#f87171"}; font-weight: bold;'>{jensen_alpha*100:+.2f}%</span> (打敗大盤之超額報酬)<br>
                        <b>歷史最大回撤 (MDD)：</b><span style='color: #f87171; font-weight: bold;'>{port_mdd*100:.2f}%</span> (最慘時的跌幅)<br>
                    </p>
                    <div class="prescription-box">
                        <strong style="color:#60a5fa; font-size:18px;">💡 AI 智能處方 (Rebalancing Suggestion)：</strong><br>
                        <span style="color:#f1f5f9;">{prescription}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # --- 視覺大招：熱力圖與配置圖 ---
            c_pie, c_heat = st.columns(2)
            
            # 圓餅圖
            c_pie.plotly_chart(px.pie(res_df, values='權重', names='股票', hole=0.4, title="真實市值權重配比 (Weight)", template="plotly_dark").update_layout(font=dict(color="white")), use_container_width=True)
            
            # 熱力圖 (Correlation Heatmap)
            fig_heat = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', origin='lower', title="資產相關性熱力圖 (Correlation Matrix)")
            fig_heat.update_layout(template="plotly_dark", font=dict(color="white"))
            c_heat.plotly_chart(fig_heat, use_container_width=True)

with tab3:
    st.header("📖 模型理論與實務應用")
    st.markdown("""
    1. **Mark-to-Market (按市值計價)**：採用「當前真實市值」計算動態權重，貼近實盤操作。
    2. **CAPM 與 Jensen's Alpha ($\\alpha$)**：$\\alpha = R_p - [R_f + \\beta_p(R_m - R_f)]$。用於量化投資組合是否創造了高於承擔系統風險下的「超額報酬」。
    3. **最大回撤 (Maximum Drawdown, MDD)**：評估在過去一段時間內，資產從最高點滑落至最低點的幅度，為衡量下行風險的極重要指標。
    4. **資產相關性矩陣 (Correlation Matrix)**：數值越接近 1 代表同漲同跌；越接近 -1 代表走勢相反。真正的「避險」建立在挑選低相關性或負相關的資產上。
    """)
