import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import scipy.stats as stats
from scipy.optimize import minimize

# ==========================================
# 1. 核心量化引擎 (強化防擋機制 + MACD)
# ==========================================
def format_ticker(ticker):
    t = str(ticker).upper().replace("/", "-").replace(".", "-").strip()
    if t.isdigit() and len(t) == 4: return t + ".TW"
    if t == "BRK.B" or t == "BRK/B": return "BRK-B"
    return t

@st.cache_data(ttl=3600)
def fetch_financial_data(ticker_name):
    try:
        clean_ticker = format_ticker(ticker_name)
        ticker_obj = yf.Ticker(clean_ticker)
        history = ticker_obj.history(period="2y")
        
        if history.empty: 
            return None, {}, f"找不到代號 {clean_ticker}"
        
        info = {}
        try:
            info = ticker_obj.info if hasattr(ticker_obj, 'info') else {}
        except:
            pass 
            
        return history, info, None
    except Exception as e: 
        return None, {}, str(e)

def calculate_indicators(df):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
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

def calculate_mdd(cumulative_returns):
    roll_max = cumulative_returns.cummax()
    return ((cumulative_returns - roll_max) / roll_max).min()

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8-sig')

# ==========================================
# 2. UI 視覺設計與 CSS
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
    .stButton>button:hover { background-color: rgba(96, 165, 250, 0.1) !important; box-shadow: 0 0 20px rgba(96, 165, 250, 0.3) !important; color: #ffffff !important; }
    .report-card { padding: 30px; border-radius: 15px; margin-bottom: 25px; border: 1px solid #334155; background-color: #1e293b; }
    .prescription-box { padding: 15px; border-radius: 10px; background-color: rgba(96, 165, 250, 0.1); border-left: 5px solid #60a5fa; margin-top: 15px; }
    .ttest-box { padding: 15px; border-radius: 10px; background-color: rgba(167, 139, 250, 0.1); border-left: 5px solid #a78bfa; margin-top: 15px; }
    .scenario-box { padding: 15px; border-radius: 10px; background-color: rgba(74, 222, 128, 0.05); border: 1px solid #334155; margin-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ 資本市場投資組合量化分析系統")

# --- 側邊欄 ---
with st.sidebar:
    st.markdown("### 🌍 市場監控中心")
    
    tnx_h, _, _ = fetch_financial_data("^TNX")
    spy_h, _, _ = fetch_financial_data("SPY")
    tw_h, _, _ = fetch_financial_data("0050")
    
    rf_rate = 0.04
    spy_ret = 0.10
    expected_market_ret = 0.095 # 期末報告 CAPM 預期大盤報酬
    
    if tnx_h is not None and not tnx_h.empty:
        rf_rate = tnx_h['Close'].iloc[-1] / 100
        st.metric("美債 10Y (無風險利率)", f"{rf_rate*100:.2f}%")
    else:
        st.metric("美債 10Y (無風險利率)", "4.00% (系統預設)")
        
    if spy_h is not None and not spy_h.empty:
        spy_ret = spy_h['Close'].pct_change(252).iloc[-1]
        st.metric("美股 S&P 500 近一年報酬", f"{spy_ret*100:.2f}%")
        fig_side = px.line(spy_h.tail(45), y='Close', template="plotly_dark").update_traces(line_color='#60a5fa')
        fig_side.update_layout(height=130, margin=dict(l=0,r=0,t=0,b=0), xaxis_visible=False, yaxis_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_side, use_container_width=True, config={'displayModeBar': False})
        
    if tw_h is not None and not tw_h.empty:
        tw_ret = tw_h['Close'].pct_change(252).iloc[-1]
        st.metric("台股 0050 近一年報酬", f"{tw_ret*100:.2f}%")
        
    st.info("💡 系統狀態：含期末報告運算模組")

# 新增 5 個 Tabs 
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 AI 市場診斷", "🛡️ 現有持倉深度績效與診斷", "🎓 期末報告專用 (最佳化引擎)", "⏳ 歷史回測與匯出", "📖 模型說明"])

# --- Tab 1: AI 診斷 ---
with tab1:
    col_in, _ = st.columns([2, 2])
    raw_ticker = col_in.text_input("輸入股票代號 (美股如 NVDA / 台股如 2330)", "2330")
    
    if raw_ticker:
        with st.spinner('正在掃描技術指標與動能...'):
            hist, info, err = fetch_financial_data(raw_ticker)
            if hist is not None and not hist.empty:
                hist = calculate_indicators(hist)
                f_dates, f_preds, f_intervals = get_ai_prediction_model(hist)
                cur_p = hist['Close'].iloc[-1]
                target_p = f_preds[-1]
                expected_ret = ((target_p - cur_p) / cur_p) * 100
                bg, border, txt = ("rgba(20, 83, 45, 0.4)", "#4ade80", "多頭趨勢 / Buy") if expected_ret > 2.0 else ("rgba(127, 29, 29, 0.4)", "#f87171", "空頭預警 / Sell") if expected_ret < -2.0 else ("rgba(30, 41, 59, 0.6)", "#94a3b8", "中立觀望 / Hold")

                st.markdown(f"<div class='report-card' style='background-color: {bg}; border-color: {border};'><h3 style='margin:0; color: white !important;'>🤖 AI 智能評級：{txt}</h3><p style='margin-top:10px; font-size:18px; color: white !important;'>預估 7 日目標：<b>${target_p:.2f}</b> | 期望收益：<b>{expected_ret:+.2f}%</b></p></div>", unsafe_allow_html=True)

                plot_data = hist.tail(150).copy()
                str_dates = plot_data.index.strftime('%Y-%m-%d')
                
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.08)
                fig.add_trace(go.Candlestick(x=str_dates, open=plot_data['Open'], high=plot_data['High'], low=plot_data['Low'], close=plot_data['Close'], name='歷史走勢'), row=1, col=1)
                fig.add_trace(go.Scatter(x=str_dates, y=plot_data['MA10'], line=dict(color='#81d4fa', width=1), name='10MA'), row=1, col=1)
                fig.add_trace(go.Scatter(x=str_dates, y=plot_data['MA50'], line=dict(color='#fbbf24', width=1.2), name='50MA'), row=1, col=1)
                
                macd_colors = ['#4ade80' if val >= 0 else '#f87171' for val in plot_data['MACD_Hist']]
                fig.add_trace(go.Bar(x=str_dates, y=plot_data['MACD_Hist'], marker_color=macd_colors, name='MACD 柱狀圖'), row=2, col=1)
                fig.add_trace(go.Scatter(x=str_dates, y=plot_data['MACD'], line=dict(color='#60a5fa', width=1.5), name='MACD'), row=2, col=1)
                
                fig.update_layout(template="plotly_dark", height=600, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=30, b=0))
                fig.update_xaxes(type='category', nticks=10, rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("即時股價", f"${cur_p:.2f}")
                c2.metric("RSI 指標 (14D)", f"{hist['RSI'].iloc[-1]:.1f}")
                c3.metric("本益比 (PE)", f"{info.get('forwardPE', 'N/A') if isinstance(info, dict) else 'N/A'}")
                c4.metric("市場風險 Beta", f"{info.get('beta', 'N/A') if isinstance(info, dict) else 'N/A'}")
            else:
                st.error("⚠️ 無法取得該標的數據。")

# --- 持倉預設值 ---
if "portfolio_df" not in st.session_state:
    st.session_state.portfolio_df = pd.DataFrame([
        {"代號": "VOO",   "持有股數": 10,  "平均成本": 450.00},
        {"代號": "BRK.B", "持有股數": 15,  "平均成本": 400.00},
        {"代號": "GOOGL", "持有股數": 30,  "平均成本": 130.00},
        {"代號": "NVDA",  "持有股數": 10,  "平均成本": 100.00},
        {"代號": "PLTR",  "持有股數": 15,  "平均成本": 35.00}
    ])

# --- Tab 2: 完整投資組合診斷 ---
with tab2:
    def color_pnl_cells(val):
        color = '#4ade80' if val >= 0 else '#f87171'
        return f'color: {color} !important; font-weight: bold;'

    st.markdown("### 💰 資金水位管理")
    cash_twd = st.number_input("💵 目前閒置現金 (台幣 TWD)", min_value=0.0, value=50000.0, step=1000.0)
    st.markdown("---")
    
    st.markdown("### 📈 現有股票持倉")
    edited = st.data_editor(st.session_state.portfolio_df, num_rows="dynamic", use_container_width=True)
    
    if st.button("🚀 執行 AI 量化診斷"):
        st.session_state.portfolio_df = edited
        with st.spinner('AI 正在運算數據中...'):
            assets_data, hist_dict = [], {}
            stock_cost, stock_val, tech_count = 0, 0, 0
            tech_tickers = ['QQQ', 'QQQM', 'NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'PLTR', 'ARKF', 'SMH', 'AAOI', '2330.TW', '2330']
            failed_tickers = [] 
            
            usd_twd_rate = 32.2 
            try:
                rate_data = yf.Ticker("TWD=X").history(period="1d")
                if not rate_data.empty: usd_twd_rate = rate_data['Close'].iloc[-1]
            except: pass
            
            cash_usd = cash_twd / usd_twd_rate

            for _, row in edited.iterrows():
                raw_ticker = str(row["代號"]).strip()
                if not raw_ticker: continue
                
                ticker = format_ticker(raw_ticker)
                h, i, err = fetch_financial_data(ticker)
                
                if h is not None and not h.empty:
                    h = calculate_indicators(h)
                    cur_price = h['Close'].iloc[-1]
                    asset_cost = float(row["持有股數"]) * float(row["平均成本"])
                    asset_val = float(row["持有股數"]) * cur_price
                    
                    stock_cost += asset_cost
                    stock_val += asset_val
                    
                    if ticker in tech_tickers: tech_count += 1
                    beta_val = i.get('beta', 1.0) if isinstance(i, dict) and i.get('beta') is not None else 1.0
                    assets_data.append({"股票": ticker, "即時現價": cur_price, "總成本": asset_cost, "目前市值": asset_val, "未實現損益": asset_val - asset_cost, "報酬率(%)": ((asset_val - asset_cost)/asset_cost)*100 if asset_cost>0 else 0, "Beta": beta_val, "RSI": h['RSI'].iloc[-1]})
                    hist_dict[ticker] = h['Close']
                else:
                    failed_tickers.append(raw_ticker)
            
            valid_idx = next((v.index for v in hist_dict.values() if v is not None), None)
            hist_dict["CASH (TWD)"] = pd.Series(1.0, index=valid_idx) if valid_idx is not None else pd.Series([1.0, 1.0])

            if failed_tickers:
                st.warning(f"⚠️ 以下標的暫時無法載入：{', '.join(failed_tickers)}")

            if assets_data:
                res_df = pd.DataFrame(assets_data)
                st.session_state['res_df'] = res_df 
                st.session_state['hist_dict'] = hist_dict
                st.session_state['cash_usd'] = cash_usd
                
                total_val = stock_val + cash_usd 
                stock_pnl = stock_val - stock_cost
                stock_pnl_pct = (stock_pnl / stock_cost)*100 if stock_cost > 0 else 0
                pnl_color = "#4ade80" if stock_pnl >= 0 else "#f87171"
                
                st.info(f"💱 目前即時匯率：1 USD = {usd_twd_rate:.2f} TWD")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("股票投入總成本 (USD)", f"${stock_cost:,.2f}")
                m2.metric("組合總市值含現金 (USD)", f"${total_val:,.2f}")
                m3.markdown(f"""<div style="padding: 15px; border-radius: 10px; border: 1px solid #334155; background-color: #1e293b; text-align: center;"><div style="color: #94a3b8; font-size: 14px; margin-bottom: 5px;">股票未實現損益</div><div style="color: {pnl_color}; font-size: 32px; font-weight: bold;">${stock_pnl:+,.2f}</div></div>""", unsafe_allow_html=True)
                m4.markdown(f"""<div style="padding: 15px; border-radius: 10px; border: 1px solid #334155; background-color: #1e293b; text-align: center;"><div style="color: #94a3b8; font-size: 14px; margin-bottom: 5px;">純股票投資報酬率</div><div style="color: {pnl_color}; font-size: 32px; font-weight: bold;">{stock_pnl_pct:+.2f}%</div></div>""", unsafe_allow_html=True)
                
                styled_table = res_df[['股票', '即時現價', '總成本', '目前市值', '未實現損益', '報酬率(%)']].style.format({'即時現價': '${:.2f}', '總成本': '${:,.2f}', '目前市值': '${:,.2f}', '未實現損益': '${:+,.2f}', '報酬率(%)': '{:+.2f}%'}).map(color_pnl_cells, subset=['未實現損益', '報酬率(%)']).set_table_styles([{'selector': 'table', 'props': [('width', '100%'), ('background-color', '#1e293b'), ('border-radius', '10px')]}, {'selector': 'th', 'props': [('background-color', '#0f172a'), ('color', '#94a3b8')]}, {'selector': 'td', 'props': [('color', '#f1f5f9')]}]).hide(axis="index").to_html()
                st.markdown(styled_table, unsafe_allow_html=True)

                hist_df = pd.DataFrame(hist_dict).dropna() 
                ret_df = hist_df.pct_change().dropna() 
                corr_matrix = ret_df.drop(columns=['CASH (TWD)']).corr() 

                weights = []
                weighted_beta, portfolio_return = 0, 0
                for idx, row in res_df.iterrows():
                    w = row["currently_val" if "currently_val" in row else "目前市值"] / total_val if total_val > 0 else 0
                    weights.append(w)
                    res_df.at[idx, '權重'] = w
                    weighted_beta += row["Beta"] * w
                    portfolio_return += hist_df[row['股票']].pct_change(252).iloc[-1] * w
                
                cash_w = cash_usd / total_val if total_val > 0 else 0
                weights.append(cash_w) 

                port_daily_ret = ret_df.dot(weights)
                port_mdd = calculate_mdd((1 + port_daily_ret).cumprod())
                jensen_alpha = portfolio_return - (rf_rate + weighted_beta * (spy_ret - rf_rate))

                # ==========================================
                # 🌟 老師要求：直接在第二頁同步執行最高夏普值規劃求解 🌟
                # ==========================================
                tickers = res_df['股票'].tolist()
                price_df_opt = pd.DataFrame({t: hist_dict[t] for t in tickers}).dropna()
                ret_df_opt = price_df_opt.pct_change().dropna()
                cov_matrix_opt = ret_df_opt.cov().values * 252
                
                mean_returns_opt = []
                for col in tickers:
                    beta_val = res_df[res_df['股票'] == col]['Beta'].values[0]
                    mean_returns_opt.append(rf_rate + beta_val * (expected_market_ret - rf_rate))
                mean_returns_opt = np.array(mean_returns_opt)
                
                def neg_sharpe_opt(w_opt, m_ret, c_mat, rf):
                    p_ret = np.sum(m_ret * w_opt)
                    p_std = np.sqrt(np.dot(w_opt.T, np.dot(c_mat, w_opt)))
                    return -(p_ret - rf) / (p_std + 1e-9)
                
                num_assets = len(tickers)
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                bounds = tuple((0.05, 0.45) for _ in range(num_assets))
                init_guess = num_assets * [1./num_assets,]
                opt_res = minimize(neg_sharpe_opt, init_guess, args=(mean_returns_opt, cov_matrix_opt, rf_rate), method='SLSQP', bounds=bounds, constraints=constraints)
                opt_weights = opt_res.x
                
                # 計算最佳化組合預期年化指標
                opt_port_ret = np.sum(mean_returns_opt * opt_weights)
                opt_port_std = np.sqrt(np.dot(opt_weights.T, np.dot(cov_matrix_opt, opt_weights)))
                
                # 建立對比資料表
                opt_compare_list = []
                for idx, t in enumerate(tickers):
                    current_w = res_df[res_df['股票'] == t]['權重'].values[0]
                    opt_compare_list.append({
                        "資產代號": t,
                        "現有權重": f"{current_w:.2%}",
                        "目標最佳權重(最高夏普)": f"{opt_weights[idx]:.2%}",
                        "建議調整方向": "🟢 逢低加碼" if opt_weights[idx] > current_w else "🔴 逢高減碼"
                    })
                df_opt_compare = pd.DataFrame(opt_compare_list)

                # ==========================================
                # 🌟 壓力測試 (Scenario Analysis) 與 再平衡金額計算 🌟
                # ==========================================
                # 市場常態狀況 (未來1年預期價值)
                normal_val = total_val * (1 + opt_port_ret)
                # 市場極端樂觀 (牛市前5%概率，Z=1.65)
                bull_val = total_val * (1 + opt_port_ret + 1.65 * opt_port_std)
                # 市場極端悲觀 (熊市後5%概率，Z=1.65)
                bear_val = total_val * (1 + opt_port_ret - 1.65 * opt_port_std)

                if spy_h is not None and not spy_h.empty:
                    spy_daily_returns = spy_h['Close'].pct_change().dropna()
                    aligned = pd.DataFrame({'Port': port_daily_ret, 'SPY': spy_daily_returns}).dropna()
                    if not aligned.empty:
                        t_stat, p_value = stats.ttest_ind(aligned['Port'], aligned['SPY'], equal_var=False)
                        ttest_result = f"P-value 為 {p_value:.4f} < 0.05。<br><span style='color:#4ade80;'>✅ 拒絕虛無假說，此組合與大盤有**統計顯著差異**。</span>" if p_value < 0.05 else f"P-value 為 {p_value:.4f} >= 0.05。<br><span style='color:#fbbf24;'>⚠️ 無法拒絕虛無假說。</span>"
                    else: t_stat, ttest_result = 0, "無足夠數據"
                else: t_stat, ttest_result = 0, "無大盤數據"

                st.divider()

                eval_color = "#60a5fa"
                st.markdown(f"""
                    <div class="report-card" style="border-left: 10px solid {eval_color};">
                        <h3 style="color: {eval_color}; margin:0;">AI 綜合診斷報告 (已將現金部位納入風險評估)</h3>
                        <p style="margin-top:20px; font-size:18px; line-height:1.7;">
                            <b>組合加權 Beta：</b>{weighted_beta:.2f} (市場敏感度)<br>
                            <b>Jensen's Alpha：</b><span style='color: {"#4ade80" if jensen_alpha > 0 else "#f87171"}; font-weight: bold;'>{jensen_alpha*100:+.2f}%</span><br>
                            <b>歷史最大回撤 (MDD)：</b><span style='color: #f87171; font-weight: bold;'>{port_mdd*100:.2f}%</span><br>
                        </p>
                        <div class="ttest-box">
                            <strong style="color:#a78bfa; font-size:18px;">🔬 統計顯著性檢定 (Two-Sample T-Test)：</strong><br>
                            <span style="color:#f1f5f9;"><b>T-Statistic:</b> {t_stat:.4f} <br><b>結論：</b> {ttest_result}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # 🌟 新增：最佳化分配權重對比表 🌟
                st.markdown("### 🏆 老師要求項目：最高夏普值資產配置對比")
                st.dataframe(df_opt_compare, use_container_width=True)
                
                # 🌟 新增：市場好/壞壓力測試與動態再平衡機制 🌟
                st.markdown("### 📊 未來 1 年市場極端情境預估 (壓力測試)")
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("🐂 市場極端樂觀 (牛市景氣)", f"${bull_val:,.2f}", "+ 1.65 Std Dev")
                sc2.metric("📊 市場常態狀況 (平穩景氣)", f"${normal_val:,.2f}", "CAPM 預期值")
                sc3.metric("🐻 市場極端悲觀 (熊市風暴)", f"${bear_val:,.2f}", "- 1.65 Std Dev")
                
                st.markdown(f"""
                <div class="report-card" style="border-top: 4px solid #fbbf24; background-color: #1e293b; padding: 20px; margin-top: 15px;">
                    <h4 style="color:#fbbf24; margin:0; font-size: 18px;">🔄 基金經理人動態再平衡門檻策略 (Rebalancing Rule)</h4>
                    <p style="font-size:16px; line-height:1.6; margin-top:10px; color:#cbd5e1 !important;">
                        為了維持投組在<b>最高夏普值</b>的完美防禦狀態，本系統設定以下<b>動態再平衡風控制度</b>：<br>
                        1️⃣ <b>權重偏離門檻</b>：當任一股票的「現有權重」與「目標最佳權重」絕對偏差<b>超過 ±5%</b> 時（例如 VOO 偏離至 50% 或跌破 40%），系統將發出交易警報。<br>
                        2️⃣ <b>資產重新分配點</b>：當投資組合總價值達到 <b>${bull_val:,.2f}</b> (獲利結清點) 或跌破 <b>${bear_val:,.2f}</b> (資產防禦點) 時，必須強制將所有股票部位賣出重組，將資金調回上方表格推薦之目標比例，以鎖定利潤、控制最大回撤。
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                pie_df = res_df.copy()
                pie_df = pd.concat([pie_df, pd.DataFrame([{"股票": "CASH (TWD)", "目前市值": cash_usd}])], ignore_index=True)

                c_pie, c_heat = st.columns(2)
                c_pie.plotly_chart(px.pie(pie_df, values='目前市值', names='股票', hole=0.4, title="真實資產權重配比 (含現金)", template="plotly_dark"), use_container_width=True)
                fig_heat = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', origin='lower', title="股票資產相關性熱力圖")
                c_heat.plotly_chart(fig_heat, use_container_width=True)

# --- Tab 3: 期末報告專用 (已修正權重限制 5% ~ 45%) ---
with tab3:
    st.markdown("### 🏆 期末報告：夏普值最佳化演算 (規劃求解)")
    st.info("此區塊會自動抓取你『Tab 2 輸入的持倉標的』(自動排除現金)，透過 CAPM 模型計算預期報酬，並找出最高夏普值的最佳權重。")
    
    if st.button("🚀 產出期末報告分析數據"):
        if 'res_df' in st.session_state and 'hist_dict' in st.session_state:
            stock_df = st.session_state['res_df'][st.session_state['res_df']['股票'] != 'CASH (TWD)']
            tickers = stock_df['股票'].tolist()
            
            if len(tickers) < 4 or len(tickers) > 6:
                st.warning(f"⚠️ 注意：目前選取了 {len(tickers)} 檔股票。教授規定是 4-6 檔，建議回到 Tab 2 調整標的數量。")
            
            hist_dict = st.session_state['hist_dict']
            spy_daily = spy_h['Close'].pct_change().dropna() if spy_h is not None else None
            
            price_df = pd.DataFrame({t: hist_dict[t] for t in tickers}).dropna()
            ret_df = price_df.pct_change().dropna()
            
            ind_stats = []
            for col in tickers:
                daily_ret = ret_df[col]
                past_ret = daily_ret.mean() * 252
                past_std = daily_ret.std() * np.sqrt(252)
                
                if spy_daily is not None:
                    aligned = pd.concat([daily_ret, spy_daily], axis=1).dropna()
                    cov_mat = np.cov(aligned.iloc[:,0], aligned.iloc[:,1])
                    beta = cov_mat[0,1] / cov_mat[1,1]
                else: beta = 1.0
                
                past_sharpe = (past_ret - rf_rate) / past_std if past_std != 0 else 0
                future_ret = rf_rate + beta * (expected_market_ret - rf_rate)
                future_std = past_std 
                future_sharpe = (future_ret - rf_rate) / future_std if future_std != 0 else 0
                
                ind_stats.append({
                    "標的": col, "過去1年報酬率": past_ret, "過去1年標準差": past_std, "過去1年Sharpe": past_sharpe,
                    "Beta值": beta, "預期未來1年報酬率(CAPM)": future_ret, "預期未來1年標準差": future_std, "預期未來1年Sharpe": future_sharpe
                })
                
            df_ind = pd.DataFrame(ind_stats)
            mean_returns = df_ind.set_index("標的")["預期未來1年報酬率(CAPM)"].values
            cov_matrix = ret_df.cov().values * 252 
            
            def neg_sharpe(weights, mean_ret, cov_mat, rf):
                p_ret = np.sum(mean_ret * weights)
                p_std = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))
                return -(p_ret - rf) / (p_std + 1e-9)

            num_assets = len(tickers)
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0.05, 0.45) for _ in range(num_assets))
            init_guess = num_assets * [1./num_assets,]
            
            opt_res = minimize(neg_sharpe, init_guess, args=(mean_returns, cov_matrix, rf_rate), method='SLSQP', bounds=bounds, constraints=constraints)
            opt_weights = opt_res.x
            
            port_future_ret = np.sum(mean_returns * opt_weights)
            port_future_std = np.sqrt(np.dot(opt_weights.T, np.dot(cov_matrix, opt_weights)))
            port_future_sharpe = (port_future_ret - rf_rate) / port_future_std
            
            st.success("✅ 規劃求解完成！已成功找到具備實務分散性的最高 Sharpe 值權重配置。")
            
            st.markdown("#### 1. 個別標的分析 (過去 1 年 vs 預期未來 1 年)")
            st.dataframe(df_ind.style.format({
                "過去1年報酬率": "{:.2%}", "過去1年標準差": "{:.2%}", "過去1年Sharpe": "{:.2f}",
                "Beta值": "{:.2f}", "預期未來1年報酬率(CAPM)": "{:.2%}", "預期未來1年標準差": "{:.2%}", "預期未來1年Sharpe": "{:.2f}"
            }), use_container_width=True)
            
            st.markdown("#### 2. 夏普值極大化配置結果 (受 5%-45% 邊界限制)")
            c_chart, c_text = st.columns(2)
            fig_pie_opt = px.pie(values=opt_weights, names=tickers, hole=0.4, title="🏆 實務最佳權重配比", template="plotly_dark")
            c_chart.plotly_chart(fig_pie_opt, use_container_width=True)
            
            df_weights = pd.DataFrame({"標的": tickers, "最佳配置權重": opt_weights})
            c_text.dataframe(df_weights.style.format({"最佳配置權重": "{:.2%}"}), use_container_width=True)
            
            st.markdown(f"""
            <div class="report-card">
                <h4 style="color:#fbbf24; margin-top:0;">🌟 最佳化投資組合整體預期表現</h4>
                <p style="font-size:18px;"><b>預期年化報酬率 (E(R))：</b> <span style="color:#4ade80;">{port_future_ret:.2%}</span></p>
                <p style="font-size:18px;"><b>預期年化風險 (Std Dev)：</b> {port_future_std:.2%}</p>
                <p style="font-size:18px;"><b>極大化夏普值 (Max Sharpe)：</b> <span style="color:#fbbf24; font-size:24px; font-weight:bold;">{port_future_sharpe:.2f}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
        else: st.warning("請先在「Tab 2: 現有持倉深度績效與診斷」頁籤點擊執行運算！")

# --- Tab 4: 回測與報表 ---
with tab4:
    st.markdown("### 📊 歷史回測與投資組合表現")
    if 'hist_dict' in st.session_state and 'res_df' in st.session_state:
        res_df = st.session_state['res_df']
        hist_dict = st.session_state['hist_dict']
        cash_usd = st.session_state.get('cash_usd', 0)
        
        total_val = res_df['目前市值'].sum() + cash_usd
        weights = (res_df['目前市值'] / total_val).tolist()
        
        cash_w = cash_usd / total_val if total_val > 0 else 0
        weights.append(cash_w)
        
        clean_hist_dict = {}
        for k, v in hist_dict.items():
            if v is not None:
                clean_hist_dict[k] = v
                    
        ret_df = pd.DataFrame(clean_hist_dict).pct_change().dropna()
        port_cum_ret = (1 + ret_df.dot(weights)).cumprod()
        
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=port_cum_ret.index, y=port_cum_ret, name='你的真實投資組合(含現金避險)', line=dict(color='#4ade80', width=2.5)))
        
        if spy_h is not None and not spy_h.empty:
            spy_cum_ret = (1 + spy_h['Close'].pct_change().dropna()).cumprod()
            fig_bt.add_trace(go.Scatter(x=spy_cum_ret.index, y=spy_cum_ret, name='大盤 (S&P 500)', line=dict(color='#94a3b8', width=1.5, dash='dash')))
            
        fig_bt.update_layout(template="plotly_dark", title="近兩年資產資金曲線 (Equity Curve)", height=450, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_bt, use_container_width=True)
        
        st.markdown("### 📥 報表匯出")
        st.download_button(label="📊 點擊下載量化分析報告 (CSV)", data=convert_df_to_csv(res_df), file_name=f"AlphaCheck_Report_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
    else: st.warning("請先在「Tab 2: 現有持倉深度績效與診斷」頁籤點擊執行運算。")

# --- Tab 5: 說明 ---
with tab5:
    st.header("📖 模型理論與實務應用")
    st.markdown("""
    1. **Mark-to-Market**：按市值計價。
    2. **CAPM & Jensen's Alpha**：量化超額報酬。
    3. **T-test**：檢定超額報酬是否為隨機機率。
    4. **MDD**：最大回撤。
    5. **MACD 動能**：結合快慢線與柱狀圖，精準捕捉趨勢反轉點。
    6. **期末報告規劃求解 (Tab 3)**：使用 Scipy Minimize 尋找最大化 Sharpe Ratio 之權重配置，**並設定 5% ~ 45% 的權重邊界**，確保投資組合符合實務上的分散風險原則。
    """)
