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

# ==========================================
# 1. 核心量化引擎 (強化防擋機制 + MACD)
# ==========================================
def format_ticker(ticker):
    t = str(ticker).upper().replace("/", "-").replace(".", "-").strip()
    # 智能辨識：4位數字自動加 .TW
    if t.isdigit() and len(t) == 4: return t + ".TW"
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
    .report-card { padding: 30px; border-radius: 15px; margin-bottom: 25px; border: 1px solid #334155; backdrop-filter: blur(10px); }
    .prescription-box { padding: 15px; border-radius: 10px; background-color: rgba(96, 165, 250, 0.1); border-left: 5px solid #60a5fa; margin-top: 15px; }
    .ttest-box { padding: 15px; border-radius: 10px; background-color: rgba(167, 139, 250, 0.1); border-left: 5px solid #a78bfa; margin-top: 15px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ ")

# --- 側邊欄 ---
with st.sidebar:
    st.markdown("### 🌍 市場監控中心")
    
    tnx_h, _, _ = fetch_financial_data("^TNX")
    spy_h, _, _ = fetch_financial_data("SPY")
    tw_h, _, _ = fetch_financial_data("0050")
    
    rf_rate = 0.04
    spy_ret = 0.10
    
    if tnx_h is not None and not tnx_h.empty:
        rf_rate = tnx_h['Close'].iloc[-1] / 100
        st.metric("美債 10Y (無風險利率)", f"{rf_rate*100:.2f}%")
    else:
        st.metric("美債 10Y (無風險利率)", "4.00% (系統預設)")
        
    if spy_h is not None and not spy_h.empty:
        spy_ret = spy_h['Close'].pct_change(252).iloc[-1]
        st.metric("美股 S&P 500 (Rm)", f"{spy_ret*100:.2f}%")
        fig_side = px.line(spy_h.tail(45), y='Close', template="plotly_dark").update_traces(line_color='#60a5fa')
        fig_side.update_layout(height=130, margin=dict(l=0,r=0,t=0,b=0), xaxis_visible=False, yaxis_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_side, use_container_width=True, config={'displayModeBar': False})
        
    if tw_h is not None and not tw_h.empty:
        tw_ret = tw_h['Close'].pct_change(252).iloc[-1]
        st.metric("台股 0050 報酬率", f"{tw_ret*100:.2f}%")
        
    st.info("💡 系統狀態：單機獨立運作模式")

tab1, tab2, tab3, tab4 = st.tabs(["🔍 AI 市場診斷", "🛡️ 投資組合深度績效與診斷", "⏳ 歷史回測與匯出", "📖 模型說明"])

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
                
                fig.add_trace(go.Candlestick(x=str_dates, open=plot_data['Open'], high=plot_data['High'], low=plot_data['Low'], close=plot_data['Close'], name='歷史走勢', increasing_line_color='#4ade80', decreasing_line_color='#f87171'), row=1, col=1)
                fig.add_trace(go.Scatter(x=str_dates, y=plot_data['MA10'], line=dict(color='#81d4fa', width=1), name='10MA (短)'), row=1, col=1)
                fig.add_trace(go.Scatter(x=str_dates, y=plot_data['MA50'], line=dict(color='#fbbf24', width=1.2), name='50MA (中)'), row=1, col=1)
                fig.add_trace(go.Scatter(x=str_dates, y=plot_data['MA200'], line=dict(color='#94a3b8', width=2), name='200MA (生命線)'), row=1, col=1)
                
                macd_colors = ['#4ade80' if val >= 0 else '#f87171' for val in plot_data['MACD_Hist']]
                fig.add_trace(go.Bar(x=str_dates, y=plot_data['MACD_Hist'], marker_color=macd_colors, name='MACD 柱狀圖'), row=2, col=1)
                fig.add_trace(go.Scatter(x=str_dates, y=plot_data['MACD'], line=dict(color='#60a5fa', width=1.5), name='MACD (12,26)'), row=2, col=1)
                fig.add_trace(go.Scatter(x=str_dates, y=plot_data['Signal_Line'], line=dict(color='#fbbf24', width=1.5), name='Signal (9)'), row=2, col=1)
                
                fig.update_layout(
                    template="plotly_dark", 
                    height=700, 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=0, r=0, t=30, b=0)
                )

                fig.update_xaxes(type='category', nticks=10, rangeslider_visible=False)
                
                st.plotly_chart(fig, use_container_width=True)
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("即時股價", f"${cur_p:.2f}")
                c2.metric("RSI 指標 (14D)", f"{hist['RSI'].iloc[-1]:.1f}")
                c3.metric("本益比 (PE)", f"{info.get('forwardPE', 'N/A') if isinstance(info, dict) else 'N/A'}")
                c4.metric("市場風險 Beta", f"{info.get('beta', 'N/A') if isinstance(info, dict) else 'N/A'}")
            else:
                st.error("⚠️ 無法取得該標的數據，請確認代號是否正確。")

# --- Tab 2: 完整投資組合診斷 ---
if "portfolio_df" not in st.session_state:
    st.session_state.portfolio_df = pd.DataFrame([
        {"代號": "CASH",  "持有股數": 50000, "平均成本": 1.00}, # 現金預設為台幣 50000
        {"代號": "AAOI",  "持有股數": 2,  "平均成本": 203.00},
        {"代號": "ARKF",  "持有股數": 10, "平均成本": 48.30},
        {"代號": "BRK.B", "持有股數": 6,  "平均成本": 474.95},
        {"代號": "GOOGL", "持有股數": 3,  "平均成本": 329.00},
        {"代號": "JPM",   "持有股數": 3,  "平均成本": 308.00},
        {"代號": "NMR",   "持有股數": 30, "平均成本": 9.49},
        {"代號": "NVDA",  "持有股數": 5,  "平均成本": 182.94},
        {"代號": "PLTR",  "持有股數": 6,  "平均成本": 156.74},
        {"代號": "VOO",   "持有股數": 9,  "平均成本": 632.15}
    ])

with tab2:
    def color_pnl_cells(val):
        color = '#4ade80' if val >= 0 else '#f87171'
        return f'color: {color} !important; font-weight: bold;'

    st.markdown("### 💰 輸入持倉資訊 (CASH 請填入台幣總額)")
    edited = st.data_editor(st.session_state.portfolio_df, num_rows="dynamic", use_container_width=True)
    
    if st.button("🚀 執行 AI 量化診斷"):
        st.session_state.portfolio_df = edited

        with st.spinner('AI 正在運算數據中...'):
            assets_data, hist_dict = [], {}
            cash_val_usd = 0
            stock_cost = 0
            stock_val = 0
            tech_count = 0
            tech_tickers = ['QQQ', 'QQQM', 'NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX', 'PLTR', 'ARKF', 'SMH', 'SOXX', 'AAOI', '2330.TW', '2330']
            failed_tickers = [] 
            
            # --- 抓取 USD/TWD 即時匯率 ---
            usd_twd_rate = 32.2 # 預設值防呆
            try:
                rate_data = yf.Ticker("TWD=X").history(period="1d")
                if not rate_data.empty:
                    usd_twd_rate = rate_data['Close'].iloc[-1]
            except:
                pass
            # -----------------------------

            for _, row in edited.iterrows():
                raw_ticker = str(row["代號"]).strip()
                if not raw_ticker: continue
                
                # --- 新增：現金 (台幣轉美金) 邏輯 ---
                if raw_ticker.upper() == "CASH":
                    twd_amt = float(row["持有股數"])
                    usd_amt = twd_amt / usd_twd_rate # 將台幣轉換為美金
                    assets_data.append({"股票": "CASH (TWD)", "即時現價": 1/usd_twd_rate, "總成本": usd_amt, "目前市值": usd_amt, "未實現損益": 0.0, "報酬率(%)": 0.0, "Beta": 0.0, "RSI": 0.0})
                    cash_val_usd += usd_amt
                    hist_dict["CASH (TWD)"] = None 
                    continue
                # ------------------------------------
                
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
            
            # --- 為現金補齊歷史時間序列，避免矩陣計算報錯 ---
            valid_idx = next((v.index for v in hist_dict.values() if v is not None), None)
            if "CASH (TWD)" in hist_dict:
                hist_dict["CASH (TWD)"] = pd.Series(1.0, index=valid_idx) if valid_idx is not None else pd.Series([1.0, 1.0])
            # ------------------------------------------------

            if failed_tickers:
                st.warning(f"⚠️ 以下標的暫時無法載入：{', '.join(failed_tickers)}")

            if assets_data:
                res_df = pd.DataFrame(assets_data)
                st.session_state['res_df'] = res_df 
                st.session_state['hist_dict'] = hist_dict
                
                # --- 分離計算：股票成本 vs 組合總市值 ---
                total_val = stock_val + cash_val_usd
                stock_pnl = stock_val - stock_cost
                stock_pnl_pct = (stock_pnl / stock_cost)*100 if stock_cost > 0 else 0
                pnl_color = "#4ade80" if stock_pnl >= 0 else "#f87171"
                
                st.info(f"💱 目前即時匯率：1 USD = {usd_twd_rate:.2f} TWD (台幣現金已自動轉換為美元計價)")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("股票投入總成本 (USD)", f"${stock_cost:,.2f}")
                m2.metric("組合總市值含現金 (USD)", f"${total_val:,.2f}")
                m3.markdown(f"""<div style="padding: 15px; border-radius: 10px; border: 1px solid #334155; background-color: #1e293b; text-align: center;"><div style="color: #94a3b8; font-size: 14px; margin-bottom: 5px;">未實現總損益</div><div style="color: {pnl_color}; font-size: 32px; font-weight: bold;">${stock_pnl:+,.2f}</div></div>""", unsafe_allow_html=True)
                m4.markdown(f"""<div style="padding: 15px; border-radius: 10px; border: 1px solid #334155; background-color: #1e293b; text-align: center;"><div style="color: #94a3b8; font-size: 14px; margin-bottom: 5px;">股票投資報酬率</div><div style="color: {pnl_color}; font-size: 32px; font-weight: bold;">{stock_pnl_pct:+.2f}%</div></div>""", unsafe_allow_html=True)
                
                styled_table = res_df[['股票', '即時現價', '總成本', '目前市值', '未實現損益', '報酬率(%)']].style.format({'即時現價': '${:.4f}', '總成本': '${:,.2f}', '目前市值': '${:,.2f}', '未實現損益': '${:+,.2f}', '報酬率(%)': '{:+.2f}%'}).map(color_pnl_cells, subset=['未實現損益', '報酬率(%)']).set_table_styles([{'selector': 'table', 'props': [('width', '100%'), ('background-color', '#1e293b'), ('border-radius', '10px')]}, {'selector': 'th', 'props': [('background-color', '#0f172a'), ('color', '#94a3b8')]}, {'selector': 'td', 'props': [('color', '#f1f5f9')]}]).hide(axis="index").to_html()
                st.markdown(styled_table, unsafe_allow_html=True)

                hist_df = pd.DataFrame(hist_dict).dropna() 
                ret_df = hist_df.pct_change().dropna() 
                corr_matrix = ret_df.corr() 

                weights = []
                weighted_beta, portfolio_return = 0, 0
                for idx, row in res_df.iterrows():
                    w = row["目前市值"] / total_val if total_val > 0 else 0
                    weights.append(w)
                    res_df.at[idx, '權重'] = w
                    weighted_beta += row["Beta"] * w
                    portfolio_return += hist_df[row['股票']].pct_change(252).iloc[-1] * w

                port_daily_ret = ret_df.dot(weights)
                port_mdd = calculate_mdd((1 + port_daily_ret).cumprod())
                jensen_alpha = portfolio_return - (rf_rate + weighted_beta * (spy_ret - rf_rate))

                if spy_h is not None and not spy_h.empty:
                    spy_daily_returns = spy_h['Close'].pct_change().dropna()
                    aligned = pd.DataFrame({'Port': port_daily_ret, 'SPY': spy_daily_returns}).dropna()
                    if not aligned.empty:
                        t_stat, p_value = stats.ttest_ind(aligned['Port'], aligned['SPY'], equal_var=False)
                        ttest_result = f"P-value 為 {p_value:.4f} < 0.05。<br><span style='color:#4ade80;'>✅ 拒絕虛無假說，此組合與大盤有**統計顯著差異**。</span>" if p_value < 0.05 else f"P-value 為 {p_value:.4f} >= 0.05。<br><span style='color:#fbbf24;'>⚠️ 無法拒絕虛無假說，超額表現可能為**隨機機率**。</span>"
                    else:
                        t_stat, ttest_result = 0, "無足夠數據"
                else: t_stat, ttest_result = 0, "無大盤數據"

                st.divider()

                top_stock = res_df.sort_values('權重', ascending=False)['股票'].iloc[0]
                tech_ratio = tech_count / len(res_df) if len(res_df) > 0 else 0
                has_redundancy = "QQQ" in res_df['股票'].values and "QQQM" in res_df['股票'].values

                if tech_ratio > 0.8 and weighted_beta > 1.15:
                    eval_title, eval_color = "高風險：極端成長型集中", "#f87171"
                    eval_content = f"組合極度向科技股傾斜。受 **{top_stock}** 等標的影響，在極端市況下抗跌能力極弱。"
                    prescription = f"建議減碼部分高估值科技股，並將資金轉入避險資產，以降低目前達 {abs(port_mdd*100):.1f}% 的最大回撤風險。"
                elif has_redundancy:
                    eval_title, eval_color = "結構冗餘：標的重疊風險", "#fbbf24"
                    eval_content = "偵測到同時持有 QQQ 與 QQQM，兩者追蹤同一指數，屬於無效的分散風險配置。"
                    prescription = "建議將 QQQ 完全轉換為 QQQM，釋出的資金佈局非科技板塊來達到防禦。"
                elif 0.9 <= weighted_beta <= 1.25 and tech_ratio < 0.6:
                    eval_title, eval_color = "精英級：均衡核心配置", "#60a5fa"
                    eval_content = f"配置展現了極高的專業度。以 **{top_stock}** 為核心定海神針，名副其實的『進可攻、退可守』。"
                    prescription = f"目前結構極佳（Alpha: {jensen_alpha*100:+.2f}%）。建議維持現有比例，逢低加碼。"
                else:
                    eval_title, eval_color = "穩健/防禦型配置", "#4ade80"
                    eval_content = f"組合抗波動能力強。以 **{top_stock}** 等穩健標的為主軸，能提供極佳保護力。"
                    prescription = "防禦力極強但可能錯失牛市超額利潤。建議提撥部分資金佈局高動能衛星標的。"

                st.markdown(f"""
                    <div class="report-card" style="border-left: 10px solid {eval_color}; background-color: #1e293b;">
                        <h2 style="color: {eval_color}; margin:0;">AI 綜合診斷：{eval_title}</h2>
                        <p style="margin-top:20px; font-size:18px; line-height:1.7; color: white !important;">
                            <b>分析師實話：</b><br>{eval_content}<br><br>
                            <b>組合加權 Beta：</b>{weighted_beta:.2f} (市場敏感度)<br>
                            <b>Jensen's Alpha：</b><span style='color: {"#4ade80" if jensen_alpha > 0 else "#f87171"}; font-weight: bold;'>{jensen_alpha*100:+.2f}%</span><br>
                            <b>歷史最大回撤 (MDD)：</b><span style='color: #f87171; font-weight: bold;'>{port_mdd*100:.2f}%</span><br>
                        </p>
                        <div class="ttest-box">
                            <strong style="color:#a78bfa; font-size:18px;">🔬 統計顯著性檢定 (Two-Sample T-Test)：</strong><br>
                            <span style="color:#f1f5f9;"><b>T-Statistic:</b> {t_stat:.4f} <br><b>結論：</b> {ttest_result}</span>
                        </div>
                        <div class="prescription-box">
                            <strong style="color:#60a5fa; font-size:18px;">💡 AI 智能處方 (Rebalancing Suggestion)：</strong><br>
                            <span style="color:#f1f5f9;">{prescription}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                c_pie, c_heat = st.columns(2)
                c_pie.plotly_chart(px.pie(res_df, values='權重', names='股票', hole=0.4, title="真實市值權重配比 (Weight)", template="plotly_dark").update_layout(font=dict(color="white")), use_container_width=True)
                fig_heat = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', origin='lower', title="資產相關性熱力圖 (Correlation Matrix)")
                fig_heat.update_layout(template="plotly_dark", font=dict(color="white"))
                c_heat.plotly_chart(fig_heat, use_container_width=True)

# --- Tab 3: 回測與報表 ---
with tab3:
    st.markdown("### 📊 歷史回測與投資組合表現")
    if 'hist_dict' in st.session_state and 'res_df' in st.session_state:
        res_df = st.session_state['res_df']
        hist_dict = st.session_state['hist_dict']
        
        total_val = res_df['目前市值'].sum()
        weights = (res_df['目前市值'] / total_val).values
        ret_df = pd.DataFrame(hist_dict).pct_change().dropna()
        
        port_cum_ret = (1 + ret_df.dot(weights)).cumprod()
        
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=port_cum_ret.index, y=port_cum_ret, name='你的投資組合', line=dict(color='#4ade80', width=2.5)))
        
        if spy_h is not None and not spy_h.empty:
            spy_cum_ret = (1 + spy_h['Close'].pct_change().dropna()).cumprod()
            fig_bt.add_trace(go.Scatter(x=spy_cum_ret.index, y=spy_cum_ret, name='大盤 (S&P 500)', line=dict(color='#94a3b8', width=1.5, dash='dash')))
            
        fig_bt.update_layout(template="plotly_dark", title="近兩年資產資金曲線 (Equity Curve)", height=450, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_bt, use_container_width=True)
        
        st.markdown("### 📥 報表匯出")
        st.download_button(label="📊 點擊下載量化分析報告 (CSV)", data=convert_df_to_csv(res_df), file_name=f"AlphaCheck_Report_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
    else: st.warning("請先在「投資組合深度分析」頁籤點擊執行運算。")

# --- Tab 4: 說明 ---
with tab4:
    st.header("📖 模型理論與實務應用")
    st.markdown("""
    1. **Mark-to-Market**：按市值計價。
    2. **CAPM & Jensen's Alpha**：量化超額報酬。
    3. **T-test**：檢定超額報酬是否為隨機機率。
    4. **MDD**：最大回撤。
    5. **MACD 動能**：結合快慢線與柱狀圖，精準捕捉趨勢反轉點。
    6. **台股辨識**：輸入 `2330` 自動辨識為 `2330.TW`。
    """)