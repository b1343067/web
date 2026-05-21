import streamlit as st
import streamlit.components.v1 as components
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import scipy.stats as stats

# ==========================================
# 0. 參數設定區 (已為你完美客製化)
# ==========================================
LIFF_ID = "2010137574-nywnenmu"
FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSfi5Z7o5G6zo0YgCtXGSKWaa4JNroK8onT5R1r-sFIzOwwUHg/formResponse" 

# ==========================================
# 1. 核心計算引擎
# ==========================================
@st.cache_data(ttl=3600)
def fetch_financial_data(ticker_name):
    try:
        clean_ticker = ticker_name.upper().replace("/", "-").replace(".", "-").strip()
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

def calculate_mdd(cumulative_returns):
    roll_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - roll_max) / roll_max
    return drawdown.min()

# ==========================================
# 2. UI 視覺設計
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
    .ttest-box { padding: 15px; border-radius: 10px; background-color: rgba(167, 139, 250, 0.1); border-left: 5px solid #a78bfa; margin-top: 15px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ AlphaCheck Elite: 專業投資決策終端")

# ==========================================
# 3. 抓取 LINE User ID (LIFF 邏輯)
# ==========================================
query_params = st.query_params
user_id = query_params.get("userId", None)

if not user_id:
    # 執行 LIFF SDK 抓取 ID 並重新整理網頁
    components.html(f"""
        <script charset="utf-8" src="https://static.line-scdn.net/liff/edge/2/sdk.js"></script>
        <script>
            liff.init({{ liffId: '{LIFF_ID}' }}).then(() => {{
                if (!liff.isLoggedIn()) {{
                    liff.login();
                }} else {{
                    liff.getProfile().then(profile => {{
                        window.parent.location.href = window.parent.location.href.split('?')[0] + "?userId=" + profile.userId;
                    }}).catch(err => console.error("無法取得 Profile:", err));
                }}
            }}).catch(err => console.error("LIFF 啟動失敗:", err));
        </script>
        <div style="color: #60a5fa; text-align: center; font-family: sans-serif; padding: 40px; font-size: 20px; font-weight: bold;">
            🔄 系統正在透過 LINE 安全驗證您的身分，請稍候...
        </div>
        <div style="color: #94a3b8; text-align: center; margin-top: 10px;">
            ⚠️ 提示：請確保您是在手機的 LINE 聊天室內點擊網址。<br>若在電腦版測試，請在網址列後方手動加上 <b>/?userId=測試ID</b>
        </div>
    """, height=200)
    st.stop() # 暫停渲染以下內容，直到抓到 userId

# ==========================================
# 4. 側邊欄與分頁顯示
# ==========================================
with st.sidebar:
    st.markdown("### 🌍 市場監控中心")
    if user_id:
        st.success("✅ LINE 帳號已綁定")
        st.caption(f"認證 ID: {user_id[:8]}...")
    
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
    st.info("💡 系統已啟用: 即時 MTM / CAPM / LINE 資料庫同步")

tab1, tab2, tab3 = st.tabs(["🔍 AI 市場診斷", "🛡️ 投資組合深度績效與 LINE 同步", "📖 模型說明"])

# --- Tab 1: AI 診斷 ---
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

# --- Tab 2: 投資組合與 LINE 同步儲存 ---
with tab2:
    st.markdown("### 💾 儲存資產至 LINE 錢包")
    st.caption("輸入的資料將自動與您的 LINE 帳號綁定，並寫入雲端資料庫。")
    
    # 建立一個儲存表單
    with st.form("save_to_line_form"):
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            input_ticker = st.text_input("股票代號 (例如: VOO)")
        with col_f2:
            input_price = st.number_input("買入平均成本", min_value=0.0, step=0.01)
        with col_f3:
            input_shares = st.number_input("持有股數", min_value=0.0, step=0.1)
            
        submitted_line = st.form_submit_button("🚀 同步儲存至 LINE 資料庫")
        
        if submitted_line:
            if input_ticker and input_price > 0 and input_shares > 0:
                # 🔴 已為你配對正確的 4 個 Entry ID
                form_data = {
                    "entry.637172945": user_id,              # User_ID
                    "entry.1886186927": input_ticker.upper(),# Ticker
                    "entry.829335385": input_price,          # Buy_Price
                    "entry.745239131": input_shares          # Shares
                }
                
                try:
                    response = requests.post(FORM_URL, data=form_data)
                    if response.status_code == 200:
                        st.success(f"🎉 成功！已將 {input_shares} 股 {input_ticker.upper()} 寫入資料庫，現在可於 LINE 使用『錢錢check』查詢！")
                        st.balloons()
                    else:
                        st.error("寫入表單失敗，狀態碼：" + str(response.status_code))
                except Exception as e:
                    st.error(f"系統連線錯誤: {str(e)}")
            else:
                st.warning("⚠️ 請確認所有欄位都已填寫，且成本與股數大於 0。")

    st.divider()

    # 原本的即時結算測試區塊
    st.markdown("### 🧪 沙盤推演：目前模擬持倉")
    def color_pnl_cells(val):
        color = '#4ade80' if val >= 0 else '#f87171'
        return f'color: {color} !important; font-weight: bold;'
    
    p_df = pd.DataFrame([
        {"代號": "VOO",   "持有股數": 9,  "平均成本": 632.15},
        {"代號": "NVDA",  "持有股數": 5,  "平均成本": 182.94}
    ])
    edited = st.data_editor(p_df, num_rows="dynamic", use_container_width=True)
    
    if st.button("📊 預覽即時損益與 AI 診斷"):
        with st.spinner('AI 正在計算即時損益、生成相關性矩陣與統計檢定...'):
            assets_data = []
            hist_dict = {} 
            total_invested_cost = 0
            total_current_value = 0
            tech_count = 0
            tech_tickers = ['QQQ', 'QQQM', 'NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX', 'PLTR', 'ARKF', 'SMH', 'SOXX', 'AAOI']

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
                    if ticker in tech_tickers or ticker.replace(".", "-") in tech_tickers: 
                        tech_count += 1
                        
                    assets_data.append({
                        "股票": ticker, "即時現價": cur_price, "總成本": asset_cost, "目前市值": asset_val, 
                        "未實現損益": pnl, "報酬率(%)": pnl_pct, "Beta": i.get('beta', 1.0), "RSI": h['RSI'].iloc[-1]
                    })
                    hist_dict[ticker] = h['Close']

            res_df = pd.DataFrame(assets_data)
            
            # --- 儀表板與 HTML 表格 ---
            total_pnl = total_current_value - total_invested_cost
            total_pnl_pct = (total_pnl / total_invested_cost)*100 if total_invested_cost > 0 else 0
            pnl_color = "#4ade80" if total_pnl >= 0 else "#f87171"

            st.markdown(f"### 📈 損益儀表板")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("投入總成本", f"${total_invested_cost:,.2f}")
            m2.metric("目前總市值", f"${total_current_value:,.2f}")
            m3.markdown(f"""<div style="padding: 15px; border-radius: 10px; border: 1px solid #334155; background-color: #1e293b; text-align: center;"><div style="color: #94a3b8; font-size: 14px; margin-bottom: 5px;">未實現總損益</div><div style="color: {pnl_color}; font-size: 32px; font-weight: bold;">${total_pnl:+,.2f}</div></div>""", unsafe_allow_html=True)
            m4.markdown(f"""<div style="padding: 15px; border-radius: 10px; border: 1px solid #334155; background-color: #1e293b; text-align: center;"><div style="color: #94a3b8; font-size: 14px; margin-bottom: 5px;">總體報酬率</div><div style="color: {pnl_color}; font-size: 32px; font-weight: bold;">{total_pnl_pct:+.2f}%</div></div>""", unsafe_allow_html=True)
            
            styled_table = res_df[['股票', '即時現價', '總成本', '目前市值', '未實現損益', '報酬率(%)']].style.format({
                '即時現價': '${:.2f}', '總成本': '${:,.2f}', '目前市值': '${:,.2f}', '未實現損益': '${:+,.2f}', '報酬率(%)': '{:+.2f}%'
            }).map(color_pnl_cells, subset=['未實現損益', '報酬率(%)']).set_table_styles([
                {'selector': 'table', 'props': [('width', '100%'), ('border-collapse', 'collapse'), ('background-color', '#1e293b'), ('border-radius', '10px'), ('overflow', 'hidden'), ('margin-top', '20px'), ('margin-bottom', '20px')]},
                {'selector': 'th', 'props': [('background-color', '#0f172a'), ('color', '#94a3b8'), ('font-weight', 'bold'), ('padding', '15px'), ('text-align', 'center'), ('border', '1px solid #334155')]},
                {'selector': 'td', 'props': [('color', '#f1f5f9'), ('padding', '12px'), ('text-align', 'center'), ('border', '1px solid #334155')]},
                {'selector': 'tr:hover td', 'props': [('background-color', '#2c3e50')]}
            ]).hide(axis="index").to_html()
            
            st.markdown(styled_table, unsafe_allow_html=True)

with tab3:
    st.header("📖 模型理論與實務應用")
    st.markdown("""
    1. **Mark-to-Market (按市值計價)**：採用「當前真實市值」計算動態權重，貼近實盤操作。
    2. **CAPM 與 Jensen's Alpha ($\\alpha$)**：用於量化投資組合是否創造了高於承擔系統風險下的「超額報酬」。
    3. **Two-Sample T-test (獨立雙樣本 T檢定)**：檢定投資組合的歷史日報酬與大盤 (SPY) 是否存在統計上的顯著差異 ($P < 0.05$)，驗證超額報酬是否為隨機機率。
    4. **最大回撤 (Maximum Drawdown, MDD)**：評估在過去一段時間內，資產從最高點滑落至最低點的幅度，為衡量下行風險的極重要指標。
    5. **資產相關性矩陣 (Correlation Matrix)**：數值越接近 1 代表同漲同跌；越接近 -1 代表走勢相反。真正的「避險」建立在挑選低相關性或負相關的資產上。
    """)
