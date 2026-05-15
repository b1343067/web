import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import scipy.stats as stats # 執行 T-test 必備

# ==========================================
# 1. 核心計算引擎
# ==========================================

@st.cache_data(ttl=3600)
def fetch_financial_data(ticker_name):
    try:
        # 自動修正代號格式 (如 BRK.B -> BRK-B)
        clean_ticker = ticker_name.upper().replace("/", "-").replace(".", "-").strip()
        ticker_obj = yf.Ticker(clean_ticker)
        history = ticker_obj.history(period="2y")
        info = ticker_obj.info if hasattr(ticker_obj, 'info') else {}
        if history.empty: return None, None, f"找不到代號 {clean_ticker}"
        return history, info, None
    except Exception as e:
        return None, None, str(e)

def calculate_indicators(df):
    """計算 RSI 與三重專業均線"""
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
    """AI 趨勢擬合：修正 NVDA 斷層，實作價格錨定與不確定性圓錐"""
    df_p = df.tail(90).reset_index()
    df_p['Date_n'] = pd.to_datetime(df_p['Date']).apply(lambda x: x.toordinal())
    X = df_p[['Date_n']].values
    y = df_p['Close'].values
    
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    
    last_d_n = df_p['Date_n'].max()
    last_d = df_p['Date'].max()
    last_close = df['Close'].iloc[-1]
    
    # 未來預測
    future_n = np.array([last_d_n + i for i in range(1, days + 1)]).reshape(-1, 1)
    raw_preds = model.predict(poly.transform(future_n))
    
    # 價格錨定 (Price Anchoring)：確保預測線從今天收盤價出發
    pred_today = model.predict(poly.transform(np.array([[last_d_n]])))[0]
    shift = last_close - pred_today
    shifted_preds = raw_preds + shift
    
    # 不確定性圓錐 (Cone of Uncertainty)
    base_std = df['Close'].tail(30).std()
    intervals = [base_std * (0.2 + (i * 0.2)) for i in range(len(shifted_preds))]
    
    # 結合今天與未來的序列
    future_d = [last_d] + [last_d + timedelta(days=i) for i in range(1, days + 1)]
    final_preds = [last_close] + shifted_preds.tolist()
    final_intervals = [0.0] + intervals
    
    return future_d, final_preds, final_intervals

def calculate_mdd(cumulative_returns):
    """計算最大回撤"""
    roll_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - roll_max) / roll_max
    return drawdown.min()

# ==========================================
# 2. UI 視覺與 CSS 美化
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
    .ttest-box { padding: 15px; border-radius: 10px; background-color: rgba(167, 139, 250, 0.1); border-left: 5px solid #a78bfa; margin-top: 15px; }
    .prescription-box { padding: 15px; border-radius: 10px; background-color: rgba(96, 165, 250, 0.1); border-left: 5px solid #60a5fa; margin-top: 15px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ AlphaCheck Elite: 專業投資決策終端")

# --- 側邊欄 ---
with st.sidebar:
    st.markdown("### 🌍 市場監控中心")
    tnx_h, _, _ = fetch_financial_data("^TNX")
    spy_h, _, _ = fetch_financial_data("SPY")
    rf_rate, spy_ret = 0.04, 0.10
    if tnx_h is not None:
        rf_rate = tnx_h['Close'].iloc[-1] / 100
        st.metric("美債 10Y (Rf)", f"{rf_rate*100:.2f}%")
    if spy_h is not None:
        spy_ret = spy_h['Close'].pct_change(252).iloc[-1]
        st.metric("S&P 500 (Rm)", f"{spy_ret*100:.2f}%")
        st.plotly_chart(px.line(spy_h.tail(45), y='Close', template="plotly_dark").update_traces(line_color='#60a5fa').update_layout(height=130, margin=dict(l=0,r=0,t=0,b=0), xaxis_visible=False, yaxis_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'), use_container_width=True)

tab1, tab2, tab3 = st.tabs(["🔍 AI 市場診斷", "🛡️ 投資組合量化分析報告", "📖 模型說明"])

# --- Tab 1: AI 市場診斷 ---
with tab1:
    raw_ticker = st.text_input("輸入美股代號", "VOO")
    if raw_ticker:
        target = raw_ticker.upper().replace("/", "-").strip()
        hist, info, err = fetch_financial_data(target)
        if not err:
            hist = calculate_indicators(hist)
            f_dates, f_preds, f_intervals = get_ai_prediction_model(hist)
            cur_p, target_p = hist['Close'].iloc[-1], f_preds[-1]
            ret = ((target_p - cur_p) / cur_p) * 100
            
            bg, border, txt = ("rgba(20, 83, 45, 0.4)", "#4ade80", "多頭趨勢") if ret > 2 else ("rgba(127, 29, 29, 0.4)", "#f87171", "空頭預警") if ret < -2 else ("rgba(30, 41, 59, 0.6)", "#94a3b8", "中立觀望")
            st.markdown(f"<div class='report-card' style='background-color: {bg}; border-color: {border};'><h3>🤖 AI 評級：{txt}</h3><p>預期收益：<b>{ret:+.2f}%</b></p></div>", unsafe_allow_html=True)
            
            fig = go.Figure(data=[go.Candlestick(x=hist.tail(120).index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], increasing_line_color='#4ade80', decreasing_line_color='#f87171', name='K線')])
            fig.add_trace(go.Scatter(x=hist.tail(120).index, y=hist['MA50'].tail(120), line=dict(color='#fbbf24', width=1), name='50MA'))
            fig.add_trace(go.Scatter(x=hist.tail(120).index, y=hist['MA200'].tail(120), line=dict(color='#94a3b8', width=2), name='200MA'))
            fig.add_trace(go.Scatter(x=f_dates + f_dates[::-1], y=[f_preds[i]+f_intervals[i] for i in range(len(f_preds))]+[f_preds[i]-f_intervals[i] for i in range(len(f_preds))][::-1], fill='toself', fillcolor='rgba(96, 165, 250, 0.1)', line_color='rgba(0,0,0,0)', name='AI 波動預期'))
            fig.add_trace(go.Scatter(x=f_dates, y=f_preds, line=dict(color='white', dash='dash'), name='AI 路徑'))
            fig.update_layout(template="plotly_dark", height=550, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: 投資組合量化分析 (含 T-test) ---
with tab2:
    def color_pnl(val):
        color = '#4ade80' if val >= 0 else '#f87171'
        return f'color: {color} !important; font-weight: bold;'

    st.markdown("### 💰 持倉資訊")
    p_df = pd.DataFrame([
        {"代號": "AAOI", "持有股數": 2, "平均成本": 203.0},
        {"代號": "NVDA", "持有股數": 5, "平均成本": 182.9},
        {"代號": "VOO", "持有股數": 9, "平均成本": 632.1}
    ])
    edited = st.data_editor(p_df, num_rows="dynamic", use_container_width=True)
    
    if st.button("🚀 執行量化診斷分析"):
        assets_data, hist_dict = [], {}
        total_cost, total_val, tech_count = 0, 0, 0
        tech_list = ['AAOI', 'NVDA', 'PLTR', 'GOOGL', 'ARKF', 'QQQ', 'QQQM', 'TSLA']

        for _, row in edited.iterrows():
            ticker = row["代號"].upper()
            h, i, _ = fetch_financial_data(ticker)
            if h is not None:
                h = calculate_indicators(h)
                cur_p = h['Close'].iloc[-1]
                val = row["持有股數"] * cur_p
                cost = row["持有股數"] * row["平均成本"]
                pnl = val - cost
                total_cost += cost
                total_val += val
                if ticker in tech_list or ticker.replace(".","-") in tech_list: tech_count += 1
                assets_data.append({"股票": ticker, "現價": cur_p, "成本": cost, "市值": val, "損益": pnl, "報酬率": (pnl/cost)*100, "Beta": i.get('beta', 1.0), "RSI": h['RSI'].iloc[-1]})
                hist_dict[ticker] = h['Close']

        res_df = pd.DataFrame(assets_data)
        pnl_tot = total_val - total_cost
        pnl_color = "#4ade80" if pnl_tot >= 0 else "#f87171"

        # 儀表板
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("總成本", f"${total_cost:,.2f}")
        m2.metric("總市值", f"${total_val:,.2f}")
        m3.markdown(f"<div style='padding:15px; border-radius:10px; border:1px solid #334155; background-color:#1e293b; text-align:center;'><div style='color:#94a3b8; font-size:14px;'>未實現總損益</div><div style='color:{pnl_color}; font-size:28px; font-weight:bold;'>${pnl_tot:+,.2f}</div></div>", unsafe_allow_html=True)
        m4.markdown(f"<div style='padding:15px; border-radius:10px; border:1px solid #334155; background-color:#1e293b; text-align:center;'><div style='color:#94a3b8; font-size:14px;'>總報酬率</div><div style='color:{pnl_color}; font-size:28px; font-weight:bold;'>{(pnl_tot/total_cost)*100:+.2f}%</div></div>", unsafe_allow_html=True)
        
        st.markdown(res_df[['股票', '現價', '成本', '市值', '損益', '報酬率']].style.format({'現價': '${:.2f}', '成本': '${:,.2f}', '市值': '${:,.2f}', '損益': '${:+,.2f}', '報酬率': '{:+.2f}%'}).map(color_pnl, subset=['損益', '報酬率']).to_html(), unsafe_allow_html=True)

        # 統計與 AI 診斷
        hist_df = pd.DataFrame(hist_dict).dropna()
        ret_df = hist_df.pct_change().dropna()
        weights = [row['市值']/total_val for _, row in res_df.iterrows()]
        port_ret_series = ret_df.dot(weights)
        mdd = calculate_mdd((1 + port_ret_series).cumprod())
        w_beta = sum(row['Beta'] * (row['市值']/total_val) for _, row in res_df.iterrows())
        
        # 🌟 T-test 實作
        spy_daily = spy_h['Close'].pct_change().dropna()
        aligned = pd.DataFrame({'Port': port_ret_series, 'Spy': spy_daily}).dropna()
        t_stat, p_val = stats.ttest_ind(aligned['Port'], aligned['Spy'], equal_var=False)

        st.divider()
        eval_color = "#60a5fa" if p_val < 0.05 else "#fbbf24"
        st.markdown(f"""
            <div class="report-card" style="border-left: 10px solid {eval_color}; background-color: #1e293b;">
                <h2 style="color: {eval_color}; margin:0;">AI 診斷：{('精英級' if p_val < 0.05 else '潛力型')}配置</h2>
                <p>組合加權 Beta: {w_beta:.2f} | 歷史最大回撤: <span style='color:#f87171;'>{mdd*100:.2f}%</span></p>
                <div class="ttest-box">
                    <strong>🔬 統計檢定 (T-test)：</strong><br>
                    與標普500相比，P-value 為 <b>{p_val:.4f}</b>。<br>
                    {'✅ 績效具備顯著性，你的選股實力已獲得數學證明！' if p_val < 0.05 else '⚠️ 績效尚未達到統計顯著，目前的領先可能是隨機波動所致。'}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        c_pie, c_heat = st.columns(2)
        c_pie.plotly_chart(px.pie(res_df, values='市值', names='股票', hole=0.4, title="資產配比", template="plotly_dark"), use_container_width=True)
        c_heat.plotly_chart(px.imshow(ret_df.corr(), text_auto=".2f", color_continuous_scale='RdBu_r', title="相關性熱力圖", template="plotly_dark"), use_container_width=True)

with tab3:
    st.header("📖 模型理論")
    st.markdown("""
    - **AI 價格錨定**：消除預測起始點與真實收盤價的視覺落差。
    - **T-test**：檢定投資組合與大盤報酬率之差異是否具有顯著性 ($P < 0.05$)。
    - **MDD**：衡量資產回撤風險。
    """)
