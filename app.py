import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import urllib.request
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- 1. ページ設定 ---
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    div[data-testid="stMetric"] {
        background-color: #ffffff; border: 1px solid #eee;
        padding: 15px; border-radius: 10px;
    }
    .stButton>button { width: 100%; font-weight: bold; background-color: #ff4b4b; color: white; border-radius: 8px; height: 3em; }
    </style>
    """, unsafe_allow_html=True)

st.title("Stock Scanner")

# --- 2. データ取得関数 ---
@st.cache_data(ttl=86400)
def get_jpx_full_data():
    cache_path = "jpx_data.parquet"
    if os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except:
            pass
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            f = io.BytesIO(response.read())
            df = pd.read_excel(f)
        
        # JPXの新フォーマット（列名変更）に完全対応する安全ロジック
        available_cols = [c for c in ['コード', '銘柄名', '市場・商品区分', '市場・商品', '33業種区分'] if c in df.columns]
        df = df[available_cols].copy()
        if '市場・商品' in df.columns and '市場・商品区分' not in df.columns:
            df = df.rename(columns={'市場・商品': '市場・商品区分'})
            
        df['コード'] = df['コード'].astype(str).str.strip()
        df.to_parquet(cache_path)
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_market_indices():
    indices = {"USD/JPY": "JPY=X", "日経平均": "^N225"}
    data = {}
    for name, ticker in indices.items():
        try:
            d = yf.download(ticker, period="5d", progress=False)
            if not d.empty:
                if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
                close = d['Close'].iloc[-1]
                delta = close - d['Close'].iloc[-2]
                data[name] = (round(float(close), 2), round(float(delta), 2))
        except: data[name] = (0.0, 0.0)
    return data

# --- 3. メインUI ---
indices = get_market_indices()
idx_c1, idx_c2, idx_c3 = st.columns(3)
with idx_c1:
    v, d = indices.get("USD/JPY", (0,0))
    st.metric("米ドル/円", f"{v} 円", f"{d}")
with idx_c2:
    v, d = indices.get("日経平均", (0,0))
    st.metric("日経平均", f"{v:,.0f} 円", f"{d:,.0f}")
with idx_c3:
    st.info("💡 出来高チャート付き・安定高速モード")

# --- 4. サイドバー ---
st.sidebar.header("🔍 スクリーニング条件")

# 戦略（スキャンモード）の選択肢
scan_mode = st.sidebar.radio("スキャンモード", ["Volume Spike (初動上昇)", "Inside Bar (25MA割れはらみ足押し目)"])

st.sidebar.divider()

# モードに応じて設定項目を出し分け
if scan_mode == "Volume Spike (初動上昇)":
    vol_days = st.sidebar.number_input("連続出来高増加日数", min_value=1, max_value=30, value=3)
    price_days = st.sidebar.number_input("株価上昇判定(日前比)", min_value=1, max_value=30, value=3)
    config_dict = {"vol_days": vol_days, "price_days": price_days}
else:
    ma_period = st.sidebar.number_input("トレンド判定移動平均線", min_value=5, max_value=75, value=25)
    config_dict = {"ma_period": ma_period}

st.sidebar.divider()
st.sidebar.subheader("予算範囲 (万円)")
inv_max = st.sidebar.number_input("最大予算 (万円)", min_value=0, value=100, step=5)
inv_min = st.sidebar.number_input("最小予算 (万円)", min_value=0, value=0, step=5)
st.sidebar.divider()

jpx_df = get_jpx_full_data()
if not jpx_df.empty and '市場・商品区分' in jpx_df.columns:
    target_keywords = ["プライム", "スタンダード", "グロース", "ETF・ETN"]
    ui_display_options = [
        m for m in sorted(jpx_df['市場・商品区分'].unique().tolist()) 
        if any(key in m for key in target_keywords) and "外国" not in m
    ]
    default_choice = [m for m in ui_display_options if "プライム" in m]
    selected_markets = st.sidebar.multiselect("対象市場", options=ui_display_options, default=default_choice)
else:
    selected_markets = []

num_scan = st.sidebar.slider("最大スキャン件数", 10, 4000, 500)
start_btn = st.sidebar.button("🚀 スキャン開始")

# --- 5. 解析関数 ---
def analyze_stocks(tickers, scan_mode, config_dict, inv_min, inv_max):
    import yfinance as yf
    import pandas as pd
    res = []
    if not tickers: return res
    
    try:
        data = yf.download(tickers, period="3mo", progress=False, group_by='ticker', threads=False)
        for t in tickers:
            try:
                df = data[t].dropna() if len(tickers) > 1 else data.dropna()
                if len(df) < 30: continue
                
                close = df['Close'].values
                high = df['High'].values
                low = df['Low'].values
                vol = df['Volume'].values
                
                p = float(close[-1])
                inv = round((p * 100) / 10000, 1)
                if not (inv_min <= inv <= inv_max): continue

                # 【1. 初動上昇（ボリュームスパイク）モード】
                if scan_mode == "Volume Spike (初動上昇)":
                    v_days = config_dict['vol_days']
                    p_days = config_dict['price_days']
                    if len(df) < max(v_days, p_days) + 1: continue
                    
                    if all(vol[-j] > vol[-j-1] for j in range(1, v_days + 1)) and close[-1] > close[-p_days-1]:
                        res.append({
                            'コード': t.replace(".T", ""), 
                            '価格': round(p, 1), 
                            '前日比': round(float(close[-1] - close[-2]), 1), 
                            '投資額(万円)': inv, 
                            '売買代金(百万)': int((p * vol[-1]) / 1000000),
                            'シグナル': '初動急騰'
                        })

                # 【2. 25MA割れはらみ足押し目モード】
                else:
                    ma_p = config_dict['ma_period']
                    if len(df) < ma_p + 3: continue
                    
                    ma = pd.Series(close).rolling(window=ma_p).mean().values
                    
                    # 条件A: 25MA自体は右肩上がり（長期上昇トレンド維持）
                    is_uptrend = ma[-1] > ma[-5]
                    # 条件B: 昨日または今日の安値が25MA以下まで突っ込んでいる（しっかり調整した）
                    is_touching_or_below_ma = (low[-2] <= ma[-2]) or (low[-1] <= ma[-1])
                    # 条件C: 本日のローソク足が前日の範囲内にスッポリ収まっている（はらみ足）
                    is_inside_bar = (high[-1] <= high[-2]) and (low[-1] >= low[-2])
                    
                    if is_uptrend and is_touching_or_below_ma and is_inside_bar:
                        res.append({
                            'コード': t.replace(".T", ""), 
                            '価格': round(p, 1), 
                            '前日比': round(float(close[-1] - close[-2]), 1), 
                            '投資額(万円)': inv, 
                            '売買代金(百万)': int((p * vol[-1]) / 1000000),
                            'シグナル': '25MA割れはらみ'
                        })
            except: continue
    except: pass
    return res

# --- 6. 実行ロジック ---
if start_btn:
    start_time = time.time()
    target_df = jpx_df[jpx_df['市場・商品区分'].isin(selected_markets)].head(num_scan)
    all_tickers = [f"{c}.T" for c in target_df['コード']]
   
    if all_tickers:
        with st.spinner(f"🚀 {len(all_tickers)}銘柄を解析中..."):
            res_list = []
            chunk_size = 80
            chunks = [all_tickers[i:i + chunk_size] for i in range(0, len(all_tickers), chunk_size)]
            with ProcessPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(analyze_stocks, chunks[i], scan_mode, config_dict, inv_min, inv_max) for i in range(len(chunks))]
                for future in as_completed(futures):
                    res_list.extend(future.result())
           
            if res_list:
                final_df = pd.DataFrame(res_list)
                st.session_state.res_df = final_df.merge(target_df[['コード', '銘柄名']], on='コード').sort_values('売買代金(百万)', ascending=False)
            else:
                st.session_state.res_df = None
            st.session_state.elapsed_time = round(time.time() - start_time, 2)

# --- 7. 結果表示 ---
if "res_df" in st.session_state and st.session_state.res_df is not None:
    st.success(f"✅ 完了: {st.session_state.elapsed_time} 秒 / ヒット: {len(st.session_state.res_df)} 銘柄")
    
    # 表示用のデータフレームを作成し、ヤフーファイナンスのリンクを自動生成
    display_df = st.session_state.res_df.copy()
    display_df['詳細リンク'] = "https://finance.yahoo.co.jp/quote/" + display_df['コード'] + ".T"
    
    cols = ['コード', '銘柄名', '詳細リンク', '価格', '前日比', '投資額(万円)', '売買代金(百万)', 'シグナル']
    cols = [c for c in cols if c in display_df.columns]
    display_df = display_df[cols]

    selection = st.dataframe(
        display_df, 
        use_container_width=True, 
        hide_index=True, 
        on_select="rerun", 
        selection_mode="single-row",
        column_config={
            "詳細リンク": st.column_config.LinkColumn("詳細リンク", display_text="Yahoo!で開く 🔗")
        }
    )

    if selection.selection.rows:
        target_row = st.session_state.res_df.iloc[selection.selection.rows[0]]
        st.divider()
       
        c_df = yf.download(f"{target_row['コード']}.T", period="6mo", progress=False)
        if isinstance(c_df.columns, pd.MultiIndex): c_df.columns = c_df.columns.get_level_values(0)
        c_df = c_df.dropna()

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        # チャートにも設定した移動平均線（25MA等）を重ねて描画
        ma_p = config_dict['ma_period'] if scan_mode == "Inside Bar (25MA割れはらみ足押し目)" else 25
        c_df['MA'] = c_df['Close'].rolling(window=ma_p).mean()
        fig.add_trace(go.Scatter(x=c_df.index, y=c_df['MA'], name=f"{ma_p}日移動平均線", line=dict(color='#FF9F43', width=1.5)), row=1, col=1)

        fig.add_trace(go.Candlestick(x=c_df.index, open=c_df['Open'], high=c_df['High'], low=c_df['Low'], close=c_df['Close'], name="株価"), row=1, col=1)
       
        fig.add_trace(go.Bar(x=c_df.index, y=c_df['Volume'], name="出来高", marker_color='#4A69BD', opacity=0.8), row=2, col=1)

        dt_all = pd.date_range(start=c_df.index[0], end=c_df.index[-1], freq='D')
        dt_obs = [d.strftime("%Y-%m-%d") for d in c_df.index]
        dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if d not in dt_obs]
        
        fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])
        
        title_suffix = f" - [{target_row['シグナル']}]" if 'シグナル' in target_row else ""
        fig.update_layout(xaxis_rangeslider_visible=False, height=600, template="plotly_white", title=f"📈 {target_row['銘柄名']} ({target_row['コード']}){title_suffix}", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

elif start_btn:
    st.warning("条件に合う銘柄は見つかりませんでした。")
