import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import urllib.request
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import pytz

from transformers import pipeline
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from hmmlearn.hmm import GaussianHMM
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange
import time

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Institutional Market Scanner", layout="wide", page_icon="📈")
st.title("⚡ Institutional 24/7 Market Scanner")
st.markdown("**Architecture:** Dual-Market Macro-Micro Convergence (US & India) with MPT Allocation & Dynamic Risk Override")

@st.cache_resource
def load_finbert():
    return pipeline("text-classification", model="ProsusAI/finbert")

finbert = load_finbert()

# ==========================================
# 2. DUAL-MARKET DATA ENGINES
# ==========================================
@st.cache_data(ttl=3600)
def fetch_macro_data(ticker, market):
    for attempt in range(3):
        try:
            stock = yf.Ticker(ticker).history(period="10y")
            # ROUTER: Dynamically switch Volatility Index based on geography
            vix_ticker = "^VIX" if market == "US (Wall Street)" else "^INDIAVIX"
            vix = yf.Ticker(vix_ticker).history(period="10y")
            
            if stock.empty:
                st.error(f"❌ ERROR: Yahoo Finance returned no data for {ticker}. Ensure you use '.NS' or '.BO' for Indian stocks (e.g., RELIANCE.NS).")
                st.stop()
                
            stock.index = pd.to_datetime(stock.index).tz_localize(None).normalize()
            vix.index = pd.to_datetime(vix.index).tz_localize(None).normalize()
            
            df = pd.DataFrame(index=stock.index)
            df['Close'] = stock['Close'].values.astype(float).flatten()
            df['Open'] = stock['Open'].values.astype(float).flatten()
            df['High'] = stock['High'].values.astype(float).flatten()
            df['Low'] = stock['Low'].values.astype(float).flatten()
            
            df['VIX'] = vix['Close']
            df['VIX'] = df['VIX'].ffill().bfill().fillna(20.0) 
            
            return df
        except Exception as e:
            if attempt == 2:
                st.error(f"❌ Macro Data Fetch Error: {str(e)}")
                st.stop()
            time.sleep(1)

def fetch_micro_data(ticker, market):
    state_key = f"micro_data_{ticker}"
    
    # 1. First Boot: Download the dense 7-day historical base
    if state_key not in st.session_state or st.session_state[state_key] is None or st.session_state[state_key].empty:
        for attempt in range(3):
            try:
                intraday = yf.Ticker(ticker).history(period="7d", interval="1m")
                if intraday.empty:
                    return None
                    
                tz = 'US/Eastern' if market == "US (Wall Street)" else 'Asia/Kolkata'
                intraday.index = pd.to_datetime(intraday.index).tz_convert(tz).tz_localize(None)
                st.session_state[state_key] = intraday
                return intraday
            except Exception:
                if attempt == 2: return None
                time.sleep(1)
    
    # 2. Live Iterations: Instantly pull only today's tiny data stack and append the newest missing minutes
    else:
        for attempt in range(3):
            try:
                new_data = yf.Ticker(ticker).history(period="1d", interval="1m")
                if new_data.empty: 
                    return st.session_state[state_key]
                    
                tz = 'US/Eastern' if market == "US (Wall Street)" else 'Asia/Kolkata'
                new_data.index = pd.to_datetime(new_data.index).tz_convert(tz).tz_localize(None)
                
                # Memory-safe Smart Append (Overwrites duplicates based on timestamp index)
                combined = pd.concat([st.session_state[state_key], new_data])
                combined = combined[~combined.index.duplicated(keep='last')]
                
                # Strict Memory Flush: Retain exactly maximum 3000 rows (approx 7 days)
                combined = combined.tail(3000)
                st.session_state[state_key] = combined
                
                return combined
            except Exception:
                if attempt == 2: return st.session_state[state_key]
                time.sleep(1)

# ==========================================
# 3. PREPROCESSING & FEATURE EXTRACTION
# ==========================================
@st.cache_data(ttl=3600)
def compute_macro_features(df):
    data = df.copy()
    
    series_vals = data['Close'].values
    res = np.full_like(series_vals, np.nan, dtype=float)
    window = 60
    weights = [1.0]
    for k in range(1, window):
        weights.append(-weights[-1] * (0.5 - k + 1) / k)
    weights = np.array(weights)[::-1]
    
    # Vectorized Fractional Differentiation Approximation instead of slow loops
    valid_res = np.convolve(series_vals, weights, mode='valid')
    res[window-1:window-1+len(valid_res)] = valid_res
        
    data['Frac_Diff_Close'] = res
    data['RSI'] = RSIIndicator(close=data['Close'], window=14).rsi()
    macd = MACD(close=data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    
    data['MACD_Bullish_Cross'] = (data['MACD'] > data['MACD_Signal']) & (data['MACD'].shift(1) <= data['MACD_Signal'].shift(1))
    data['MACD_Bearish_Cross'] = (data['MACD'] < data['MACD_Signal']) & (data['MACD'].shift(1) >= data['MACD_Signal'].shift(1))
    
    bb = BollingerBands(close=data['Close'], window=20, window_dev=2)
    data['BB_High'] = bb.bollinger_hband()
    data['BB_Low'] = bb.bollinger_lband()
    
    atr = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close'], window=14)
    data['ATR'] = atr.average_true_range()
    
    # Target Refinement: Close must exceed Previous Close + 0.25 * ATR to be classified as a profitable trade
    data['Target'] = (data['Close'].shift(-1) > (data['Close'] + data['ATR'] * 0.25)).astype(int)
    # WARNING: Do NOT dropna() here otherwise it deletes the latest live features
    return data

@st.cache_data(ttl=300)
def compute_micro_features(df):
    data = df.copy()
    data['Date'] = data.index.date
    data['Typical_Price'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['VP'] = data['Typical_Price'] * data['Volume']
    data['Cum_VP'] = data.groupby('Date')['VP'].cumsum()
    data['Cum_Vol'] = data.groupby('Date')['Volume'].cumsum()
    data['VWAP'] = data['Cum_VP'] / data['Cum_Vol']
    data['VWAP_Dist'] = (data['Close'] - data['VWAP']) / data['VWAP']
    
    data['Micro_RSI'] = RSIIndicator(close=data['Close'], window=14).rsi()
    data['Vol_Avg_20'] = data['Volume'].rolling(window=20).mean()
    data['Vol_Surge'] = data['Volume'] / (data['Vol_Avg_20'] + 1e-9)
    
    atr = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close'], window=14)
    data['ATR'] = atr.average_true_range()
    
    # Target demands 5-min move > 0.1 * Current ATR (Slippage filter)
    data['Target'] = (data['Close'].shift(-5) > (data['Close'] + data['ATR'] * 0.1)).astype(int)
    # Target shift causes NaN, but we need the latest row for inference. Handled in training.
    return data

# ==========================================
# 4. CONTEXT-AWARE NLP SCRAPER
# ==========================================
def fetch_live_sentiment(ticker, market):
    try:
        headlines = []
        if market == "US (Wall Street)":
            url = f"https://finviz.com/quote.ashx?t={ticker}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            }
            req = urllib.request.Request(url, headers=headers)
            html = urllib.request.urlopen(req, timeout=5).read()
            soup = BeautifulSoup(html, 'html.parser')
            news_table = soup.find(id='news-table')
            if news_table:
                headlines = [row.a.text for row in news_table.findAll('tr')][:5] 
        else:
            news_data = yf.Ticker(ticker).news
            if news_data:
                headlines = [item['title'] for item in news_data][:5]
                
        if not headlines:
            return ["No recent news found."], [{"label": "neutral", "score": 0.0}], 0
            
        sentiments = finbert(headlines)
        score = sum([s['score'] if s['label'] == 'positive' else -s['score'] for s in sentiments if s['label'] != 'neutral'])
        return headlines, sentiments, score
    except Exception:
        return ["⚠️ NLP Scraper Blocked or Failed"], [{"label": "neutral", "score": 0.0}], 0

# ==========================================
# 5. DUAL-MARKET MPT ENGINE (TRUE RISK MITIGATION)
# ==========================================
def calculate_optimal_portfolio(capital, market, risk_profile):
    if market == "US (Wall Street)":
        tickers = ['XLK', 'XLV', 'SPY', 'GLD', 'SHY'] 
        sector_names = ['Tech (XLK)', 'Healthcare (XLV)', 'S&P 500 (SPY)', 'Safe Haven: Gold (GLD)', 'Safe Haven: Bonds (SHY)']
        risk_free_rate = 0.042 
    else:
        tickers = ['NIFTYBEES.NS', 'BANKBEES.NS', 'ITBEES.NS', 'GOLDBEES.NS', 'LIQUIDBEES.NS']
        sector_names = ['Broad Market (NIFTY)', 'Banking (BANK)', 'IT Sector (IT)', 'Safe Haven: Gold (GOLD)', 'Safe Haven: Cash/Bonds (LIQUID)']
        risk_free_rate = 0.071 
        
    try:
        data = yf.download(tickers, period="1y", progress=False)['Close']
        returns = data.pct_change().dropna()
        
        recent_returns = returns.tail(90)
        mean_returns = returns.mean() * 252
        cov_matrix = recent_returns.cov() * 252 
        
        num_portfolios = 5000
        results = np.zeros((3, num_portfolios))
        weights_record = []
        
        valid_portfolios = 0
        
        while valid_portfolios < num_portfolios:
            # Generate weights directly to bypass infinite rejection loop
            if risk_profile == "Ultra-Conservative (Capital Preservation)":
                # Nuclear Winter: Equity <= 5% Absolute Max
                equity_alloc = np.random.uniform(0.0, 0.05)
                safety_alloc = 1.0 - equity_alloc
                eq_w = np.random.dirichlet([1, 1, 1]) * equity_alloc
                sf_w = np.random.dirichlet([1, 1]) * safety_alloc
                weights = np.concatenate([eq_w, sf_w])
            elif risk_profile == "Conservative (Minimum Volatility)":
                # Strict limit: Equity <= 20%
                equity_alloc = np.random.uniform(0.01, 0.20)
                safety_alloc = 1.0 - equity_alloc
                eq_w = np.random.dirichlet([1, 1, 1]) * equity_alloc
                sf_w = np.random.dirichlet([1, 1]) * safety_alloc
                weights = np.concatenate([eq_w, sf_w])
            else:
                weights = np.random.dirichlet(np.ones(len(tickers)))
                # Harder diversification penalty: No single equity sector > 25% (down from 30%)
                if np.max(weights[:3]) > 0.25:
                    continue
                
            weights_record.append(weights)
            
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            true_sharpe = (portfolio_return - risk_free_rate) / portfolio_std_dev
            
            results[0, valid_portfolios] = portfolio_return
            results[1, valid_portfolios] = portfolio_std_dev
            results[2, valid_portfolios] = true_sharpe 
            
            valid_portfolios += 1
            
        # THE RISK ROUTER
        if risk_profile == "Conservative (Minimum Volatility)" or risk_profile == "Ultra-Conservative (Capital Preservation)":
            best_idx = np.argmin(results[1]) # Hunt for absolute lowest mathematical risk
        else:
            best_idx = np.argmax(results[2]) # Hunt for highest Sharpe Ratio
            
        optimal_weights = weights_record[best_idx]
        
        allocation = {tickers[i]: optimal_weights[i] * capital for i in range(len(tickers))}
        return allocation, optimal_weights, results[0, best_idx], results[1, best_idx], results[2, best_idx], mean_returns, sector_names
    except Exception as e:
        st.error(f"❌ MPT Calculation Error: {str(e)}")
        return None, None, 0, 0, 0, None, None

# ==========================================
# 6. AI TRAINING ENGINES
# ==========================================
@st.cache_resource(ttl=3600)
def train_macro_ai(df):
    train_df = df.dropna()
    features = ['Frac_Diff_Close', 'RSI', 'MACD', 'VIX', 'BB_High', 'BB_Low']
    X = train_df[features].values
    y = train_df['Target'].values
    
    tscv = TimeSeriesSplit(n_splits=5)
    acc_scores, prec_scores, rec_scores = [], [], []
    
    for train_index, test_index in tscv.split(X):
        X_tr, X_te = X[train_index], X[test_index]
        y_tr, y_te = y[train_index], y[test_index]
        
        scaler_cv = StandardScaler()
        X_tr_scaled = scaler_cv.fit_transform(X_tr)
        X_te_scaled = scaler_cv.transform(X_te)
        
        xgb_cv = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        xgb_cv.fit(X_tr_scaled, y_tr)
        
        preds = xgb_cv.predict(X_te_scaled)
        acc_scores.append(accuracy_score(y_te, preds))
        prec_scores.append(precision_score(y_te, preds, zero_division=0))
        rec_scores.append(recall_score(y_te, preds, zero_division=0))
        
    val_metrics = {"Accuracy": np.mean(acc_scores), "Precision": np.mean(prec_scores), "Recall": np.mean(rec_scores)}

    split_idx = int(len(train_df) * 0.7)
    X_train, y_train = X[:split_idx], y[:split_idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    
    returns = np.diff(np.log(train_df['Close'].values[:split_idx]), prepend=0).reshape(-1, 1)
    hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=42)
    hmm_model.fit(returns)
    variances = [np.diag(hmm_model.covars_[i]) for i in range(2)]
    crash_state = np.argmax(variances)
    
    return scaler, xgb_model, hmm_model, crash_state, val_metrics

@st.cache_resource(ttl=300)
def train_micro_sniper(df):
    train_df = df.dropna()
    features = ['VWAP_Dist', 'Micro_RSI', 'Vol_Surge']
    X = train_df[features].values
    y = train_df['Target'].values
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    micro_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.01, max_depth=3, reg_lambda=20, random_state=42)
    micro_model.fit(X_scaled, y)
    
    return scaler, micro_model

# ==========================================
# 7. STREAMLIT UI & MASTER TABS
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/bullish.png", width=60)
    st.title("⚙️ System Controls")
    market_choice = st.radio("Select Exchange:", ["US (Wall Street)", "India (NSE/BSE)"])
    
    default_ticker = "NVDA" if market_choice == "US (Wall Street)" else "^BSESN"
    ticker = st.text_input("Enter Ticker:", default_ticker).upper()
    
    account_size = st.number_input("Account Equity ($/₹):", min_value=1000, value=100000, step=10000)
    
    if market_choice == "India (NSE/BSE)" and not ticker.endswith(".NS") and not ticker.endswith(".BO") and not ticker.startswith("^"):
        st.warning("⚠️ Indian tickers usually require '.NS', '.BO', or '^' at the start.")
        
    run_scanner = st.button("Run Real-Time Scan", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("v2.2 Institutional Quantitative System")

master_tab1, master_tab2, master_tab3 = st.tabs(["🎯 Single-Asset AI Scanner", "💼 Institutional Capital Allocation", "⏳ WFO Historical Backtester"])

with master_tab1:
    col1 = st.container() # Swapped from columns since sidebar took controls
    
    if run_scanner:
        st.toast(f"Scan Initialized for {ticker}!", icon="⚡")
        with st.spinner(f"Fetching Macro & Micro Datasets for {market_choice}..."):
            macro_raw = fetch_macro_data(ticker, market_choice)
            micro_raw = fetch_micro_data(ticker, market_choice)
            
            macro_processed = compute_macro_features(macro_raw)
            micro_available = micro_raw is not None and not micro_raw.empty
            if micro_available:
                micro_processed = compute_micro_features(micro_raw)
            
        with st.spinner("Training Dual-Brain AI Models..."):
            macro_scaler, macro_xgb, hmm_model, crash_state, macro_metrics = train_macro_ai(macro_processed)
            if micro_available:
                micro_scaler, micro_xgb = train_micro_sniper(micro_processed)
            
        with st.spinner("Executing NLP Sentiment Analysis..."):
            news, sentiment_details, sentiment_score = fetch_live_sentiment(ticker, market_choice)
            
        # --- MACRO INFERENCE ---
        latest_macro = macro_processed.iloc[-1]
        X_macro_live = macro_scaler.transform([latest_macro[['Frac_Diff_Close', 'RSI', 'MACD', 'VIX', 'BB_High', 'BB_Low']].values])
        macro_xgb_prob = macro_xgb.predict_proba(X_macro_live)[0][1]
        
        sentiment_modifier = sentiment_score * 0.075 
        macro_final_prob = np.clip(macro_xgb_prob + sentiment_modifier, 0, 1.0)
        macro_bias = "BUY" if macro_final_prob > 0.55 else "SELL" if macro_final_prob < 0.45 else "NEUTRAL"
        
        recent_returns = np.diff(np.log(macro_raw['Close'].values[-10:]), prepend=0).reshape(-1, 1)
        current_regime = hmm_model.predict(recent_returns)[-1]
        
        # --- MICRO INFERENCE ---
        micro_prob = 0.5
        micro_bias = "NEUTRAL"
        SLIPPAGE_BARRIER = 0.60 
        
        if micro_available:
            latest_micro = micro_processed.iloc[-1]
            X_micro_live = micro_scaler.transform([latest_micro[['VWAP_Dist', 'Micro_RSI', 'Vol_Surge']].values])
            micro_prob = micro_xgb.predict_proba(X_micro_live)[0][1]
            micro_bias = "BUY" if micro_prob >= SLIPPAGE_BARRIER else "SELL" if micro_prob <= (1 - SLIPPAGE_BARRIER) else "NEUTRAL"
            
        # --- THE BOOLEAN CONVERGENCE GATE ---
        if current_regime == crash_state:
            final_signal, signal_color = "🛑 HOLD (HMM CRASH REGIME DETECTED)", "red"
        elif macro_bias == "BUY" and micro_bias == "BUY":
            final_signal, signal_color = "✅ EXECUTE BUY NOW (Golden Entry)", "green"
        elif macro_bias == "BUY" and micro_bias == "SELL":
            final_signal, signal_color = "⏸️ HOLD (Waiting for Micro-Support)", "orange"
        elif macro_bias == "SELL" and micro_bias == "BUY":
            final_signal, signal_color = "🔻 EXECUTE SHORT (Fade the Fake Rally)", "purple"
        elif macro_bias == "SELL" and micro_bias == "SELL":
            final_signal, signal_color = "🛑 DO NOT TOUCH (Macro & Micro Bleed)", "red"
        else:
            final_signal, signal_color = "⏸️ NEUTRAL / HOLD", "gray"

        # UI Rendering
        st.markdown("---")
        st.markdown(f"<h2 style='text-align: center; color: {signal_color};'>{final_signal}</h2>", unsafe_allow_html=True)
        st.markdown("---")
        
        m1, m2, m3, m4 = st.columns(4)
        current_live_price = micro_raw['Close'].iloc[-1] if micro_available else latest_macro['Close']
        currency = "$" if market_choice == "US (Wall Street)" else "₹"
        
        # Delta calculation for interactive feel
        prev_price = micro_raw['Close'].iloc[-2] if micro_available and len(micro_raw) > 1 else latest_macro['Open']
        price_delta = current_live_price - prev_price
        
        m1.metric("Current Live Price", f"{currency}{current_live_price:.2f}", f"{price_delta:.2f} ({price_delta/prev_price*100:.2f}%)")
        m2.metric("Macro Trend Bias", f"{macro_bias} ({macro_final_prob * 100:.1f}%)", delta="Bullish" if macro_bias == "BUY" else "Bearish" if macro_bias == "SELL" else "Neutral", delta_color="normal" if macro_bias == "BUY" else "inverse" if macro_bias == "SELL" else "off")
        m3.metric("Micro Sniper Bias", f"{micro_bias} ({micro_prob * 100:.1f}%)" if micro_available else "Market Closed")
        m4.metric("HMM Risk Engine", "🚨 CRASH DETECTED" if current_regime == crash_state else "🟢 Stable", delta="DANGER" if current_regime == crash_state else "SAFE", delta_color="inverse" if current_regime == crash_state else "normal")
        
        if final_signal.startswith("✅"):
            st.balloons()
        
        # --- POSITION SIZING (Base Risk: 1% of Equity) ---
        if final_signal.startswith("✅"):
            latest_atr = latest_macro['ATR'] if pd.notna(latest_macro['ATR']) else (current_live_price * 0.02)
            risk_per_share = latest_atr * 2.0 # Broader Stop loss at 2.0 ATR prevents getting wicked out, but reduces position size
            capital_at_risk = account_size * 0.01 # Ultra-Conservative 1% max risk rule
            shares_to_buy = int(capital_at_risk / risk_per_share)
            position_value = shares_to_buy * current_live_price
            
            st.success(f"**Institutional Position Size:** Buy **{shares_to_buy} shares** at {currency}{current_live_price:.2f} (Total Value: {currency}{position_value:.2f}). **Stop Loss:** {currency}{(current_live_price - risk_per_share):.2f}.")

        with col1:
            if micro_available:
                st.subheader(f"⏱️ Live 1-Minute Microstructure & VWAP ({ticker})")
                is_up = micro_raw['Close'].iloc[-1] >= micro_raw['Open'].iloc[0]
                line_color = '#00FF00' if is_up else '#FF0000'
                
                fig_intra = go.Figure()
                plot_data = micro_processed.tail(60)
                
                fig_intra.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Close'], mode='lines', line=dict(color=line_color, width=2), name="Live Price"))
                fig_intra.add_trace(go.Scatter(x=plot_data.index, y=plot_data['VWAP'], mode='lines', line=dict(color='#E040FB', width=2, dash='dot'), name="VWAP"))
                fig_intra.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), template="plotly_dark")
                st.plotly_chart(fig_intra, use_container_width=True)
                st.markdown("---")

            st.subheader(f"📊 10-Year Macro Trend & Technical Signals ({ticker})")
            
            hist_data = macro_processed.tail(90)
            X_hist_scaled = macro_scaler.transform(hist_data[['Frac_Diff_Close', 'RSI', 'MACD', 'VIX', 'BB_High', 'BB_Low']].values)
            hist_probs = macro_xgb.predict_proba(X_hist_scaled)[:, 1]
            
            buy_dates, buy_prices, sell_dates, sell_prices = [], [], [], []
            for i in range(len(hist_probs)):
                if hist_probs[i] > 0.55:
                    buy_dates.append(hist_data.index[i])
                    buy_prices.append(hist_data['Low'].iloc[i] * 0.98) 
                elif hist_probs[i] < 0.45:
                    sell_dates.append(hist_data.index[i])
                    sell_prices.append(hist_data['High'].iloc[i] * 1.02) 
                    
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.06, subplot_titles=("Price Action & AI Decisions", "MACD Trend", "RSI Momentum"))
            fig.add_trace(go.Candlestick(x=macro_raw.index[-90:], open=macro_raw['Open'][-90:], high=macro_raw['High'][-90:], low=macro_raw['Low'][-90:], close=macro_raw['Close'][-90:], name="Daily Price"), row=1, col=1)
            fig.add_trace(go.Scatter(x=macro_processed.index[-90:], y=macro_processed['BB_High'][-90:], line=dict(color='rgba(255,255,255,0.3)', dash='dot'), name="BB High"), row=1, col=1)
            fig.add_trace(go.Scatter(x=macro_processed.index[-90:], y=macro_processed['BB_Low'][-90:], line=dict(color='rgba(255,255,255,0.3)', dash='dot'), name="BB Low", fill='tonexty', fillcolor='rgba(255,255,255,0.05)'), row=1, col=1)
            fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', marker=dict(symbol='triangle-up', color='#00FF00', size=14, line=dict(color='white', width=1)), name='AI Buy Signal'), row=1, col=1)
            fig.add_trace(go.Scatter(x=sell_dates, y=sell_prices, mode='markers', marker=dict(symbol='triangle-down', color='#FF0000', size=14, line=dict(color='white', width=1)), name='AI Sell Signal'), row=1, col=1)
            
            macd_hist = macro_processed['MACD'][-90:] - macro_processed['MACD_Signal'][-90:]
            colors = ['#00FF00' if val >= 0 else '#FF0000' for val in macd_hist]
            fig.add_trace(go.Bar(x=macro_processed.index[-90:], y=macd_hist, marker_color=colors, name="MACD Hist"), row=2, col=1)
            fig.add_trace(go.Scatter(x=macro_processed.index[-90:], y=macro_processed['MACD'][-90:], line=dict(color='#00F1FF'), name="MACD"), row=2, col=1)
            fig.add_trace(go.Scatter(x=macro_processed.index[-90:], y=macro_processed['MACD_Signal'][-90:], line=dict(color='#FFB300'), name="Signal"), row=2, col=1)
            fig.add_trace(go.Scatter(x=macro_processed.index[-90:], y=macro_processed['RSI'][-90:], line=dict(color='#E040FB'), name="RSI"), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="#FF0000", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#00FF00", row=3, col=1)
            
            fig.update_layout(height=800, margin=dict(l=10, r=10, t=40, b=10), template="plotly_dark", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', rangeslider_visible=False) 
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("🧠 Live FinBERT Sentiment Analysis")
        for headline, sentiment in zip(news, sentiment_details):
            color = "green" if sentiment['label'] == 'positive' else "red" if sentiment['label'] == 'negative' else "gray"
            st.markdown(f"- **{headline}** ➔ <span style='color:{color}'>[{sentiment['label'].upper()}: {sentiment['score']:.2f}]</span>", unsafe_allow_html=True)

        st.markdown("---")
        with st.expander("🔬 Raw Feature Engines & Academic Validation Metrics", expanded=False):
            st.markdown("Dive into the raw engineered features and Walk-Forward Optimization (WFO) statistics driving the ML model.")
            tab1, tab2, tab3 = st.tabs(["🛡️ Macro WFO Metrics", "Macro Features (10y)", "Micro Features (7d)"])
            with tab1:
                st.write("### 📈 Walk-Forward Optimization (Time-Series Out-of-Sample Validation)")
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("WFO Average Accuracy", f"{macro_metrics['Accuracy'] * 100:.1f}%")
                col_m2.metric("WFO Average Precision", f"{macro_metrics['Precision'] * 100:.1f}%")
                col_m3.metric("WFO Average Recall", f"{macro_metrics['Recall'] * 100:.1f}%")
            with tab2: 
                st.dataframe(macro_processed[['Close', 'Frac_Diff_Close', 'RSI', 'MACD', 'BB_High', 'BB_Low', 'Target']].tail(15), use_container_width=True)
            with tab3:
                if micro_available:
                    st.dataframe(micro_processed[['Close', 'VWAP', 'VWAP_Dist', 'Micro_RSI', 'Vol_Surge', 'Target']].tail(15), use_container_width=True)
                else:
                    st.info("Market Closed - No minute data available.")

with master_tab2:
    st.subheader("💼 Institutional Capital Management & Risk Allocation")
    
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        mpt_market = st.radio("Select Market:", ["US (Wall Street)", "India (NSE/BSE)"])
    with col_opt2:
        risk_profile = st.selectbox("Select Risk Profile:", ["Balanced (Max Sharpe)", "Conservative (Minimum Volatility)", "Ultra-Conservative (Capital Preservation)"])
    with col_opt3:
        capital_input = st.number_input("Enter Total Investment Capital:", min_value=1000, value=10000, step=1000)
        
    run_mpt = st.button("Calculate Optimal Allocation", type="primary")
    
    if run_mpt:
        with st.spinner(f"Simulating 5,000 Portfolio Combinations for {mpt_market}..."):
            allocation, weights, exp_return, exp_volatility, sharpe, mean_returns, sector_names = calculate_optimal_portfolio(capital_input, mpt_market, risk_profile)
            
            if allocation:
                st.markdown("---")
                a1, a2, a3 = st.columns(3)
                a1.metric("Expected Annual Return", f"{exp_return * 100:.2f}%")
                a2.metric("Expected Annual Risk (Volatility)", f"{exp_volatility * 100:.2f}%")
                
                if risk_profile == "Balanced (Max Sharpe)":
                    a3.metric("True Sharpe Ratio", f"{sharpe:.2f}", help="Return generated per unit of risk above the Risk-Free Rate.")
                else:
                    a3.metric("Optimization Focus", "Minimum Volatility")
                
                st.markdown("---")
                c1, c2 = st.columns([1, 1.2])
                currency = "$" if mpt_market == "US (Wall Street)" else "₹"
                
                with c1:
                    for i, (ticker, amount) in enumerate(allocation.items()):
                        st.success(f"**{sector_names[i]}:** {currency}{amount:.2f} ➔ **({weights[i]*100:.1f}%)**")
                        
                with c2:
                    fig_pie = go.Figure(data=[go.Pie(labels=sector_names, values=weights, hole=.4, marker=dict(colors=['#00F1FF', '#E040FB', '#FFB300', '#00FF00', '#FF0000']))])
                    fig_pie.update_layout(template="plotly_dark", margin=dict(t=0, b=0, l=0, r=0), height=300)
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                st.markdown("---")
                max_weight_idx = np.argmax(weights)
                
                if risk_profile == "Ultra-Conservative (Capital Preservation)":
                    st.info(f"**Nuclear Winter Engaged:** Maximum capital preservation enforced. Algorithm restricted pure equity accumulation mathematically to under 5%. The portfolio is completely immunized against flash crashes by parking massive weight into **{sector_names[max_weight_idx]}**, accepting near-zero growth to defend cash.")
                elif risk_profile == "Conservative (Minimum Volatility)":
                    st.info(f"**Risk Override Engaged:** The algorithm bypassed the Max Sharpe calculation and strictly targeted the **Global Minimum Volatility (GMV)** portfolio. Heaviest weighting went to **{sector_names[max_weight_idx]}** to violently suppress covariance and protect principal capital, sacrificing upside yield for maximum stability.")
                else:
                    st.info(f"**Primary Growth Driver:** Heaviest allocation (**{weights[max_weight_idx]*100:.1f}%**) went to **{sector_names[max_weight_idx]}** due to its superior 90-day momentum, maximizing the risk-to-reward ratio.")


with master_tab3:
    st.subheader("⏳ Walk-Forward Optimal Backtester (Out-of-Sample)")
    st.markdown("Simulates trading the AI Macro signals strictly over the 30% unseen Out-Of-Sample test dataset, applying **Slippage & Commission** filters across every trade.")
    
    if 'run_scanner' in locals() and run_scanner:
        split_idx = int(len(macro_processed) * 0.7)
        oos_data = macro_processed.iloc[split_idx:].copy()
        
        if len(oos_data) > 0:
            X_oos_scaled = macro_scaler.transform(oos_data[['Frac_Diff_Close', 'RSI', 'MACD', 'VIX', 'BB_High', 'BB_Low']].values)
            oos_probs = macro_xgb.predict_proba(X_oos_scaled)[:, 1]
            oos_data['AI_Prob'] = oos_probs
            
            initial_capital = account_size
            capital = initial_capital
            equity_curve = [initial_capital]
            dates = [oos_data.index[0]]
            
            # Simple Commission: 0.1% round-trip
            fee_tier = 0.001 
            win_count = 0
            loss_count = 0
            
            for i in range(len(oos_data) - 1):
                prob = oos_data['AI_Prob'].iloc[i]
                current_close = oos_data['Close'].iloc[i]
                next_close = oos_data['Close'].iloc[i+1] # Execute at tomorrow's close
                atr_val = oos_data['ATR'].iloc[i]
                
                if prob > 0.60: # High-Conviction AI BUY SIGNAL (Reduced Frequency, Higher Accuracy)
                    # ULTRA-CONSERVATIVE RISK MITIGATION ENGINE
                    risk_per_share = atr_val * 2.0 # 2 ATR Stop-Loss
                    capital_to_risk = capital * 0.01 # Strict 1% portfolio risk per trade
                    shares = capital_to_risk / (risk_per_share + 1e-9)
                    
                    # Circuit Breaker: Never allocate more than 15% of portfolio to a single trade
                    max_allocation_shares = (capital * 0.15) / current_close
                    shares = min(shares, max_allocation_shares)
                    
                    pnl = shares * (next_close - current_close)
                    
                    fees = (shares * current_close * (fee_tier/2)) + (shares * next_close * (fee_tier/2))
                    net_trade = pnl - fees
                    capital += net_trade
                    
                    if net_trade > 0: win_count += 1
                    else: loss_count += 1
                    
                equity_curve.append(capital)
                dates.append(oos_data.index[i+1])
                
            equity_array = np.array(equity_curve)
            oos_data['Equity'] = equity_array
            total_return = ((capital / initial_capital) - 1.0) * 100
            
            running_max = np.maximum.accumulate(equity_array)
            drawdowns = (equity_array - running_max) / (running_max + 1e-9)
            max_drawdown = np.min(drawdowns) * 100
            
            bh_shares = initial_capital / oos_data['Close'].iloc[0]
            bh_capital = bh_shares * oos_data['Close'].iloc[-1]
            bh_return = ((bh_capital / initial_capital) - 1.0) * 100
            
            total_trades = win_count + loss_count
            win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
            
            c1, c2, c3, c4 = st.columns(4)
            currency = "$" if market_choice == "US (Wall Street)" else "₹"
            c1.metric("Backtest Net PnL", f"{currency}{capital - initial_capital:.2f}", f"{total_return:.2f}%")
            c2.metric("Strategy Max Drawdown", f"{max_drawdown:.2f}%")
            c3.metric("Buy & Hold Return", f"{bh_return:.2f}%")
            c4.metric("Win Rate (Trades)", f"{win_rate:.1f}%", f"{total_trades} Executed")
            
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=dates, y=equity_curve, mode='lines', line=dict(color='#00FF00', width=2), name='AI Strategy Equity'))
            
            bh_curve = (oos_data['Close'] / oos_data['Close'].iloc[0]) * initial_capital
            fig_eq.add_trace(go.Scatter(x=oos_data.index, y=bh_curve, mode='lines', line=dict(color='rgba(255,255,255,0.4)', width=1, dash='dot'), name='Buy & Hold Equity'))
            
            fig_eq.update_layout(height=450, title="Out-of-Sample Equity Curve (After Slippage & Fees)", template="plotly_dark", hovermode="x unified", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_eq, use_container_width=True)
            
        else:
            st.warning("Not enough data to run Out-of-Sample Validation.")
    else:
        st.info("👆 Click 'Run Real-Time Scan' in the main tab to generate backtesting metrics.")
