import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import random

st.set_page_config(page_title="Evolutionary AI Trader", page_icon="📈", layout="centered")
st.title("📈 Evolutionary AI Trading Dashboard")
st.caption("4H Macro + 1H EMA/ADX Sniper Engine • EUR/USD & Gold")

balance = st.number_input("Account Balance ($)", min_value=50.0, value=1000.0, step=50.0)
risk_pct = st.slider("Risk Per Trade (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5) / 100.0
gold_offset = st.number_input("Gold Spot Offset (XAUUSD minus GC=F)", value=0.0, step=1.0)

class EliteBot:
    def __init__(self, bot_id, capital, fast_ema=None, adx_min=None, rr_target=None, stop_mult=None):
        self.bot_id = bot_id
        self.capital = capital
        self.fast_ema = fast_ema or random.choice((13, 20, 34))
        self.adx_min = adx_min or random.choice((18, 22, 26))
        self.rr_target = rr_target or random.choice((2.0, 2.5, 3.0))
        self.stop_mult = stop_mult or random.choice((1.5, 2.0, 2.5))
        self.position = None
        self.stop_price = 0.0
        self.tp_price = 0.0
        self.dollar_risk = 0.0

@st.cache_data(ttl=900)
def get_mtf_data(ticker):
    df = yf.download(ticker, period="45d", interval="1h", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df_4h = df.resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    df_4h["EMA_50_4H"] = df_4h["Close"].ewm(span=50, adjust=False).mean()
    df["MACRO_BULL"] = (df_4h["Close"] > df_4h["EMA_50_4H"]).reindex(df.index, method="ffill")

    for span in (13, 20, 34, 50):
        df[f"EMA_{span}"] = df["Close"].ewm(span=span, adjust=False).mean()

    tr = pd.concat((df["High"] - df["Low"], np.abs(df["High"] - df["Close"].shift()), np.abs(df["Low"] - df["Close"].shift())), axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(window=14).mean()

    up_m, dn_m = df["High"].diff(), -df["Low"].diff()
    p_dm = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
    m_dm = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)
    p_di = 100 * pd.Series(p_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / df["ATR_14"]
    m_di = 100 * pd.Series(m_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / df["ATR_14"]
    dx = 100 * np.abs(p_di - m_di) / (p_di + m_di).replace(0, np.nan)
    df["ADX_14"] = dx.ewm(alpha=1/14, adjust=False).mean()
    return df.dropna()

def render_asset_card(name, ticker, offset=0.0):
    df = get_mtf_data(ticker)
    bots = [EliteBot(i, 100.0) for i in range(150)]
    for i in range(1, len(df)):
        ts = df.index[i]
        if not (7 <= (ts.hour if hasattr(ts, "hour") else 12) <= 20): continue
        c_open, c_close = float(df["Open"].iloc[i]), float(df["Close"].iloc[i])
        c_high, c_low = float(df["High"].iloc[i]), float(df["Low"].iloc[i])
        atr, adx = float(df["ATR_14"].iloc[i]), float(df["ADX_14"].iloc[i])
        ema_50, macro_bull = float(df["EMA_50"].iloc[i]), bool(df["MACRO_BULL"].iloc[i])

        for b in bots:
            if b.capital < 20.0: continue
            f_ema = float(df[f"EMA_{b.fast_ema}"].iloc[i])
            if b.position == "BUY":
                if c_low <= b.stop_price: b.capital -= b.dollar_risk; b.position = None
                elif c_high >= b.tp_price: b.capital += b.dollar_risk * b.rr_target; b.position = None
            elif b.position == "SELL":
                if c_high >= b.stop_price: b.capital -= b.dollar_risk; b.position = None
                elif c_low <= b.tp_price: b.capital += b.dollar_risk * b.rr_target; b.position = None
            elif adx >= b.adx_min:
                stop_d = max(atr * b.stop_mult, 1e-5)
                if macro_bull and f_ema > ema_50 and c_low <= f_ema and c_close > f_ema and c_close > c_open:
                    b.position, b.stop_price, b.tp_price = "BUY", c_close - stop_d, c_close + stop_d * b.rr_target
                    b.dollar_risk = b.capital * 0.03
                elif (not macro_bull) and f_ema < ema_50 and c_high >= f_ema and c_close < f_ema and c_close < c_open:
                    b.position, b.stop_price, b.tp_price = "SELL", c_close + stop_d, c_close - stop_d * b.rr_target
                    b.dollar_risk = b.capital * 0.03

    champ = max(bots, key=lambda x: x.capital)
    latest = df.iloc[-1]
    price = float(latest["Close"]) + offset
    f_ema = float(latest[f"EMA_{champ.fast_ema}"]) + offset
    ema_50 = float(latest["EMA_50"]) + offset
    atr, adx = float(latest["ATR_14"]), float(latest["ADX_14"])

    is_gold = ("GC" in ticker)
    dec = 2 if is_gold else 5
    risk_usd = balance * risk_pct
    stop_dist = atr * champ.stop_mult
    lots = max(0.01, round(risk_usd / (stop_dist * (100.0 if is_gold else 100000.0)), 2))
    in_zone = abs(price - f_ema) <= (atr * 0.6)

    st.subheader(f"{name} • Champion DNA: EMA {champ.fast_ema}/50 | ADX>{champ.adx_min}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Live Price", f"{price:.{dec}f}", "🔥 STRIKE ZONE" if in_zone else "Wait Pullback")
    c2.metric("Entry Zone", f"{f_ema:.{dec}f}")
    c3.metric("Lot Size (2%)", f"{lots:.2f} lots", f"${risk_usd:.2f} risk")

    if latest["MACRO_BULL"] and f_ema > ema_50 and adx >= champ.adx_min:
        st.success(f"**BUY SETUP** at `{f_ema:.{dec}f}` | **SL:** `{f_ema - stop_dist:.{dec}f}` | **TP1 (1.5R):** `{f_ema + stop_dist*1.5:.{dec}f}` | **TP2 (3R):** `{f_ema + stop_dist*3.0:.{dec}f}`")
    elif (not latest["MACRO_BULL"]) and f_ema < ema_50 and adx >= champ.adx_min:
        st.error(f"**SELL SETUP** at `{f_ema:.{dec}f}` | **SL:** `{f_ema + stop_dist:.{dec}f}` | **TP1 (1.5R):** `{f_ema - stop_dist*1.5:.{dec}f}` | **TP2 (3R):** `{f_ema - stop_dist*3.0:.{dec}f}`")
    else:
        st.info("**NO TRADE** — Waiting for 4H + 1H Trend & ADX alignment.")
    st.divider()

if st.button("🔄 Refresh Live Market Scan", use_container_width=True):
    st.cache_data.clear()

render_asset_card("EUR/USD", "EURUSD=X", 0.0)
render_asset_card("GOLD (XAU/USD)", "GC=F", gold_offset)
