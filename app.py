import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="All-in-One Delta V1", layout="wide")
st.title("All-in-One Delta V1 — Multi-Asset Backtester")
st.caption("Backtest only • No live orders • Delta Exchange public market data")

BASE = "https://api.india.delta.exchange"

@st.cache_data(ttl=3600)
def get_products():
    r = requests.get(f"{BASE}/v2/products", params={"page_size": 100}, timeout=20)
    r.raise_for_status()
    data = r.json()["result"]
    return pd.DataFrame(data)

@st.cache_data(ttl=3600)
def get_candles(symbol, resolution, start_ts, end_ts):
    # Delta returns max 2000 candles/request. Chunk by 2000 bars.
    step = {"1m":60, "3m":180, "5m":300, "15m":900, "30m":1800,
            "1h":3600, "2h":7200, "4h":14400, "1d":86400}[resolution]
    out = []
    cur = int(start_ts)
    end = int(end_ts)
    chunk = step * 1900
    while cur < end:
        nxt = min(cur + chunk, end)
        p = {"resolution": resolution, "symbol": symbol,
             "start": cur, "end": nxt}
        r = requests.get(f"{BASE}/v2/history/candles", params=p, timeout=20)
        r.raise_for_status()
        rows = r.json().get("result", [])
        if rows:
            out.extend(rows)
        cur = nxt + step
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    # API commonly returns time/open/high/low/close/volume.
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    for c in ["open","high","low","close","volume"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop_duplicates("time").sort_values("time").reset_index(drop=True)

def indicators(df, atr_len=20, factor=2.0):
    x = df.copy()
    prev_close = x.close.shift(1)
    tr = pd.concat([(x.high-x.low), (x.high-prev_close).abs(),
                    (x.low-prev_close).abs()], axis=1).max(axis=1)
    x["atr"] = tr.rolling(atr_len).mean()

    hl2 = (x.high+x.low)/2
    upper = hl2 + factor*x.atr
    lower = hl2 - factor*x.atr
    st = np.full(len(x), np.nan)
    direction = np.zeros(len(x))
    for i in range(1, len(x)):
        if np.isnan(x.atr.iloc[i]):
            continue
        pu = upper.iloc[i-1] if not np.isnan(upper.iloc[i-1]) else upper.iloc[i]
        pl = lower.iloc[i-1] if not np.isnan(lower.iloc[i-1]) else lower.iloc[i]
        upper.iloc[i] = min(upper.iloc[i], pu) if x.close.iloc[i-1] > pu else upper.iloc[i]
        lower.iloc[i] = max(lower.iloc[i], pl) if x.close.iloc[i-1] < pl else lower.iloc[i]
        if direction[i-1] <= 0:
            direction[i] = 1 if x.close.iloc[i] > upper.iloc[i] else -1
        else:
            direction[i] = -1 if x.close.iloc[i] < lower.iloc[i] else 1
        st[i] = lower.iloc[i] if direction[i] == 1 else upper.iloc[i]
    x["supertrend"] = st
    x["st_dir"] = direction

    e12 = x.close.ewm(span=12, adjust=False).mean()
    e26 = x.close.ewm(span=26, adjust=False).mean()
    x["macd"] = e12-e26
    x["macd_signal"] = x.macd.ewm(span=9, adjust=False).mean()
    x["vol_ma"] = x.volume.rolling(20).mean()

    # Simple structure proxy: recent high/low breakout.
    x["swing_high"] = x.high.shift(1).rolling(20).max()
    x["swing_low"] = x.low.shift(1).rolling(20).min()
    x["long_break"] = x.close > x.swing_high
    x["short_break"] = x.close < x.swing_low
    return x

def backtest(df, balance, risk_pct, rr, fee_pct, slip_pct, min_score):
    cash = balance
    equity = balance
    pos = None
    trades = []
    peak = balance
    max_dd = 0

    for i in range(50, len(df)-1):
        r = df.iloc[i]
        if pos is None:
            score_l = int(r.st_dir == 1) + int(r.macd > r.macd_signal) + int(r.volume > r.vol_ma) + int(r.long_break)
            score_s = int(r.st_dir == -1) + int(r.macd < r.macd_signal) + int(r.volume > r.vol_ma) + int(r.short_break)
            side = 1 if score_l >= min_score and score_l > score_s else (-1 if score_s >= min_score and score_s > score_l else 0)
            if side and r.atr > 0:
                entry = df.iloc[i+1].open * (1 + slip_pct/100*side)
                sl_dist = r.atr * 1.5
                sl = entry - side*sl_dist
                tp = entry + side*sl_dist*rr
                risk_cash = cash * risk_pct/100
                qty = risk_cash / sl_dist
                pos = dict(entry_i=i+1, entry=entry, sl=sl, tp=tp, side=side, qty=qty)
        else:
            bar = r
            hit_sl = (bar.low <= pos["sl"]) if pos["side"] == 1 else (bar.high >= pos["sl"])
            hit_tp = (bar.high >= pos["tp"]) if pos["side"] == 1 else (bar.low <= pos["tp"])
            exit_price = None
            reason = None
            if hit_sl and hit_tp:
                # Conservative: assume SL hit first.
                exit_price, reason = pos["sl"], "SL"
            elif hit_sl:
                exit_price, reason = pos["sl"], "SL"
            elif hit_tp:
                exit_price, reason = pos["tp"], "TP"
            if exit_price is not None:
                exit_price *= (1 - slip_pct/100*pos["side"])
                gross = (exit_price-pos["entry"])*pos["qty"]*pos["side"]
                fees = (abs(pos["entry"]*pos["qty"])+abs(exit_price*pos["qty"])) * fee_pct/100
                pnl = gross-fees
                cash += pnl
                trades.append({
                    "entry_time": df.iloc[pos["entry_i"]].time,
                    "exit_time": bar.time, "side":"LONG" if pos["side"]==1 else "SHORT",
                    "entry":pos["entry"], "exit":exit_price, "qty":pos["qty"],
                    "pnl":pnl, "reason":reason
                })
                pos = None
        peak = max(peak, cash)
        max_dd = max(max_dd, (peak-cash)/peak*100)
    return pd.DataFrame(trades), cash, max_dd

st.sidebar.header("Backtest Settings")
balance = st.sidebar.number_input("Starting balance", 1000.0, 10000000.0, 10000.0, step=1000.0)
risk = st.sidebar.slider("Risk per trade (%)", 0.1, 2.0, 0.5, 0.1)
rr = st.sidebar.slider("Risk / Reward", 0.5, 4.0, 1.5, 0.1)
fee = st.sidebar.number_input("Fee per side (%)", 0.0, 1.0, 0.05, 0.01)
slip = st.sidebar.number_input("Slippage (%)", 0.0, 1.0, 0.02, 0.01)
score = st.sidebar.slider("Minimum confluence score", 2, 4, 3)
months = st.sidebar.slider("Months of history", 1, 24, 12)

try:
    products = get_products()
    candidates = products[
        products.get("state","").astype(str).str.lower().eq("live")
    ].copy()
    if "contract_type" in candidates:
        candidates = candidates[candidates.contract_type.astype(str).str.lower().isin(
            ["perpetual_futures","futures","perpetual"]
        )]
    symbols = candidates["symbol"].dropna().tolist()
except Exception as e:
    st.error(f"Could not load Delta products: {e}")
    symbols = []

default = [s for s in ["BTCUSD","ETHUSD","SOLUSD","XAUTUSD","PAXGUSD","SLVONUSD"] if s in symbols]
selected = st.multiselect("Select Delta markets", symbols, default=default)

run = st.button("🚀 Run Backtest", type="primary")

if run:
    if not selected:
        st.warning("Select at least one market.")
    else:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=30*months)
        all_trades = []
        progress = st.progress(0)
        for n, symbol in enumerate(selected):
            try:
                raw = get_candles(symbol, "15m", int(start.timestamp()), int(end.timestamp()))
                if raw.empty:
                    st.warning(f"No candles returned for {symbol}")
                    continue
                ind = indicators(raw)
                trades, final_eq, dd = backtest(ind, balance, risk, rr, fee, slip, score)
                if not trades.empty:
                    trades["symbol"] = symbol
                    all_trades.append(trades)
                st.write(f"**{symbol}** — candles: {len(raw):,}, trades: {len(trades)}, final: ₹{final_eq:,.2f}, max DD: {dd:.2f}%")
            except Exception as e:
                st.warning(f"{symbol}: {e}")
            progress.progress((n+1)/len(selected))

        if all_trades:
            t = pd.concat(all_trades, ignore_index=True)
            wins = t[t.pnl > 0].pnl
            losses = t[t.pnl < 0].pnl
            net = t.pnl.sum()
            pf = wins.sum()/abs(losses.sum()) if len(losses) else np.inf
            winrate = len(wins)/len(t)*100
            st.subheader("Portfolio Results")
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Net P&L", f"₹{net:,.2f}")
            c2.metric("Win rate", f"{winrate:.2f}%")
            c3.metric("Profit factor", f"{pf:.2f}")
            c4.metric("Trades", len(t))
            c5.metric("Avg trade", f"₹{t.pnl.mean():,.2f}")
            st.subheader("Trades")
            st.dataframe(t.sort_values("exit_time", ascending=False), use_container_width=True)
            st.download_button("Download trades CSV", t.to_csv(index=False), "all_in_one_delta_v1_trades.csv", "text/csv")
        else:
            st.info("No completed trades were generated with these settings.")
else:
    st.info("Choose markets and press Run Backtest. This version is simulation-only.")
