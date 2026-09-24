
import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="All-in-One Delta V1", layout="wide")

BASE = "https://api.india.delta.exchange"
HEADERS = {"Accept": "application/json"}

st.title("All-in-One Delta V1 — Multi-Asset Backtester")
st.caption("Backtest only • No live orders • Delta Exchange India public market data")

@st.cache_data(ttl=1800)
def get_products():
    """Load the live product catalogue using Delta's cursor pagination."""
    rows = []
    after = None

    for _ in range(20):
        params = {"page_size": 100}
        if after:
            params["after"] = after

        r = requests.get(
            f"{BASE}/v2/products",
            params=params,
            headers=HEADERS,
            timeout=25,
        )
        r.raise_for_status()
        payload = r.json()

        rows.extend(payload.get("result", []))
        after = payload.get("meta", {}).get("after")

        if not after:
            break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).drop_duplicates("symbol")

    # Keep only live perpetual/futures products locally.
    if "state" in df.columns:
        df = df[df["state"].astype(str).str.lower().eq("live")]

    if "contract_type" in df.columns:
        allowed = {"perpetual_futures", "futures"}
        df = df[df["contract_type"].astype(str).str.lower().isin(allowed)]

    return df.sort_values("symbol").reset_index(drop=True)


@st.cache_data(ttl=1800)
def get_candles(symbol, resolution, start_ts, end_ts):
    """Download candles in <=1900-bar chunks because Delta caps each response."""
    step = {
        "1m": 60,
        "3m": 180,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "2h": 7200,
        "4h": 14400,
        "6h": 21600,
        "1d": 86400,
        "1w": 604800,
    }[resolution]

    out = []
    cur = int(start_ts)
    end = int(end_ts)
    chunk = step * 1900

    while cur < end:
        nxt = min(cur + chunk, end)

        params = {
            "resolution": resolution,
            "symbol": symbol,
            "start": cur,
            "end": nxt,
        }

        r = requests.get(
            f"{BASE}/v2/history/candles",
            params=params,
            headers=HEADERS,
            timeout=25,
        )
        r.raise_for_status()

        rows = r.json().get("result", [])
        if rows:
            out.extend(rows)

        cur = nxt + step

        # Be polite to the public endpoint.
        time.sleep(0.05)

    if not out:
        return pd.DataFrame()

    df = pd.DataFrame(out)

    required = {"time", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return (
        df.dropna(subset=["open", "high", "low", "close"])
          .drop_duplicates("time")
          .sort_values("time")
          .reset_index(drop=True)
    )


def add_indicators(df, atr_len=20, factor=2.0):
    x = df.copy()

    prev_close = x["close"].shift(1)
    tr = pd.concat(
        [
            x["high"] - x["low"],
            (x["high"] - prev_close).abs(),
            (x["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    x["atr"] = tr.rolling(atr_len).mean()

    hl2 = (x["high"] + x["low"]) / 2
    basic_upper = hl2 + factor * x["atr"]
    basic_lower = hl2 - factor * x["atr"]

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()

    direction = np.zeros(len(x))
    supertrend = np.full(len(x), np.nan)

    for i in range(1, len(x)):
        if pd.isna(x["atr"].iloc[i]):
            continue

        if x["close"].iloc[i - 1] <= final_upper.iloc[i - 1]:
            final_upper.iloc[i] = min(
                basic_upper.iloc[i], final_upper.iloc[i - 1]
            )
        else:
            final_upper.iloc[i] = basic_upper.iloc[i]

        if x["close"].iloc[i - 1] >= final_lower.iloc[i - 1]:
            final_lower.iloc[i] = max(
                basic_lower.iloc[i], final_lower.iloc[i - 1]
            )
        else:
            final_lower.iloc[i] = basic_lower.iloc[i]

        if direction[i - 1] <= 0:
            direction[i] = (
                1 if x["close"].iloc[i] > final_upper.iloc[i] else -1
            )
        else:
            direction[i] = (
                -1 if x["close"].iloc[i] < final_lower.iloc[i] else 1
            )

        supertrend[i] = (
            final_lower.iloc[i]
            if direction[i] == 1
            else final_upper.iloc[i]
        )

    x["supertrend"] = supertrend
    x["st_dir"] = direction

    ema12 = x["close"].ewm(span=12, adjust=False).mean()
    ema26 = x["close"].ewm(span=26, adjust=False).mean()

    x["macd"] = ema12 - ema26
    x["macd_signal"] = x["macd"].ewm(span=9, adjust=False).mean()

    x["vol_ma"] = x["volume"].rolling(20).mean()

    # Mechanical liquidity/structure proxy:
    # breakout of the previous 20-bar high/low.
    x["swing_high"] = x["high"].shift(1).rolling(20).max()
    x["swing_low"] = x["low"].shift(1).rolling(20).min()
    x["long_break"] = x["close"] > x["swing_high"]
    x["short_break"] = x["close"] < x["swing_low"]

    return x


def run_backtest(
    df,
    symbol,
    contract_value,
    balance,
    risk_pct,
    rr,
    fee_pct,
    slippage_pct,
    min_score,
    atr_mult,
):
    """
    V1 assumes a linear/vanilla contract.

    PnL per contract ~= price move * contract_value.
    Delta's product metadata supplies contract_value for each product.
    """
    cash = float(balance)
    pos = None
    trades = []

    peak = cash
    max_dd = 0.0

    contract_value = float(contract_value or 1.0)
    if contract_value <= 0:
        contract_value = 1.0

    for i in range(50, len(df) - 1):
        bar = df.iloc[i]

        if pos is None:
            long_score = (
                int(bar.st_dir == 1)
                + int(bar.macd > bar.macd_signal)
                + int(bar.volume > bar.vol_ma)
                + int(bar.long_break)
            )

            short_score = (
                int(bar.st_dir == -1)
                + int(bar.macd < bar.macd_signal)
                + int(bar.volume > bar.vol_ma)
                + int(bar.short_break)
            )

            if long_score >= min_score and long_score > short_score:
                side = 1
                score = long_score
            elif short_score >= min_score and short_score > long_score:
                side = -1
                score = short_score
            else:
                side = 0
                score = 0

            if side and pd.notna(bar.atr) and bar.atr > 0:
                entry = (
                    df.iloc[i + 1]["open"]
                    * (1 + slippage_pct / 100 * side)
                )

                sl_distance = float(bar.atr) * atr_mult
                sl = entry - side * sl_distance
                tp = entry + side * sl_distance * rr

                risk_cash = max(cash, 0) * risk_pct / 100

                # Risk per contract = SL distance × contract value.
                risk_per_contract = sl_distance * contract_value

                if risk_per_contract <= 0:
                    continue

                contracts = int(risk_cash / risk_per_contract)

                # Avoid fabricating fractional contracts.
                if contracts < 1:
                    continue

                pos = {
                    "entry_i": i + 1,
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "side": side,
                    "contracts": contracts,
                    "signal_score": score,
                }

        else:
            hit_sl = (
                bar.low <= pos["sl"]
                if pos["side"] == 1
                else bar.high >= pos["sl"]
            )

            hit_tp = (
                bar.high >= pos["tp"]
                if pos["side"] == 1
                else bar.low <= pos["tp"]
            )

            exit_price = None
            reason = None

            # Conservative assumption when both are touched in one candle.
            if hit_sl and hit_tp:
                exit_price = pos["sl"]
                reason = "SL"
            elif hit_sl:
                exit_price = pos["sl"]
                reason = "SL"
            elif hit_tp:
                exit_price = pos["tp"]
                reason = "TP"

            if exit_price is not None:
                exit_price *= (
                    1 - slippage_pct / 100 * pos["side"]
                )

                gross = (
                    (exit_price - pos["entry"])
                    * pos["contracts"]
                    * contract_value
                    * pos["side"]
                )

                entry_notional = (
                    pos["entry"]
                    * pos["contracts"]
                    * contract_value
                )
                exit_notional = (
                    exit_price
                    * pos["contracts"]
                    * contract_value
                )

                fees = (
                    abs(entry_notional) + abs(exit_notional)
                ) * fee_pct / 100

                pnl = gross - fees
                cash += pnl

                trades.append(
                    {
                        "symbol": symbol,
                        "entry_time": df.iloc[pos["entry_i"]]["time"],
                        "exit_time": bar["time"],
                        "side": "LONG" if pos["side"] == 1 else "SHORT",
                        "signal_score": pos["signal_score"],
                        "contracts": pos["contracts"],
                        "entry": pos["entry"],
                        "exit": exit_price,
                        "gross_pnl": gross,
                        "fees": fees,
                        "pnl": pnl,
                        "exit_reason": reason,
                    }
                )

                pos = None

        peak = max(peak, cash)
        if peak > 0:
            max_dd = max(max_dd, (peak - cash) / peak * 100)

    return pd.DataFrame(trades), cash, max_dd


# ---------------- Sidebar ----------------
st.sidebar.header("Backtest Settings")

balance = st.sidebar.number_input(
    "Starting balance (₹)",
    min_value=1000.0,
    max_value=10_000_000.0,
    value=10_000.0,
    step=1000.0,
)

risk_pct = st.sidebar.slider(
    "Risk per trade (%)", 0.1, 2.0, 0.5, 0.1
)

rr = st.sidebar.slider(
    "Risk / Reward", 0.5, 4.0, 1.5, 0.1
)

fee_pct = st.sidebar.number_input(
    "Fee per side (%)", 0.0, 1.0, 0.05, 0.01
)

slippage_pct = st.sidebar.number_input(
    "Slippage (%)", 0.0, 1.0, 0.02, 0.01
)

min_score = st.sidebar.slider(
    "Minimum confluence score", 2, 4, 3
)

atr_mult = st.sidebar.slider(
    "ATR stop multiplier", 0.5, 4.0, 1.5, 0.1
)

months = st.sidebar.slider(
    "Months of history", 1, 24, 12
)

st.sidebar.caption(
    "V1 strategy: Supertrend ATR20×2 + MACD + volume + "
    "20-bar structure breakout."
)

# ---------------- Product catalogue ----------------
try:
    products = get_products()

    if products.empty:
        st.error(
            "Delta returned no live futures/perpetual products. "
            "Try refreshing the Streamlit app."
        )
        symbols = []
    else:
        symbols = products["symbol"].dropna().astype(str).tolist()
        st.success(
            f"Loaded {len(symbols)} live Delta futures/perpetual markets."
        )

except Exception as exc:
    st.error(f"Could not load Delta products: {exc}")
    products = pd.DataFrame()
    symbols = []

preferred = [
    s for s in [
        "BTCUSD",
        "ETHUSD",
        "SOLUSD",
        "XRPUSD",
        "XAUTUSD",
        "PAXGUSD",
        "SLVONUSD",
    ]
    if s in symbols
]

selected = st.multiselect(
    "Select Delta markets",
    options=symbols,
    default=preferred,
    help="Markets are loaded from Delta Exchange India's live product catalogue.",
)

st.caption(
    "If the dropdown is empty, use the Streamlit ⋮ menu → Rerun. "
    "The app needs internet access to Delta's public API."
)

run = st.button("🚀 Run Backtest", type="primary")

if run:
    if not selected:
        st.warning("Select at least one market.")
        st.stop()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30 * months)

    all_trades = []
    summary_rows = []

    progress = st.progress(0)

    for n, symbol in enumerate(selected):
        try:
            product_rows = products[
                products["symbol"].astype(str) == symbol
            ]

            if product_rows.empty:
                st.warning(f"{symbol}: product metadata not found.")
                progress.progress((n + 1) / len(selected))
                continue

            product = product_rows.iloc[0]

            contract_value = pd.to_numeric(
                product.get("contract_value", 1),
                errors="coerce",
            )

            if pd.isna(contract_value) or contract_value <= 0:
                contract_value = 1.0

            raw = get_candles(
                symbol,
                "15m",
                int(start.timestamp()),
                int(end.timestamp()),
            )

            if raw.empty:
                st.warning(f"{symbol}: no historical candles returned.")
                progress.progress((n + 1) / len(selected))
                continue

            ind = add_indicators(raw)

            trades, final_balance, max_dd = run_backtest(
                ind,
                symbol,
                contract_value,
                balance,
                risk_pct,
                rr,
                fee_pct,
                slippage_pct,
                min_score,
                atr_mult,
            )

            if not trades.empty:
                all_trades.append(trades)

                wins = trades.loc[trades.pnl > 0, "pnl"]
                losses = trades.loc[trades.pnl < 0, "pnl"]

                profit_factor = (
                    wins.sum() / abs(losses.sum())
                    if len(losses)
                    else np.inf
                )

                summary_rows.append(
                    {
                        "Symbol": symbol,
                        "Trades": len(trades),
                        "Net P&L": trades.pnl.sum(),
                        "Win rate %": len(wins) / len(trades) * 100,
                        "Profit factor": profit_factor,
                        "Max DD %": max_dd,
                        "Final balance": final_balance,
                    }
                )
            else:
                summary_rows.append(
                    {
                        "Symbol": symbol,
                        "Trades": 0,
                        "Net P&L": 0.0,
                        "Win rate %": 0.0,
                        "Profit factor": 0.0,
                        "Max DD %": max_dd,
                        "Final balance": balance,
                    }
                )

            st.write(
                f"**{symbol}** — {len(raw):,} candles • "
                f"{len(trades):,} trades • "
                f"final ₹{final_balance:,.2f} • "
                f"max DD {max_dd:.2f}%"
            )

        except Exception as exc:
            st.warning(f"{symbol}: {exc}")

        progress.progress((n + 1) / len(selected))

    # ---------------- Results ----------------
    st.subheader("Portfolio Results")

    if not all_trades:
        st.info(
            "No completed trades were generated. "
            "Try lowering the minimum confluence score to 2."
        )
        st.stop()

    trades = pd.concat(all_trades, ignore_index=True)

    wins = trades.loc[trades.pnl > 0, "pnl"]
    losses = trades.loc[trades.pnl < 0, "pnl"]

    net_pnl = trades.pnl.sum()
    final_balance = balance + net_pnl
    win_rate = len(wins) / len(trades) * 100
    profit_factor = (
        wins.sum() / abs(losses.sum())
        if len(losses)
        else np.inf
    )

    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    expectancy = trades.pnl.mean()

    # Trade-sequence drawdown.
    equity = balance + trades.pnl.cumsum()
    peak_equity = equity.cummax()
    dd = ((peak_equity - equity) / peak_equity * 100).max()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net P&L", f"₹{net_pnl:,.2f}")
    c2.metric("Final balance", f"₹{final_balance:,.2f}")
    c3.metric("Win rate", f"{win_rate:.2f}%")
    c4.metric("Profit factor", f"{profit_factor:.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Trades", f"{len(trades):,}")
    c6.metric("Avg win", f"₹{avg_win:,.2f}")
    c7.metric("Avg loss", f"₹{avg_loss:,.2f}")
    c8.metric("Max DD", f"{dd:.2f}%")

    st.subheader("Equity Curve")
    curve = pd.DataFrame(
        {"Equity": equity.values},
        index=pd.to_datetime(trades["exit_time"]),
    )
    st.line_chart(curve)

    st.subheader("Asset Results")
    summary = pd.DataFrame(summary_rows)
    st.dataframe(summary, use_container_width=True)

    st.subheader("Trade Log")
    st.dataframe(
        trades.sort_values("exit_time", ascending=False),
        use_container_width=True,
    )

    st.download_button(
        "⬇️ Download trades CSV",
        data=trades.to_csv(index=False),
        file_name="all_in_one_delta_v1_trades.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Research software only. This V1 is an independently designed strategy "
    "and is not a reproduction of MirrorPip's proprietary All-in-One Delta rules."
)
