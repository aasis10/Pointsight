############################################################
#                 POINTSIGHT STREAMLIT APP
#                   REAL-LTP + SHARED STATE
#
#   - JSON instruments (NSE + MCX)
#   - Shared positions via pointsight_state.json
#   - Upstox LTP via token_store.json or env var
#   - Auto-refresh, multi-positions, search, MFE/MAE
############################################################

import streamlit as st
from streamlit_autorefresh import st_autorefresh

import time
import csv
import os
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(page_title="PointSight", layout="wide")

# ----------------------------------------------------------
# BASE PATHS
# ----------------------------------------------------------
BASE_DIR = Path(__file__).parent

# Folder where your JSON instruments live (inside project)
JSON_FOLDER = BASE_DIR / "Bod_Json"
NSE_JSON = JSON_FOLDER / "NSE.json"   # make sure this file exists
MCX_JSON = JSON_FOLDER / "MCX.json"   # make sure this file exists

# Files created/used by the app
SNAPSHOT_FILE = BASE_DIR / "pointsight_snapshots.csv"
EOD_FILE = BASE_DIR / "pointsight_eod_master.csv"
STATE_FILE = BASE_DIR / "pointsight_state.json"     # 🔥 shared positions here
TOKEN_STORE_FILE = BASE_DIR / "token_store.json"

# Timing
SNAPSHOT_INTERVAL_SEC = 300    # default 5 min, change if you want
REFRESH_INTERVAL_MS = 2000     # UI auto-refresh every 2 sec


# ==========================================================
# TOKEN HELPERS
# ==========================================================

def load_token_store():
    p = Path(TOKEN_STORE_FILE)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}


def save_token_store(d: dict):
    with open(TOKEN_STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)


def set_access_token(token: str):
    store = load_token_store()
    store["access_token"] = token.strip()
    save_token_store(store)


def get_access_token() -> str | None:
    # 1) Prefer env var for cloud
    env_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if env_token:
        return env_token.strip()

    # 2) Fallback to local token_store.json (for laptop use)
    store = load_token_store()
    return store.get("access_token")


# ==========================================================
# SHARED STATE HELPERS (positions + last_snapshot)
# ==========================================================

def load_state():
    """
    Global shared state for ALL users/devices:
      {
        "positions": [...],
        "last_snapshot_ts": float
      }
    """
    if not STATE_FILE.exists():
        return {"positions": [], "last_snapshot_ts": 0.0}
    try:
        with STATE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if "positions" not in data:
                data["positions"] = []
            if "last_snapshot_ts" not in data:
                data["last_snapshot_ts"] = 0.0
            return data
    except:
        return {"positions": [], "last_snapshot_ts": 0.0}


def save_state(state: dict):
    """
    Atomic write so we don't corrupt file mid-write.
    """
    tmp = STATE_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


# ==========================================================
# BOD JSON LOAD (NSE + MCX)  (cached)
# ==========================================================

@st.cache_data
def load_instruments():
    def load_one(path: Path) -> pd.DataFrame:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        # Upstox may give list or {"data": [..]}
        if isinstance(raw, list):
            df = pd.DataFrame(raw)
        elif isinstance(raw, dict) and isinstance(raw.get("data"), list):
            df = pd.DataFrame(raw["data"])
        else:
            df = pd.DataFrame(raw)

        # normalize column names
        rename_map = {
            "trading_symbol": "tradingsymbol",
            "tradingSymbol": "tradingsymbol",
            "strike_price": "strike",
            "strikePrice": "strike",
            "optionType": "option_type",
            "instrumentType": "instrument_type",
        }
        df = df.rename(columns=rename_map)

        # ensure core cols exist
        for col in ["tradingsymbol", "name", "instrument_key", "exchange", "segment"]:
            if col not in df.columns:
                df[col] = None

        return df

    df_nse = load_one(NSE_JSON)
    df_mcx = load_one(MCX_JSON)
    df_all = pd.concat([df_nse, df_mcx], ignore_index=True)
    return df_all


df_all_instruments = load_instruments()


# ==========================================================
# SEARCH HELPERS
# ==========================================================

def search_instruments(query: str, top: int = 25) -> pd.DataFrame:
    """
    Case-insensitive, multi-word, partial search over
    tradingsymbol + name. Works for:
      - "nifty 25500 ce"
      - "banknifty 48500 pe"
      - "gold dec fut"
      - "reliance"
    """
    q_raw = (query or "").strip()
    if not q_raw:
        return pd.DataFrame()

    q = q_raw.lower()
    words = q.split()

    df = df_all_instruments.copy()

    # Prepare lower-case helpers
    df["__ts_lower"] = df["tradingsymbol"].astype(str).str.lower()
    df["__name_lower"] = df["name"].astype(str).str.lower()

    # Word-by-word filter
    for w in words:
        df = df[
            df["__ts_lower"].str.contains(w, na=False)
            | df["__name_lower"].str.contains(w, na=False)
        ]
        if df.empty:
            break

    # If nothing found at all, try simple contains on full query
    if df.empty:
        df = df_all_instruments.copy()
        df["__ts_lower"] = df["tradingsymbol"].astype(str).str.lower()
        df["__name_lower"] = df["name"].astype(str).str.lower()
        df = df[
            df["__ts_lower"].str.contains(q, na=False)
            | df["__name_lower"].str.contains(q, na=False)
        ]

    if df.empty:
        return pd.DataFrame()

    # remove helper cols
    df = df.drop(columns=["__ts_lower", "__name_lower"], errors="ignore")

    return df.head(top)


# ==========================================================
# LTP FETCH (Upstox v3) WITH SIMPLE BACKOFF
# ==========================================================

def request_with_backoff(url: str, max_retries: int = 3, timeout: int = 5):
    token = get_access_token()
    if not token:
        return None

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
        except requests.RequestException:
            if attempt == max_retries:
                return None
            time.sleep(1.5 * attempt)
            continue

        if r.status_code == 200:
            return r
        elif r.status_code in (429, 500, 502, 503, 504):
            if attempt == max_retries:
                return None
            time.sleep(1.5 * attempt)
            continue
        else:
            # 401 or other error → just return
            return None

    return None


def fetch_upstox_ltp(instrument_key: str):
    """
    Calls Upstox v3 market-quote/ltp for a single instrument_key.
    Returns float LTP or None.
    """
    if not instrument_key:
        return None

    url = f"https://api.upstox.com/v3/market-quote/ltp?instrument_key={instrument_key}"
    r = request_with_backoff(url)
    if r is None:
        return None

    try:
        j = r.json()
    except:
        return None

    data = j.get("data") or {}
    # data is usually: { "NSE_FO:NIFTY25NOV25500CE": {...} }
    for _, rec in data.items():
        for f in ("last_price", "ltp", "cp"):
            if f in rec and rec[f] is not None:
                try:
                    return float(rec[f])
                except:
                    pass
    return None


# ==========================================================
# SESSION STATE (ONLY FOR UI STUFF, NOT POSITIONS)
# ==========================================================

if "search_query" not in st.session_state:
    st.session_state.search_query = ""

if "search_results" not in st.session_state:
    st.session_state.search_results = pd.DataFrame()

if "selected_instrument_key" not in st.session_state:
    st.session_state.selected_instrument_key = ""


# ==========================================================
# AUTO-REFRESH (every 2 seconds)
# ==========================================================

st_autorefresh(interval=REFRESH_INTERVAL_MS, key="ps_refresh")


# ==========================================================
# LOAD GLOBAL STATE (SHARED POSITIONS)
# ==========================================================

state = load_state()
positions = state.get("positions", [])
last_snapshot_ts = state.get("last_snapshot_ts", 0.0)

# ensure types
if not isinstance(positions, list):
    positions = []
if not isinstance(last_snapshot_ts, (int, float)):
    last_snapshot_ts = 0.0


# ==========================================================
# SIDEBAR: TOKEN + MANUAL EOD
# ==========================================================

st.sidebar.title("⚙️ Settings")

current_token = get_access_token() or ""
token_input = st.sidebar.text_area(
    "Upstox Access Token (env var overrides this on cloud)",
    value=current_token,
    height=80,
)
if st.sidebar.button("Save token locally"):
    if token_input.strip():
        set_access_token(token_input.strip())
        st.sidebar.success("Token saved to token_store.json")
    else:
        st.sidebar.warning("Empty token not saved.")

if not get_access_token():
    st.sidebar.warning("No access token set. LTP will not work.")

# Manual EOD save button
if st.sidebar.button("Save EOD now"):
    if not positions:
        st.sidebar.info("No open positions to save.")
    else:
        summary = []
        for p in positions:
            pts = p.get("points")
            pl_points = pts if pts is not None else None
            summary.append({
                "pos_id": p.get("id"),
                "name": p.get("name"),
                "instrument_key": p.get("instrument_key"),
                "entry": p.get("entry"),
                "final_ltp": p.get("ltp"),
                "final_points": pts,
                "pl_points": pl_points,
                "side": p.get("side"),
                "mfe": p.get("mfe"),
                "mae": p.get("mae"),
                "open_time": datetime.fromtimestamp(p.get("open_ts")).strftime("%Y-%m-%d %H:%M:%S"),
                "close_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

        df_eod = pd.DataFrame(summary)
        write_header = (not EOD_FILE.exists()) or (os.path.getsize(EOD_FILE) == 0)
        df_eod.to_csv(EOD_FILE, mode="a", index=False, header=write_header)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        df_eod.to_csv(BASE_DIR / f"pointsight_eod_{ts}.csv", index=False)

        st.sidebar.success(f"EOD saved → {EOD_FILE.name} and pointsight_eod_{ts}.csv")


# ==========================================================
# UI LAYOUT
# ==========================================================

st.title("📈 PointSight – Live Positions (Real LTP, Shared)")

col_main, col_stats = st.columns([2.5, 1.2])


# ==========================================================
# SEARCH UI (LEFT)
# ==========================================================

with col_main:
    with st.expander("🔍 Search Instrument", expanded=True):
        query = st.text_input(
            "Type symbol / strike / name (e.g. 'nifty 25500 ce', 'gold dec fut')",
            value=st.session_state.search_query,
            key="search_box",
        )

        if query != st.session_state.search_query:
            st.session_state.search_query = query
            st.session_state.search_results = search_instruments(query)

        results = st.session_state.search_results

        if results.empty and query.strip():
            st.info("No match found. Try a different search.")
        elif not results.empty:
            st.dataframe(
                results[["tradingsymbol", "instrument_key", "name", "exchange", "segment"]],
                height=220,
                use_container_width=True,
            )
            choices = results["instrument_key"].tolist()
            labels = [
                f"{row['tradingsymbol']} | {row['instrument_key']} | {row['exchange']}"
                for _, row in results.iterrows()
            ]
            selected_idx = st.selectbox(
                "Select exact instrument",
                options=list(range(len(choices))),
                format_func=lambda i: labels[i],
                key="instrument_select",
            )
            if st.button("Use Selected Instrument", key="use_selected_btn"):
                st.session_state.selected_instrument_key = choices[selected_idx]
                st.success(f"Selected → {choices[selected_idx]}")


# ==========================================================
# ADD / CLOSE POSITIONS (LEFT)
# ==========================================================

with col_main:
    st.markdown("---")
    with st.expander("➕ Add / ❌ Close Positions", expanded=True):

        # ---------- ADD ----------
        st.markdown("### ➕ Add Position")
        with st.form("add_pos_form"):
            name = st.text_input("Display Name (e.g. NIFTY CALL)", key="add_name")
            default_ikey = st.session_state.selected_instrument_key or ""
            ikey = st.text_input("Instrument Key", value=default_ikey, key="add_ikey")
            entry = st.number_input("Entry Price", step=0.05, format="%.2f", key="add_entry")
            side = st.selectbox("Side", ["buy", "sell"], key="add_side")
            submitted = st.form_submit_button("Add Position")

            if submitted:
                if not ikey.strip():
                    st.error("Instrument Key is required.")
                else:
                    new_id = (max([p.get("id", 0) for p in positions]) + 1) if positions else 1
                    pos = {
                        "id": new_id,
                        "name": name.strip() or ikey,
                        "instrument_key": ikey.strip(),
                        "entry": float(entry),
                        "side": side,
                        "mfe": -10**9,
                        "mae": 10**9,
                        "ltp": None,
                        "points": None,
                        "prev_ltp": None,
                        "prev_points": None,
                        "open_ts": time.time(),
                    }
                    positions.append(pos)
                    st.success(f"Position added → {pos['name']}")

        # ---------- CLOSE ----------
        st.markdown("### ❌ Close Position")
        if positions:
            pos_map = {
                f"{p['id']} – {p['name']} ({p['instrument_key']})": p["id"]
                for p in positions
            }
            choice_label = st.selectbox(
                "Select Position to close",
                list(pos_map.keys()),
                key="close_select",
            )
            if st.button("Close Selected Position", key="close_btn"):
                pid = pos_map[choice_label]
                positions = [p for p in positions if p["id"] != pid]
                st.success(f"Closed position id={pid}")
        else:
            st.info("No open positions to close.")

    st.markdown("---")
    st.subheader("📡 Live Positions")


# ==========================================================
# UPDATE LTP + POINTS + DELTAS + MFE/MAE
# ==========================================================

rows = []

for p in positions:
    prev_ltp = p.get("ltp")
    prev_points = p.get("points")

    ltp = fetch_upstox_ltp(p["instrument_key"])
    p["ltp"] = ltp

    if ltp is not None:
        if p["side"] == "buy":
            pts = ltp - p["entry"]
        else:
            pts = p["entry"] - ltp
        p["points"] = pts
    else:
        pts = None
        p["points"] = None

    if prev_ltp is not None and ltp is not None:
        delta_price = ltp - prev_ltp
    else:
        delta_price = 0.0

    if prev_points is not None and pts is not None:
        delta_points = pts - prev_points
    else:
        delta_points = 0.0

    p["prev_ltp"] = prev_ltp
    p["prev_points"] = prev_points

    # update mfe/mae
    if pts is not None:
        if pts > p["mfe"]:
            p["mfe"] = pts
        if pts < p["mae"]:
            p["mae"] = pts

    rows.append({
        "ID": p["id"],
        "Name": p["name"],
        "Instrument": p["instrument_key"],
        "Side": p["side"],
        "Entry": p["entry"],
        "LTP": p["ltp"],
        "ΔPrice": delta_price,
        "Points": p["points"],
        "ΔPoints": delta_points,
        "MFE": p["mfe"],
        "MAE": p["mae"],
    })


# ==========================================================
# DISPLAY TABLE + PORTFOLIO STATS
# ==========================================================

with col_main:
    if rows:
        df = pd.DataFrame(rows)

        def color_points(val):
            try:
                v = float(val)
            except:
                return ""
            if v > 0:
                return "color: green; font-weight: 600"
            if v < 0:
                return "color: red; font-weight: 600"
            return ""

        def color_delta(val):
            try:
                v = float(val)
            except:
                return ""
            if v > 0:
                return "color: green;"
            if v < 0:
                return "color: red;"
            return ""

        styler = df.style
        styler = styler.format({
            "Entry": "{:.2f}",
            "LTP": "{:.2f}",
            "ΔPrice": "{:+.2f}",
            "Points": "{:.2f}",
            "ΔPoints": "{:+.2f}",
            "MFE": "{:.2f}",
            "MAE": "{:.2f}",
        })
        styler = styler.applymap(color_points, subset=["Points"])
        styler = styler.applymap(color_delta, subset=["ΔPrice", "ΔPoints"])

        st.dataframe(styler, height=360, use_container_width=True)
    else:
        st.info("Add a position to start tracking.")


with col_stats:
    st.subheader("📊 Portfolio Stats")
    if not rows:
        st.info("No open positions.")
    else:
        total_points = sum((p.get("points") or 0.0) for p in positions)

        best = max(
            positions,
            key=lambda p: p.get("points") if p.get("points") is not None else -10**9
        )
        worst = min(
            positions,
            key=lambda p: p.get("points") if p.get("points") is not None else 10**9
        )

        st.metric("Total Points (sum)", f"{total_points:.2f}")
        st.metric(
            "Best Position (points)",
            f"{(best.get('points') or 0):.2f}",
            help=f"{best.get('name')} ({best.get('instrument_key')})"
        )
        st.metric(
            "Worst Position (points)",
            f"{(worst.get('points') or 0):.2f}",
            help=f"{worst.get('name']} ({worst.get('instrument_key')})"
        )


# ==========================================================
# SNAPSHOT SAVE EVERY N SECONDS (GLOBAL, SHARED)
# ==========================================================

now_ts = time.time()
if now_ts - last_snapshot_ts >= SNAPSHOT_INTERVAL_SEC:
    SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not SNAPSHOT_FILE.exists()) or (os.path.getsize(SNAPSHOT_FILE) == 0)

    with SNAPSHOT_FILE.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "timestamp", "id", "name", "instrument_key",
                "entry", "ltp", "side",
                "points", "mfe", "mae"
            ])

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for p in positions:
            w.writerow([
                ts,
                p["id"], p["name"], p["instrument_key"],
                p["entry"], p["ltp"], p["side"],
                p["points"], p["mfe"], p["mae"]
            ])

    last_snapshot_ts = now_ts
    st.toast("Snapshot saved.", icon="💾")


# ==========================================================
# SAVE UPDATED GLOBAL STATE (POSITIONS + SNAPSHOT TS)
# ==========================================================

state["positions"] = positions
state["last_snapshot_ts"] = last_snapshot_ts
save_state(state)
