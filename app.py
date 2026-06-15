import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Options Pricing Model",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .stApp { background: #0a0f1e; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #0d1428 !important;
    border-right: 1px solid #1e2d50;
  }

  /* Cards */
  .metric-card {
    background: linear-gradient(135deg, #111b35 0%, #0d1428 100%);
    border: 1px solid #1e2d50;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
  }
  .metric-card .label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4a7fc1;
    margin-bottom: 0.3rem;
  }
  .metric-card .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #e8f0ff;
    line-height: 1;
  }
  .metric-card .sub {
    font-size: 0.78rem;
    color: #5a7aab;
    margin-top: 0.25rem;
  }

  /* Model result cards */
  .model-card {
    background: #0d1428;
    border: 1px solid #1e2d50;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
  }
  .model-name {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a7fc1;
    margin-bottom: 0.4rem;
  }
  .call-price {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #34d399;
  }
  .put-price {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #f87171;
  }
  .price-label {
    font-size: 0.72rem;
    color: #5a7aab;
    margin-right: 0.5rem;
  }

  /* Section headers */
  .section-header {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4a7fc1;
    border-bottom: 1px solid #1e2d50;
    padding-bottom: 0.4rem;
    margin: 1.2rem 0 0.8rem 0;
  }

  /* Greek badge */
  .greek-badge {
    display: inline-block;
    background: #111b35;
    border: 1px solid #1e2d50;
    border-radius: 6px;
    padding: 0.4rem 0.8rem;
    margin: 0.2rem;
    text-align: center;
  }
  .greek-name { font-size: 0.65rem; color: #5a7aab; letter-spacing: 0.08em; text-transform: uppercase; }
  .greek-val  { font-family: 'JetBrains Mono', monospace; font-size: 0.95rem; font-weight: 600; color: #c8d8f8; }

  h1 { color: #e8f0ff !important; font-weight: 700 !important; }
  h2, h3 { color: #b0c8f0 !important; font-weight: 600 !important; }
  p, li, label { color: #8aa8d4 !important; }
  .stSelectbox label, .stNumberInput label, .stSlider label { color: #8aa8d4 !important; }
  .stTabs [data-baseweb="tab"] { color: #8aa8d4; }
  .stTabs [aria-selected="true"] { color: #e8f0ff !important; }

  /* Info box */
  .info-box {
    background: #0c1830;
    border-left: 3px solid #4a7fc1;
    border-radius: 4px;
    padding: 0.7rem 1rem;
    font-size: 0.82rem;
    color: #8aa8d4 !important;
    margin: 0.5rem 0;
  }
</style>
""", unsafe_allow_html=True)


# ── Helper: fetch spot price ──────────────────────────────────────────────────
@st.cache_data(ttl=60)
def get_spot_price(ticker: str) -> tuple[float, str, str]:
    """Returns (price, exchange, currency)."""
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        price = info.last_price
        if price is None or np.isnan(price):
            hist = t.history(period="1d")
            price = float(hist["Close"].iloc[-1])
        exchange = getattr(info, "exchange", "NSE")
        currency = getattr(info, "currency", "INR")
        return round(float(price), 2), exchange, currency
    except Exception as e:
        raise ValueError(f"Could not fetch price for '{ticker}': {e}")


@st.cache_data(ttl=300)
def get_historical_volatility(ticker: str, window: int = 30) -> float:
    """Annualised historical volatility from log returns."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="6mo")["Close"]
        log_ret = np.log(hist / hist.shift(1)).dropna()
        vol = log_ret.rolling(window).std().iloc[-1] * np.sqrt(252)
        return round(float(vol), 4)
    except:
        return 0.20  # fallback


# ── Pricing models ────────────────────────────────────────────────────────────

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """European option price via Black-Scholes."""
    if T <= 0:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return intrinsic, intrinsic
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return round(price, 4), (d1, d2)


def bs_greeks(S, K, T, r, sigma):
    """Return dict of Greeks (BS model)."""
    if T <= 0:
        return {g: 0.0 for g in ["delta_call","delta_put","gamma","theta_call","theta_put","vega","rho_call","rho_put"]}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    nd1 = norm.pdf(d1)
    Nd1_c = norm.cdf(d1);  Nd1_p = norm.cdf(-d1)
    Nd2_c = norm.cdf(d2);  Nd2_p = norm.cdf(-d2)
    gamma   = nd1 / (S * sigma * np.sqrt(T))
    vega    = S * nd1 * np.sqrt(T) / 100  # per 1% move
    theta_c = (-(S * nd1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * Nd2_c) / 365
    theta_p = (-(S * nd1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * Nd2_p) / 365
    rho_c   = K * T * np.exp(-r * T) * Nd2_c / 100
    rho_p   = -K * T * np.exp(-r * T) * Nd2_p / 100
    return {
        "delta_call": round(Nd1_c, 4),
        "delta_put":  round(-Nd1_p, 4),
        "gamma":      round(gamma, 6),
        "theta_call": round(theta_c, 4),
        "theta_put":  round(theta_p, 4),
        "vega":       round(vega, 4),
        "rho_call":   round(rho_c, 4),
        "rho_put":    round(rho_p, 4),
    }


def monte_carlo(S, K, T, r, sigma, n_sim=50_000, option_type="call", seed=42):
    """Monte Carlo simulation for European option."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_sim)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
    price = np.exp(-r * T) * payoffs.mean()
    se    = payoffs.std() / np.sqrt(n_sim)
    ci    = (round(price - 1.96 * se * np.exp(-r * T), 4),
             round(price + 1.96 * se * np.exp(-r * T), 4))
    return round(price, 4), ci, ST


def binomial_tree(S, K, T, r, sigma, n_steps=200, option_type="call", american=False):
    """Cox-Ross-Rubinstein binomial tree."""
    dt = T / n_steps
    u  = np.exp(sigma * np.sqrt(dt))
    d  = 1 / u
    p  = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Terminal asset prices
    j = np.arange(n_steps + 1)
    ST = S * (u ** (n_steps - j)) * (d ** j)

    # Terminal payoffs
    if option_type == "call":
        V = np.maximum(ST - K, 0)
    else:
        V = np.maximum(K - ST, 0)

    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        ST = S * (u ** (i - np.arange(i + 1))) * (d ** np.arange(i + 1))
        V  = disc * (p * V[:-1] + (1 - p) * V[1:])
        if american:
            if option_type == "call":
                V = np.maximum(V, ST - K)
            else:
                V = np.maximum(V, K - ST)
    return round(V[0], 4)


# ── Popular Indian tickers ────────────────────────────────────────────────────
NIFTY_TICKERS = {
    "NIFTY 50 Index": "^NSEI",
    "BANKNIFTY Index": "^NSEBANK",
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Wipro": "WIPRO.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "SBI": "SBIN.NS",
    "HUL": "HINDUNILVR.NS",
    "Axis Bank": "AXISBANK.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "ITC": "ITC.NS",
    "Custom ticker": None,
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Parameters")
    st.markdown('<div class="section-header">Underlying</div>', unsafe_allow_html=True)

    selected_name = st.selectbox("Stock / Index", list(NIFTY_TICKERS.keys()), index=0)
    if NIFTY_TICKERS[selected_name] is None:
        ticker_input = st.text_input("Enter ticker (e.g. TATAMOTORS.NS)", value="TATAMOTORS.NS")
        ticker = ticker_input.strip().upper()
    else:
        ticker = NIFTY_TICKERS[selected_name]

    fetch_col, _ = st.columns([3, 1])
    fetch_btn = fetch_col.button("Fetch Live Price", use_container_width=True)

    spot_override = st.number_input("Spot Price (S) ₹", min_value=1.0, value=22500.0, step=50.0, format="%.2f")

    st.markdown('<div class="section-header">Option Contract</div>', unsafe_allow_html=True)
    strike = st.number_input("Strike Price (K) ₹", min_value=1.0, value=22500.0, step=50.0, format="%.2f")
    expiry = st.date_input("Expiry Date", value=date.today() + timedelta(days=30),
                           min_value=date.today())
    T_days = (expiry - date.today()).days
    T_years = max(T_days / 365, 1e-6)

    st.markdown('<div class="section-header">Market Parameters</div>', unsafe_allow_html=True)
    risk_free = st.slider("Risk-Free Rate (%) — RBI Repo", 4.0, 10.0, 6.5, 0.25) / 100
    vol_auto  = st.checkbox("Auto-compute Historical Volatility", value=False)
    if vol_auto:
        vol_pct = get_historical_volatility(ticker) * 100
        st.info(f"Historical vol (30d): **{vol_pct:.1f}%**")
    else:
        vol_pct = st.slider("Implied Volatility (%)", 5.0, 100.0, 20.0, 0.5)
    sigma = vol_pct / 100

    st.markdown('<div class="section-header">Model Settings</div>', unsafe_allow_html=True)
    n_sim   = st.select_slider("Monte Carlo Simulations", [10_000, 25_000, 50_000, 100_000], value=50_000)
    n_steps = st.select_slider("Binomial Steps", [50, 100, 200, 500], value=200)
    american = st.checkbox("American-style (Binomial only)", value=False)

# ── Fetch live price ──────────────────────────────────────────────────────────
if fetch_btn:
    with st.spinner(f"Fetching {ticker} from Yahoo Finance…"):
        try:
            live_price, exch, curr = get_spot_price(ticker)
            st.session_state["live_price"] = live_price
            st.session_state["live_ticker"] = ticker
            st.sidebar.success(f" ₹{live_price:,.2f}  ({exch})")
        except Exception as e:
            st.sidebar.error(str(e))

S = st.session_state.get("live_price", spot_override) if fetch_btn is False else spot_override
# Use sidebar spot override as primary; live price updates after button press
if "live_price" in st.session_state and st.session_state.get("live_ticker") == ticker:
    S = st.session_state["live_price"]


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("# Options Pricing Model")
st.markdown(
    '<div class="info-box">Prices European / American options on NSE-listed stocks and indices '
    'using three industry models. Spot data sourced live from Yahoo Finance. '
    'All prices in <strong>INR (₹)</strong>.</div>',
    unsafe_allow_html=True,
)

# KPI row
c1, c2, c3, c4 = st.columns(4)
moneyness = "ATM" if abs(S - strike) / S < 0.005 else ("ITM" if S > strike else "OTM")
with c1:
    st.markdown(f'<div class="metric-card"><div class="label">Spot Price</div>'
                f'<div class="value">₹{S:,.2f}</div>'
                f'<div class="sub">{ticker}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="label">Strike / Moneyness</div>'
                f'<div class="value">₹{strike:,.2f}</div>'
                f'<div class="sub">{moneyness}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="label">Days to Expiry</div>'
                f'<div class="value">{T_days}</div>'
                f'<div class="sub">{expiry.strftime("%d %b %Y")}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="label">Implied Volatility</div>'
                f'<div class="value">{vol_pct:.1f}%</div>'
                f'<div class="sub">Risk-free: {risk_free*100:.2f}%</div></div>', unsafe_allow_html=True)

# ── Compute ───────────────────────────────────────────────────────────────────
bs_call, bs_d = black_scholes(S, strike, T_years, risk_free, sigma, "call")
bs_put,  _    = black_scholes(S, strike, T_years, risk_free, sigma, "put")

mc_call, mc_call_ci, mc_paths = monte_carlo(S, strike, T_years, risk_free, sigma, n_sim, "call")
mc_put,  mc_put_ci,  _        = monte_carlo(S, strike, T_years, risk_free, sigma, n_sim, "put")

binom_call = binomial_tree(S, strike, T_years, risk_free, sigma, n_steps, "call", american)
binom_put  = binomial_tree(S, strike, T_years, risk_free, sigma, n_steps, "put",  american)

greeks = bs_greeks(S, strike, T_years, risk_free, sigma)

# ── Results section ───────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["  Prices & Greeks", "  Payoff Diagram", "  MC Distribution"])

with tab1:
    st.markdown('<div class="section-header">Model Prices</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="model-card">
          <div class="model-name">Black-Scholes</div>
          <span class="price-label">CALL</span><span class="call-price">₹{bs_call:,.2f}</span><br>
          <span class="price-label">PUT &nbsp;</span><span class="put-price">₹{bs_put:,.2f}</span>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="model-card">
          <div class="model-name">Monte Carlo ({n_sim:,} sims)</div>
          <span class="price-label">CALL</span><span class="call-price">₹{mc_call:,.2f}</span>
          <span style="font-size:0.68rem;color:#5a7aab">&nbsp;[{mc_call_ci[0]:.2f}–{mc_call_ci[1]:.2f}]</span><br>
          <span class="price-label">PUT &nbsp;</span><span class="put-price">₹{mc_put:,.2f}</span>
          <span style="font-size:0.68rem;color:#5a7aab">&nbsp;[{mc_put_ci[0]:.2f}–{mc_put_ci[1]:.2f}]</span>
        </div>""", unsafe_allow_html=True)

    with col3:
        style = "American" if american else "European"
        st.markdown(f"""
        <div class="model-card">
          <div class="model-name">Binomial CRR ({n_steps} steps · {style})</div>
          <span class="price-label">CALL</span><span class="call-price">₹{binom_call:,.2f}</span><br>
          <span class="price-label">PUT &nbsp;</span><span class="put-price">₹{binom_put:,.2f}</span>
        </div>""", unsafe_allow_html=True)

    # ── Greeks ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Option Greeks (Black-Scholes)</div>', unsafe_allow_html=True)
    gcol1, gcol2 = st.columns(2)
    with gcol1:
        st.markdown("**CALL**")
        st.markdown(f"""
        <div>
          <div class="greek-badge"><div class="greek-name">Δ Delta</div><div class="greek-val">{greeks['delta_call']}</div></div>
          <div class="greek-badge"><div class="greek-name">Γ Gamma</div><div class="greek-val">{greeks['gamma']}</div></div>
          <div class="greek-badge"><div class="greek-name">Θ Theta / day</div><div class="greek-val">₹{greeks['theta_call']}</div></div>
          <div class="greek-badge"><div class="greek-name">ν Vega / 1% vol</div><div class="greek-val">₹{greeks['vega']}</div></div>
          <div class="greek-badge"><div class="greek-name">ρ Rho / 1% rate</div><div class="greek-val">₹{greeks['rho_call']}</div></div>
        </div>""", unsafe_allow_html=True)
    with gcol2:
        st.markdown("**PUT**")
        st.markdown(f"""
        <div>
          <div class="greek-badge"><div class="greek-name">Δ Delta</div><div class="greek-val">{greeks['delta_put']}</div></div>
          <div class="greek-badge"><div class="greek-name">Γ Gamma</div><div class="greek-val">{greeks['gamma']}</div></div>
          <div class="greek-badge"><div class="greek-name">Θ Theta / day</div><div class="greek-val">₹{greeks['theta_put']}</div></div>
          <div class="greek-badge"><div class="greek-name">ν Vega / 1% vol</div><div class="greek-val">₹{greeks['vega']}</div></div>
          <div class="greek-badge"><div class="greek-name">ρ Rho / 1% rate</div><div class="greek-val">₹{greeks['rho_put']}</div></div>
        </div>""", unsafe_allow_html=True)

    # ── Model comparison table ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Comparison Table</div>', unsafe_allow_html=True)
    df_compare = pd.DataFrame({
        "Model":      ["Black-Scholes", "Monte Carlo", f"Binomial ({style})"],
        "Call (₹)":   [bs_call, mc_call, binom_call],
        "Put (₹)":    [bs_put,  mc_put,  binom_put],
        "Call Δ vs BS": [0.0, round(mc_call - bs_call, 4), round(binom_call - bs_call, 4)],
        "Put Δ vs BS":  [0.0, round(mc_put  - bs_put,  4), round(binom_put  - bs_put,  4)],
    })
    st.dataframe(df_compare.set_index("Model"), use_container_width=True)


with tab2:
    st.markdown("### Payoff at Expiry")
    S_range = np.linspace(max(1, S * 0.5), S * 1.5, 300)

    # BS prices across spot range
    call_vals = [black_scholes(s, strike, T_years, risk_free, sigma, "call")[0] for s in S_range]
    put_vals  = [black_scholes(s, strike, T_years, risk_free, sigma, "put")[0]  for s in S_range]
    call_payoff_exp = np.maximum(S_range - strike, 0)
    put_payoff_exp  = np.maximum(strike - S_range, 0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_range, y=call_vals, name="Call (BS today)",
                             line=dict(color="#34d399", width=2)))
    fig.add_trace(go.Scatter(x=S_range, y=put_vals,  name="Put (BS today)",
                             line=dict(color="#f87171", width=2)))
    fig.add_trace(go.Scatter(x=S_range, y=call_payoff_exp, name="Call payoff at expiry",
                             line=dict(color="#34d399", width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=S_range, y=put_payoff_exp,  name="Put payoff at expiry",
                             line=dict(color="#f87171", width=1.5, dash="dash")))
    fig.add_vline(x=S,      line_width=1, line_dash="dot", line_color="#4a7fc1",
                  annotation_text=f"Spot ₹{S:,.0f}", annotation_position="top right")
    fig.add_vline(x=strike, line_width=1, line_dash="dot", line_color="#f59e0b",
                  annotation_text=f"Strike ₹{strike:,.0f}", annotation_position="top left")
    fig.update_layout(
        paper_bgcolor="#0a0f1e", plot_bgcolor="#0d1428",
        font=dict(family="Inter", color="#8aa8d4"),
        xaxis=dict(title="Spot Price (₹)", gridcolor="#1e2d50", color="#8aa8d4"),
        yaxis=dict(title="Option Price (₹)", gridcolor="#1e2d50", color="#8aa8d4"),
        legend=dict(bgcolor="#0d1428", bordercolor="#1e2d50"),
        margin=dict(l=10, r=10, t=30, b=10),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Sensitivity: price vs vol
    st.markdown("### Price vs Implied Volatility")
    vols = np.linspace(0.05, 1.0, 100)
    cv = [black_scholes(S, strike, T_years, risk_free, v, "call")[0] for v in vols]
    pv = [black_scholes(S, strike, T_years, risk_free, v, "put")[0]  for v in vols]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=vols*100, y=cv, name="Call", line=dict(color="#34d399", width=2)))
    fig2.add_trace(go.Scatter(x=vols*100, y=pv, name="Put",  line=dict(color="#f87171", width=2)))
    fig2.add_vline(x=vol_pct, line_width=1, line_dash="dot", line_color="#a78bfa",
                   annotation_text=f"Current {vol_pct:.0f}%")
    fig2.update_layout(
        paper_bgcolor="#0a0f1e", plot_bgcolor="#0d1428",
        font=dict(family="Inter", color="#8aa8d4"),
        xaxis=dict(title="Volatility (%)", gridcolor="#1e2d50", color="#8aa8d4"),
        yaxis=dict(title="Option Price (₹)", gridcolor="#1e2d50", color="#8aa8d4"),
        legend=dict(bgcolor="#0d1428", bordercolor="#1e2d50"),
        margin=dict(l=10, r=10, t=30, b=10), height=380,
    )
    st.plotly_chart(fig2, use_container_width=True)


with tab3:
    st.markdown("### Monte Carlo Terminal Price Distribution")
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(
        x=mc_paths, nbinsx=100,
        marker_color="#4a7fc1", opacity=0.75, name="Terminal S",
    ))
    fig3.add_vline(x=S,      line_color="#4a7fc1", line_dash="dot",
                   annotation_text=f"Spot ₹{S:,.0f}")
    fig3.add_vline(x=strike, line_color="#f59e0b", line_dash="dot",
                   annotation_text=f"Strike ₹{strike:,.0f}")
    fig3.update_layout(
        paper_bgcolor="#0a0f1e", plot_bgcolor="#0d1428",
        font=dict(family="Inter", color="#8aa8d4"),
        xaxis=dict(title="Terminal Spot Price (₹)", gridcolor="#1e2d50", color="#8aa8d4"),
        yaxis=dict(title="Frequency", gridcolor="#1e2d50", color="#8aa8d4"),
        margin=dict(l=10, r=10, t=30, b=10), height=400,
        showlegend=False,
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(f"""
    <div class="info-box">
      <strong>{n_sim:,} simulated paths</strong> · 
      Mean terminal price: <strong>₹{mc_paths.mean():,.2f}</strong> · 
      Std dev: <strong>₹{mc_paths.std():,.2f}</strong><br>
      Probability (call ITM): <strong>{(mc_paths > strike).mean()*100:.1f}%</strong> &nbsp;|&nbsp;
      Probability (put ITM): <strong>{(mc_paths < strike).mean()*100:.1f}%</strong>
    </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;font-size:0.75rem;color:#2a4070">'
    'For educational purposes only. Not financial advice. '
    'NSE F&O lot sizes and SEBI margin rules apply in live trading.'
    '</p>',
    unsafe_allow_html=True,
)
