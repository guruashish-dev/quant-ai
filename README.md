# Option Pricing Model

A Streamlit web app for pricing NSE/BSE options using three industry-standard models.

## Features
- **Black-Scholes** — closed-form European pricing + full Greeks (Δ, Γ, Θ, ν, ρ)
- **Monte Carlo** — configurable simulations with 95% confidence intervals
- **Binomial CRR** — European and **American**-style pricing
- **Live spot prices** via Yahoo Finance (NSE/BSE tickers)
- Payoff diagrams, vol sensitivity, and MC terminal distribution charts
- Pre-loaded with NIFTY 50, BANKNIFTY, Reliance, TCS, HDFC Bank, and more

---

## Local Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tickers Reference

| Security | Yahoo Finance Ticker |
|---|---|
| NIFTY 50 | `^NSEI` |
| BANKNIFTY | `^NSEBANK` |
| NSE stocks | `SYMBOL.NS` (e.g. `RELIANCE.NS`) |
| BSE stocks | `SYMBOL.BO` (e.g. `RELIANCE.BO`) |
