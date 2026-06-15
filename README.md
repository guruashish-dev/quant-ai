# 📈 Indian Options Pricer

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

---

## 🚀 Deployment

> **Note:** Vercel is designed for Node.js/static sites and does **not** support Python/Streamlit.  
> Use one of these free platforms instead — all support Streamlit natively:

### Option 1 — Streamlit Community Cloud (Recommended, Free)

1. Push this folder to a **public GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → select your repo → set main file to `app.py`
4. Click **Deploy** — you'll get a public URL like `https://your-app.streamlit.app`

### Option 2 — Railway (Free tier, supports Python)

```bash
# Install Railway CLI
npm i -g @railway/cli
railway login
railway init
railway up
```
Set start command: `streamlit run app.py --server.port $PORT`

### Option 3 — Render (Free tier)

1. Push to GitHub
2. New Web Service → connect repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

### Option 4 — Hugging Face Spaces (Free, zero config)

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose **Streamlit** as the SDK
3. Upload `app.py` and `requirements.txt`
4. Done — public URL in ~2 minutes

---

## Tickers Reference

| Security | Yahoo Finance Ticker |
|---|---|
| NIFTY 50 | `^NSEI` |
| BANKNIFTY | `^NSEBANK` |
| NSE stocks | `SYMBOL.NS` (e.g. `RELIANCE.NS`) |
| BSE stocks | `SYMBOL.BO` (e.g. `RELIANCE.BO`) |
