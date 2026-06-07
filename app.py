"""
Portfolio Management System - Flask Backend
Deployable on Railway.app with password protection.
LSTM and NSGA-II run as decoupled subprocesses.
"""

from flask import Flask, jsonify, request, render_template, session, redirect, url_for
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json, os, sys, subprocess, threading, warnings, hashlib
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# CONFIG — set via Railway environment variables
# ─────────────────────────────────────────────
app.secret_key    = os.environ.get("SECRET_KEY", "fundmgr-dev-secret-change-me")
APP_PASSWORD      = os.environ.get("APP_PASSWORD", "fundmgr2024")
PORT              = int(os.environ.get("PORT", 5757))

# Use /data volume on Railway, fallback to local
DATA_DIR          = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", os.path.dirname(os.path.abspath(__file__)))
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
PORTFOLIO_FILE    = os.path.join(DATA_DIR, "portfolio.json")
LSTM_RESULTS      = os.path.join(DATA_DIR, "lstm_results.json")
NSGA2_RESULTS     = os.path.join(DATA_DIR, "nsga2_results.json")
LSTM_WORKER       = os.path.join(BASE_DIR, "lstm_worker.py")
NSGA2_WORKER      = os.path.join(BASE_DIR, "nsga2_worker.py")

# Job tracker
_jobs      = {}
_jobs_lock = threading.Lock()

# ─────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────
def check_auth():
    return session.get("authenticated") == True

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        pw = request.form.get("password", "")
        if pw == APP_PASSWORD:
            session["authenticated"] = True
            return redirect(url_for("index"))
        error = "Invalid password"
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ─────────────────────────────────────────────
# PORTFOLIO STATE
# ─────────────────────────────────────────────
DEFAULT_PORTFOLIO = {
    "holdings": {
        "AAPL":    {"shares": 10, "avg_cost": 145.00, "target_pct": 20},
        "MSFT":    {"shares": 8,  "avg_cost": 280.00, "target_pct": 20},
        "GOOGL":   {"shares": 5,  "avg_cost": 120.00, "target_pct": 15},
        "NVDA":    {"shares": 6,  "avg_cost": 400.00, "target_pct": 15},
        "ASML.AS": {"shares": 3,  "avg_cost": 700.00, "target_pct": 15},
        "BRK-B":   {"shares": 20, "avg_cost": 310.00, "target_pct": 15},
    },
    "cash": 5000.0,
    "transactions": [],
    "alerts": [],
    "created": datetime.now().isoformat()
}

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    save_portfolio(DEFAULT_PORTFOLIO)
    return DEFAULT_PORTFOLIO

def save_portfolio(p):
    os.makedirs(os.path.dirname(PORTFOLIO_FILE), exist_ok=True)
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(p, f, indent=2, default=str)

# ─────────────────────────────────────────────
# MARKET DATA
# ─────────────────────────────────────────────
def get_current_prices(tickers):
    prices = {}
    for t in tickers:
        try:
            tk   = yf.Ticker(t)
            hist = tk.history(period="5d")
            prices[t] = float(hist['Close'].dropna().iloc[-1]) if not hist.empty else None
        except:
            prices[t] = None
    return prices

# ─────────────────────────────────────────────
# DRIFT & SIGNAL ENGINE
# ─────────────────────────────────────────────
def compute_drift_and_signals(portfolio, prices):
    holdings    = portfolio["holdings"]
    cash        = portfolio.get("cash", 0)
    results     = []
    total_value = cash

    for t, h in holdings.items():
        p = prices.get(t)
        if p: total_value += h["shares"] * p

    for ticker, h in holdings.items():
        price         = prices.get(ticker) or h["avg_cost"]
        shares        = h["shares"]
        avg_cost      = h["avg_cost"]
        target_pct    = h["target_pct"]
        current_value = shares * price
        cost_basis    = shares * avg_cost
        current_pct   = (current_value / total_value * 100) if total_value > 0 else 0
        drift         = current_pct - target_pct
        gain_loss_amt = current_value - cost_basis
        gain_loss_pct = (gain_loss_amt / cost_basis * 100) if cost_basis > 0 else 0

        signal = "HOLD"; signal_reason = ""; action_shares = 0; action_value = 0

        if drift > 5:
            trim_value    = (drift / 100) * total_value
            signal        = "TRIM"
            signal_reason = f"Overweight by {drift:+.1f}% vs target {target_pct}%"
            action_shares = round(trim_value / price, 4)
            action_value  = round(trim_value, 2)
        elif drift < -5:
            buy_value     = (abs(drift) / 100) * total_value
            signal        = "BUY"
            signal_reason = f"Underweight by {drift:+.1f}% vs target {target_pct}%"
            action_shares = round(buy_value / price, 4)
            action_value  = round(buy_value, 2)

        if gain_loss_pct >= 30 and signal == "HOLD":
            signal        = "SELL"
            signal_reason = f"Profit target hit: +{gain_loss_pct:.1f}%"
            action_shares = round(shares * 0.5, 4)
            action_value  = round(action_shares * price, 2)
        elif gain_loss_pct <= -15 and signal == "HOLD":
            signal        = "SELL"
            signal_reason = f"Stop-loss hit: {gain_loss_pct:.1f}%"
            action_shares = shares
            action_value  = round(shares * price, 2)

        results.append({
            "ticker": ticker, "shares": shares,
            "price": round(price, 2), "avg_cost": round(avg_cost, 2),
            "current_value": round(current_value, 2), "cost_basis": round(cost_basis, 2),
            "target_pct": target_pct, "current_pct": round(current_pct, 2),
            "drift": round(drift, 2), "gain_loss_amt": round(gain_loss_amt, 2),
            "gain_loss_pct": round(gain_loss_pct, 2), "signal": signal,
            "signal_reason": signal_reason, "action_shares": action_shares,
            "action_value": action_value,
        })

    results.sort(key=lambda x: abs(x["drift"]), reverse=True)
    return results, round(total_value, 2), round(cash, 2)

# ─────────────────────────────────────────────
# SUBPROCESS JOB RUNNER
# ─────────────────────────────────────────────
def _run_worker(job_id, script, tickers, result_file):
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "started": datetime.now().isoformat()}
    try:
        if os.path.exists(result_file):
            os.remove(result_file)
        proc = subprocess.run(
            [sys.executable, script] + tickers,
            capture_output=True, text=True,
            cwd=DATA_DIR, timeout=600
        )
        if os.path.exists(result_file):
            with open(result_file) as f:
                result = json.load(f)
            with _jobs_lock:
                _jobs[job_id] = {"status": "done", "result": result, "finished": datetime.now().isoformat()}
        else:
            err = proc.stderr[-2000:] if proc.stderr else "No output"
            with _jobs_lock:
                _jobs[job_id] = {"status": "error", "error": f"Worker failed: {err}"}
    except subprocess.TimeoutExpired:
        with _jobs_lock:
            _jobs[job_id] = {"status": "error", "error": "Timeout after 10 minutes"}
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id] = {"status": "error", "error": str(e)}

# ─────────────────────────────────────────────
# AUTH MIDDLEWARE
# ─────────────────────────────────────────────
@app.before_request
def require_login():
    if request.endpoint in ("login", "static", "logout"):
        return
    if not check_auth():
        if request.path.startswith("/api/"):
            return jsonify({"error": "Unauthorized"}), 401
        return redirect(url_for("login"))

# ─────────────────────────────────────────────
# ROUTES — PAGES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

# ─────────────────────────────────────────────
# ROUTES — PORTFOLIO
# ─────────────────────────────────────────────
@app.route("/api/portfolio", methods=["GET"])
def get_portfolio():
    p       = load_portfolio()
    tickers = list(p["holdings"].keys())
    prices  = get_current_prices(tickers)
    signals, total, cash = compute_drift_and_signals(p, prices)
    return jsonify({
        "holdings": signals, "total_value": total, "cash": cash,
        "cash_pct": round(cash / total * 100, 2) if total > 0 else 0,
        "tickers": tickers
    })

@app.route("/api/portfolio/holding", methods=["POST"])
def add_holding():
    p      = load_portfolio()
    data   = request.json
    ticker = data["ticker"].upper()
    p["holdings"][ticker] = {
        "shares":     float(data["shares"]),
        "avg_cost":   float(data["avg_cost"]),
        "target_pct": float(data.get("target_pct", 0))
    }
    total_target = sum(h["target_pct"] for h in p["holdings"].values())
    if total_target > 0:
        for h in p["holdings"].values():
            h["target_pct"] = round(h["target_pct"] / total_target * 100, 2)
    save_portfolio(p)
    return jsonify({"ok": True})

@app.route("/api/portfolio/holding/<ticker>", methods=["DELETE"])
def delete_holding(ticker):
    p = load_portfolio()
    p["holdings"].pop(ticker, None)
    save_portfolio(p)
    return jsonify({"ok": True})

@app.route("/api/portfolio/cash", methods=["POST"])
def update_cash():
    p = load_portfolio()
    p["cash"] = float(request.json["cash"])
    save_portfolio(p)
    return jsonify({"ok": True})

@app.route("/api/transaction", methods=["POST"])
def record_transaction():
    p      = load_portfolio()
    data   = request.json
    ticker = data["ticker"].upper()
    action = data["action"]
    shares = float(data["shares"])
    price  = float(data["price"])
    cost   = shares * price
    tx     = {"date": datetime.now().isoformat(), "ticker": ticker,
               "action": action, "shares": shares, "price": price, "value": round(cost, 2)}
    p.setdefault("transactions", []).insert(0, tx)
    if ticker in p["holdings"]:
        h = p["holdings"][ticker]
        if action == "BUY":
            total_s       = h["shares"] + shares
            h["avg_cost"] = round((h["shares"] * h["avg_cost"] + cost) / total_s, 4)
            h["shares"]   = round(total_s, 4)
            p["cash"]     = p.get("cash", 0) - cost
        elif action in ("SELL", "TRIM"):
            h["shares"] = round(h["shares"] - shares, 4)
            p["cash"]   = p.get("cash", 0) + cost
            if h["shares"] <= 0:
                del p["holdings"][ticker]
    elif action == "BUY":
        p["holdings"][ticker] = {"shares": shares, "avg_cost": price, "target_pct": 0}
        p["cash"] = p.get("cash", 0) - cost
    p["transactions"] = p["transactions"][:200]
    save_portfolio(p)
    return jsonify({"ok": True, "transaction": tx})

@app.route("/api/transactions", methods=["GET"])
def get_transactions():
    return jsonify(load_portfolio().get("transactions", []))

@app.route("/api/performance", methods=["GET"])
def get_performance():
    p       = load_portfolio()
    tickers = list(p["holdings"].keys())
    prices  = get_current_prices(tickers)
    total_cost  = sum(h["shares"] * h["avg_cost"] for h in p["holdings"].values())
    total_value = sum(h["shares"] * (prices.get(t) or h["avg_cost"]) for t, h in p["holdings"].items())
    total_gain  = total_value - total_cost
    per_asset   = []
    for t, h in p["holdings"].items():
        price = prices.get(t) or h["avg_cost"]
        val   = h["shares"] * price
        cost  = h["shares"] * h["avg_cost"]
        per_asset.append({"ticker": t, "value": round(val, 2), "cost": round(cost, 2),
                           "gain": round(val-cost, 2),
                           "gain_pct": round((val-cost)/cost*100, 2) if cost > 0 else 0})
    per_asset.sort(key=lambda x: x["gain_pct"], reverse=True)
    return jsonify({"total_cost": round(total_cost, 2), "total_value": round(total_value, 2),
                    "total_gain": round(total_gain, 2),
                    "total_gain_pct": round(total_gain/total_cost*100, 2) if total_cost > 0 else 0,
                    "per_asset": per_asset})

# ─────────────────────────────────────────────
# ROUTES — LSTM
# ─────────────────────────────────────────────
@app.route("/api/forecast/start", methods=["POST"])
def forecast_start():
    tickers = request.json.get("tickers", [])
    if not tickers: return jsonify({"error": "No tickers"})
    job_id = f"lstm_{datetime.now().strftime('%H%M%S')}"
    threading.Thread(target=_run_worker, args=(job_id, LSTM_WORKER, tickers, LSTM_RESULTS), daemon=True).start()
    return jsonify({"job_id": job_id, "status": "started"})

@app.route("/api/forecast/status/<job_id>", methods=["GET"])
def forecast_status(job_id):
    with _jobs_lock:
        return jsonify(_jobs.get(job_id, {"status": "unknown"}))

# ─────────────────────────────────────────────
# ROUTES — NSGA-II
# ─────────────────────────────────────────────
@app.route("/api/optimize/start", methods=["POST"])
def optimize_start():
    tickers = request.json.get("tickers", [])
    if not tickers: return jsonify({"error": "No tickers"})
    job_id = f"nsga2_{datetime.now().strftime('%H%M%S')}"
    threading.Thread(target=_run_worker, args=(job_id, NSGA2_WORKER, tickers, NSGA2_RESULTS), daemon=True).start()
    return jsonify({"job_id": job_id, "status": "started"})

@app.route("/api/optimize/status/<job_id>", methods=["GET"])
def optimize_status(job_id):
    with _jobs_lock:
        return jsonify(_jobs.get(job_id, {"status": "unknown"}))

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  📈 Portfolio Manager — http://localhost:{PORT}")
    print(f"  Password: {APP_PASSWORD}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    app.run(debug=False, port=PORT, host="0.0.0.0", threaded=True)
