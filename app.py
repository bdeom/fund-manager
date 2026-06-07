"""
Portfolio Management System - Flask Backend
LSTM and NSGA-II run in background threads (Render-compatible, no subprocess).
"""

from flask import Flask, jsonify, request, render_template, session, redirect, url_for
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json, os, threading, warnings, hashlib
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
app.secret_key = os.environ.get("SECRET_KEY", "fundmgr-dev-secret-2024")
APP_PASSWORD   = os.environ.get("APP_PASSWORD", "fundmgr2024")
PORT           = int(os.environ.get("PORT", 5757))
DATA_DIR       = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH",
                  os.path.dirname(os.path.abspath(__file__)))
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
PORTFOLIO_FILE = os.path.join(DATA_DIR, "portfolio.json")

_jobs      = {}
_jobs_lock = threading.Lock()

# ─────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────
def check_auth():
    return session.get("authenticated") is True

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if request.form.get("password", "") == APP_PASSWORD:
            session["authenticated"] = True
            return redirect(url_for("index"))
        error = "Invalid password"
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.before_request
def require_login():
    if request.endpoint in ("login", "static", "logout"):
        return
    if not check_auth():
        if request.path.startswith("/api/"):
            return jsonify({"error": "Unauthorized"}), 401
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
    "cash": 5000.0, "transactions": [], "alerts": [],
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
            hist = yf.Ticker(t).history(period="5d")
            prices[t] = float(hist['Close'].dropna().iloc[-1]) if not hist.empty else None
        except:
            prices[t] = None
    return prices

# ─────────────────────────────────────────────
# SIGNALS ENGINE
# ─────────────────────────────────────────────
def compute_drift_and_signals(portfolio, prices):
    holdings = portfolio["holdings"]
    cash = portfolio.get("cash", 0)
    results = []
    total_value = cash + sum(
        h["shares"] * (prices.get(t) or h["avg_cost"])
        for t, h in holdings.items()
    )

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
            signal = "SELL"; signal_reason = f"Profit target hit: +{gain_loss_pct:.1f}%"
            action_shares = round(shares * 0.5, 4); action_value = round(action_shares * price, 2)
        elif gain_loss_pct <= -15 and signal == "HOLD":
            signal = "SELL"; signal_reason = f"Stop-loss hit: {gain_loss_pct:.1f}%"
            action_shares = shares; action_value = round(shares * price, 2)

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
# LSTM — inline thread
# ─────────────────────────────────────────────
def _lstm_forecast_single(prices_array, forecast_days=10, lookback=20, epochs=30):
    """Try PyTorch LSTM first, fall back to sklearn ensemble if torch unavailable."""
    series = prices_array.reshape(-1, 1)
    if len(series) < lookback + 20:
        return None, None

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0.1, 0.9))
    scaled = scaler.fit_transform(series)

    # ── Try PyTorch LSTM ──
    try:
        import torch, torch.nn as nn

        X, y = [], []
        for i in range(len(scaled) - lookback):
            X.append(scaled[i:i+lookback])
            y.append(scaled[i+lookback])
        Xt = torch.tensor(np.array(X), dtype=torch.float32)
        yt = torch.tensor(np.array(y), dtype=torch.float32)

        class LSTMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm1 = nn.LSTM(1, 48, num_layers=1, batch_first=True)
                self.lstm2 = nn.LSTM(48, 24, num_layers=1, batch_first=True)
                self.fc    = nn.Linear(24, 1)
            def forward(self, x):
                out, _ = self.lstm1(x)
                out, _ = self.lstm2(out)
                return self.fc(out[:, -1, :])

        model = LSTMModel()
        opt   = torch.optim.Adam(model.parameters(), lr=0.003)
        loss_fn = nn.HuberLoss()
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(model(Xt), yt)
            loss.backward()
            opt.step()

        model.eval()
        seq = torch.tensor(scaled[-lookback:].reshape(1, lookback, 1), dtype=torch.float32)
        preds = []
        with torch.no_grad():
            for _ in range(forecast_days):
                out = model(seq)
                preds.append(out.item())
                seq = torch.cat([seq[:, 1:, :], out.view(1,1,1)], dim=1)

        forecast = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
        current  = float(series[-1][0])
        return forecast.tolist(), round((forecast[-1]-current)/current*100, 4)

    except Exception as torch_err:
        print(f"[LSTM] PyTorch unavailable ({torch_err}), using sklearn ensemble")

    # ── Fallback: Fast Ridge + momentum model (runs in <2s per ticker) ──
    try:
        from sklearn.linear_model import Ridge

        X, y = [], []
        for i in range(len(scaled) - lookback):
            X.append(scaled[i:i+lookback].flatten())
            y.append(scaled[i+lookback][0])
        X, y = np.array(X), np.array(y)

        model = Ridge(alpha=0.5).fit(X, y)

        # Add momentum signal: recent trend direction
        recent_trend = float(scaled[-1][0] - scaled[-lookback//2][0])

        seq = scaled[-lookback:].flatten()
        preds = []
        for step in range(forecast_days):
            pred = model.predict(seq.reshape(1, -1))[0]
            # Dampen momentum over horizon
            momentum = recent_trend * (1 - step/forecast_days) * 0.3
            pred = float(np.clip(pred + momentum, 0.05, 0.95))
            preds.append(pred)
            seq = np.append(seq[1:], pred)

        forecast = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
        current  = float(series[-1][0])
        return forecast.tolist(), round((forecast[-1]-current)/current*100, 4)

    except Exception as e:
        print(f"[LSTM] Fallback also failed: {e}")
        return None, None

def _run_lstm_job(job_id, tickers):
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "started": datetime.now().isoformat()}
    results = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
            if df.empty or len(df) < 60:
                results[ticker] = {"error": "Insufficient data"}
                continue
            prices = df["Close"].squeeze().dropna().values
            forecast_vals, exp_return = _lstm_forecast_single(prices)
            if forecast_vals is None:
                results[ticker] = {"error": "Model failed"}
                continue
            signal = "BULLISH" if exp_return > 2 else ("BEARISH" if exp_return < -2 else "NEUTRAL")
            results[ticker] = {
                "current_price":   round(float(prices[-1]), 2),
                "forecast_10d":    [round(v, 2) for v in forecast_vals],
                "expected_return": round(exp_return, 2),
                "signal":          signal,
                "dates": [(datetime.now()+timedelta(days=j+1)).strftime("%m/%d") for j in range(10)]
            }
        except Exception as e:
            results[ticker] = {"error": str(e)}

    with _jobs_lock:
        _jobs[job_id] = {"status": "done", "result": results, "finished": datetime.now().isoformat()}

# ─────────────────────────────────────────────
# NSGA-II — inline thread
# ─────────────────────────────────────────────
def _run_nsga2_job(job_id, tickers):
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "started": datetime.now().isoformat()}
    try:
        from pymoo.core.problem import Problem
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.sampling.rnd import FloatRandomSampling
        from pymoo.termination import get_termination

        closes = pd.DataFrame()
        valid  = []
        for t in tickers:
            try:
                df = yf.download(t, period="2y", auto_adjust=True, progress=False)
                if not df.empty and len(df) > 60:
                    closes[t] = df["Close"].squeeze().dropna()
                    valid.append(t)
            except:
                pass

        if len(valid) < 2:
            with _jobs_lock:
                _jobs[job_id] = {"status": "error", "error": f"Need 2+ valid tickers. Got: {valid}"}
            return

        closes = closes[valid].dropna()
        returns_df = closes.pct_change().dropna()
        n   = len(valid)
        mu  = returns_df.mean().values * 252
        cov = returns_df.cov().values  * 252
        current_w = np.ones(n) / n
        ret_matrix = returns_df.values

        class PortfolioProblem(Problem):
            def __init__(self):
                super().__init__(n_var=n, n_obj=3, n_ieq_constr=0,
                                 xl=np.zeros(n), xu=np.ones(n))
            def _evaluate(self, X, out, *args, **kwargs):
                rs = X.sum(axis=1, keepdims=True)
                W  = np.where(rs > 1e-9, X/rs, 1.0/n)
                port_ret = W @ mu
                port_var = np.einsum('ij,jk,ik->i', W, cov, W)
                port_vol = np.sqrt(np.maximum(port_var, 1e-12))
                sharpe   = port_ret / port_vol
                sim_rets = ret_matrix @ W.T
                threshold = np.percentile(sim_rets, 5, axis=0)
                cvar = np.array([
                    -sim_rets[sim_rets[:,i]<=threshold[i],i].mean()
                    if (sim_rets[:,i]<=threshold[i]).any() else 0.0
                    for i in range(W.shape[0])
                ])
                turnover = np.abs(W - current_w).sum(axis=1)
                out["F"] = np.column_stack([-sharpe, cvar, turnover])

        res = minimize(
            PortfolioProblem(),
            NSGA2(pop_size=80, sampling=FloatRandomSampling(),
                  crossover=SBX(prob=0.9, eta=15),
                  mutation=PM(eta=20), eliminate_duplicates=True),
            get_termination("n_gen", 60),
            verbose=False, seed=42
        )

        if res.X is None or len(res.X) == 0:
            with _jobs_lock:
                _jobs[job_id] = {"status": "error", "error": "No Pareto solutions found"}
            return

        rs = res.X.sum(axis=1, keepdims=True)
        W  = np.where(rs > 1e-9, res.X/rs, 1.0/n)
        F  = res.F
        F_norm = (F - F.min(axis=0)) / np.where((F.max(axis=0)-F.min(axis=0))>1e-9,
                                                  F.max(axis=0)-F.min(axis=0), 1.0)
        knee = int(np.argmin(np.linalg.norm(F_norm, axis=1)))

        solutions = []
        for i, (w, f) in enumerate(zip(W, F)):
            port_ret = float(w @ mu)
            port_vol = float(np.sqrt(w @ cov @ w))
            solutions.append({
                "id": i,
                "weights":    {t: round(float(w[j]), 4) for j, t in enumerate(valid)},
                "sharpe":     round(-float(f[0]), 3),
                "cvar":       round(float(f[1])*100, 3),
                "turnover":   round(float(f[2])*100, 2),
                "ann_return": round(port_ret*100, 2),
                "ann_vol":    round(port_vol*100, 2),
                "is_knee":    (i == knee)
            })

        top = sorted(solutions, key=lambda x: x["sharpe"], reverse=True)[:6]
        knee_sol = solutions[knee]
        if knee_sol not in top:
            top.append(knee_sol)

        with _jobs_lock:
            _jobs[job_id] = {
                "status": "done",
                "result": {"solutions": top, "tickers": valid,
                           "skipped": [t for t in tickers if t not in valid]},
                "finished": datetime.now().isoformat()
            }
    except Exception as e:
        import traceback
        with _jobs_lock:
            _jobs[job_id] = {"status": "error", "error": str(e),
                             "traceback": traceback.format_exc()}

# ─────────────────────────────────────────────
# ROUTES — CORE
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/portfolio", methods=["GET"])
def get_portfolio():
    p = load_portfolio()
    tickers = list(p["holdings"].keys())
    prices  = get_current_prices(tickers)
    signals, total, cash = compute_drift_and_signals(p, prices)
    return jsonify({"holdings": signals, "total_value": total, "cash": cash,
                    "cash_pct": round(cash/total*100, 2) if total > 0 else 0,
                    "tickers": tickers})

@app.route("/api/portfolio/holding", methods=["POST"])
def add_holding():
    p = load_portfolio()
    data = request.json
    ticker = data["ticker"].upper()
    p["holdings"][ticker] = {"shares": float(data["shares"]),
                              "avg_cost": float(data["avg_cost"]),
                              "target_pct": float(data.get("target_pct", 0))}
    total_t = sum(h["target_pct"] for h in p["holdings"].values())
    if total_t > 0:
        for h in p["holdings"].values():
            h["target_pct"] = round(h["target_pct"] / total_t * 100, 2)
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
    p = load_portfolio()
    data = request.json
    ticker = data["ticker"].upper()
    action = data["action"]
    shares = float(data["shares"])
    price  = float(data["price"])
    cost   = shares * price
    tx = {"date": datetime.now().isoformat(), "ticker": ticker,
          "action": action, "shares": shares, "price": price, "value": round(cost, 2)}
    p.setdefault("transactions", []).insert(0, tx)
    if ticker in p["holdings"]:
        h = p["holdings"][ticker]
        if action == "BUY":
            total_s = h["shares"] + shares
            h["avg_cost"] = round((h["shares"]*h["avg_cost"]+cost)/total_s, 4)
            h["shares"] = round(total_s, 4)
            p["cash"] = p.get("cash", 0) - cost
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
    p = load_portfolio()
    tickers = list(p["holdings"].keys())
    prices  = get_current_prices(tickers)
    total_cost  = sum(h["shares"]*h["avg_cost"] for h in p["holdings"].values())
    total_value = sum(h["shares"]*(prices.get(t) or h["avg_cost"]) for t,h in p["holdings"].items())
    per_asset = []
    for t, h in p["holdings"].items():
        price = prices.get(t) or h["avg_cost"]
        val  = h["shares"] * price
        cost = h["shares"] * h["avg_cost"]
        per_asset.append({"ticker": t, "value": round(val, 2), "cost": round(cost, 2),
                          "gain": round(val-cost, 2),
                          "gain_pct": round((val-cost)/cost*100, 2) if cost > 0 else 0})
    per_asset.sort(key=lambda x: x["gain_pct"], reverse=True)
    total_gain = total_value - total_cost
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
    if not tickers:
        return jsonify({"error": "No tickers"})
    job_id = f"lstm_{datetime.now().strftime('%H%M%S')}"
    threading.Thread(target=_run_lstm_job, args=(job_id, tickers), daemon=True).start()
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
    if not tickers:
        return jsonify({"error": "No tickers"})
    job_id = f"nsga2_{datetime.now().strftime('%H%M%S')}"
    threading.Thread(target=_run_nsga2_job, args=(job_id, tickers), daemon=True).start()
    return jsonify({"job_id": job_id, "status": "started"})

@app.route("/api/optimize/status/<job_id>", methods=["GET"])
def optimize_status(job_id):
    with _jobs_lock:
        return jsonify(_jobs.get(job_id, {"status": "unknown"}))

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print(f"📈 Portfolio Manager — http://localhost:{PORT}")
    app.run(debug=False, port=PORT, host="0.0.0.0", threaded=True)
