from flask import Flask, render_template, jsonify, request
import requests
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import time

app = Flask(__name__)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
    'Accept': 'application/json'
}

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# ── Google Trends ─────────────────────────────────────────────────────────────
_trends_cache: dict = {}
_trends_ts:    dict = {}
TRENDS_TTL = 3600

def get_trends_score(keyword: str) -> int | None:
    now = time.time()
    if keyword in _trends_cache and now - _trends_ts.get(keyword,0) < TRENDS_TTL:
        return _trends_cache[keyword]
    try:
        from pytrends.request import TrendReq
        pt = TrendReq(hl="en-US", tz=0, timeout=(5,10), retries=1, backoff_factor=0.5)
        pt.build_payload([keyword], timeframe="today 3-m", geo="")
        df = pt.interest_over_time()
        if df is None or df.empty or keyword not in df.columns:
            return None
        vals = df[keyword].values.astype(float)
        if len(vals) < 4: return None
        recent = vals[-2:].mean()
        prior  = vals[-12:-2].mean() if len(vals)>=12 else vals[:-2].mean()
        if prior <= 0: return None
        score  = round(min(100, (recent/prior)*50))
        _trends_cache[keyword]=score; _trends_ts[keyword]=now
        return score
    except Exception:
        return None

# ── CoinGlass / OKX funding rates ────────────────────────────────────────────
_funding_cache: dict = {}
_funding_ts:    dict = {}
FUNDING_TTL = 900

def get_funding_rate(symbol: str) -> dict:
    now = time.time()
    sym = symbol.upper()
    if sym in _funding_cache and now - _funding_ts.get(sym,0) < FUNDING_TTL:
        return _funding_cache[sym]
    result = {"rate": None, "signal": "neutral"}
    try:
        # OKX public funding rate (no key needed)
        okx_sym = sym + "-USDT-SWAP"
        r = requests.get("https://www.okx.com/api/v5/public/funding-rate",
                         params={"instId": okx_sym}, headers=HEADERS, timeout=6)
        if r.status_code == 200:
            items = r.json().get("data",[])
            if items:
                rate = float(items[0].get("fundingRate",0))*100
                result["rate"]   = round(rate, 4)
                result["signal"] = ("bullish" if rate < -0.01 else
                                    "bearish" if rate >  0.05 else "neutral")
    except Exception:
        pass
    _funding_cache[sym]=result; _funding_ts[sym]=now
    return result

# ── BTC price for relative RS calc ───────────────────────────────────────────
_btc_cache: dict  = {}
_btc_ts:    float = 0
BTC_TTL = 300

def get_btc_prices_30d() -> list:
    global _btc_ts
    now = time.time()
    if _btc_cache.get("p") and now - _btc_ts < BTC_TTL:
        return _btc_cache["p"]
    try:
        r = requests.get(f"{COINGECKO_BASE}/coins/bitcoin/market_chart",
                         params={"vs_currency":"eur","days":"30","interval":"daily"},
                         headers=HEADERS, timeout=10)
        if r.status_code == 200:
            prices = [p[1] for p in r.json().get("prices",[])]
            _btc_cache["p"]=prices; _btc_ts=now
            return prices
    except Exception:
        pass
    return []
UNIVERSE = {
    # OKX EUR pairs (MiCA-compliant, Bruno's trading universe)
    "OKX — EUR Pairs": [
        "bitcoin","ethereum","solana","sui","chainlink","injective-protocol",
        "avalanche-2","polkadot","cardano","near","cosmos","algorand",
        "aptos","arbitrum","optimism","polygon","base","fantom",
        "uniswap","aave","compound-governance-token","maker","curve-dao-token",
        "synthetix-network-token","yearn-finance","lido-dao","rocket-pool",
        "the-graph","filecoin","render-token","fetch-ai","ocean-protocol",
        "helium","arweave","livepeer","storj","siacoin","theta-token"
    ],
    # Large Cap L1s
    "Layer 1 — Large Cap": [
        "bitcoin","ethereum","solana","avalanche-2","cardano","polkadot",
        "near","cosmos","algorand","tron","eos","tezos","flow",
        "hedera-hashgraph","internet-computer","aptos","sui","sei-network",
        "injective-protocol","celestia","mantle"
    ],
    # Layer 2s & Scaling
    "Layer 2 & Scaling": [
        "matic-network","arbitrum","optimism","base","zksync","starknet",
        "linea","scroll","immutable-x","loopring","metis-token",
        "boba-network","fuel-network","taiko"
    ],
    # DeFi Blue Chips
    "DeFi — Blue Chips": [
        "uniswap","aave","compound-governance-token","maker","curve-dao-token",
        "synthetix-network-token","yearn-finance","balancer","sushi","dydx",
        "1inch","bancor","kyber-network-crystal","0x","venus",
        "pancakeswap-token","trader-joe-2","gmx","gains-network","perpetual-protocol"
    ],
    # DeFi Infrastructure
    "DeFi — Infrastructure": [
        "lido-dao","rocket-pool","frax-share","liquity","alchemix",
        "convex-finance","stake-dao","euler","morpho","radiant-capital",
        "maple","clearpool","goldfinch","truefi","centrifuge"
    ],
    # AI & Data
    "AI & Data Layer": [
        "fetch-ai","singularitynet","ocean-protocol","the-graph","render-token",
        "livepeer","bittensor","akash-network","io-net","grass",
        "worldcoin","chaingpt","cookie-dao","numerai","autonolas"
    ],
    # RWA & Institutional
    "RWA & Institutional": [
        "maker","ondo-finance","centrifuge","maple","goldfinch","truefi",
        "backed-finance","clearpool","credix","flux-finance",
        "allianceblock","realtoken-ecosystem-governance","mantra-dao"
    ],
    # Gaming & Metaverse
    "Gaming & Metaverse": [
        "axie-infinity","the-sandbox","decentraland","gala","illuvium",
        "star-atlas","stepn","gods-unchained","alien-worlds","splinterlands",
        "immutable-x","beam-2","ronin","pixels","wildcard"
    ],
    # Infrastructure & Oracles
    "Infrastructure & Oracles": [
        "chainlink","band-protocol","api3","uma","tellor","dia-data",
        "supra","pyth-network","redstone","chronicle",
        "filecoin","arweave","storj","siacoin","theta-token",
        "helium","akash-network","pocket-network"
    ],
    # Mid Cap Gems (potential 10x candidates)
    "Mid Cap Gems": [
        "injective-protocol","sui","aptos","sei-network","celestia",
        "pyth-network","jito-governance-token","bonk","jupiter-exchange-solana",
        "kamino","marinade","drift-protocol","zeta-markets",
        "hyperliquid","vertex-protocol","gains-network","gmx",
        "pendle","ethena","usual"
    ],
    # Narrative Plays 2025
    "Narrative 2025": [
        "bittensor","render-token","fetch-ai","io-net","akash-network",
        "ethena","usual","pendle","hyperliquid","jito-governance-token",
        "jupiter-exchange-solana","bonk","dogwifhat","popcat",
        "mantra-dao","ondo-finance","celestia","eigen-layer"
    ],
    "Custom": []
}

# ── CoinGecko fetcher ─────────────────────────────────────────────────────────
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

def fetch_markets_batch(coin_ids: list) -> dict:
    """Fetch up to 250 coins per call from CoinGecko /coins/markets."""
    result = {}
    chunk_size = 100  # stay well under rate limits
    for i in range(0, len(coin_ids), chunk_size):
        chunk = coin_ids[i:i+chunk_size]
        try:
            r = requests.get(f"{COINGECKO_BASE}/coins/markets", params={
                "vs_currency": "eur",
                "ids": ",".join(chunk),
                "order": "market_cap_desc",
                "per_page": len(chunk),
                "page": 1,
                "sparkline": "false",
                "price_change_percentage": "1h,24h,7d,30d",
                "locale": "en"
            }, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                for coin in r.json():
                    result[coin["id"]] = coin
            elif r.status_code == 429:
                time.sleep(12)
                # retry once
                r2 = requests.get(f"{COINGECKO_BASE}/coins/markets", params={
                    "vs_currency":"eur","ids":",".join(chunk),
                    "order":"market_cap_desc","per_page":len(chunk),"page":1,
                    "sparkline":"false","price_change_percentage":"1h,24h,7d,30d"
                }, headers=HEADERS, timeout=20)
                if r2.status_code == 200:
                    for coin in r2.json():
                        result[coin["id"]] = coin
        except Exception as e:
            print(f"CoinGecko batch error: {e}")
        time.sleep(1.2)  # respect rate limit
    return result


def fetch_coin_detail(coin_id: str) -> dict:
    """Fetch detailed coin data including developer stats."""
    try:
        r = requests.get(f"{COINGECKO_BASE}/coins/{coin_id}", params={
            "localization": "false",
            "ticks": "false",
            "market_data": "true",
            "community_data": "true",
            "developer_data": "true",
            "sparkline": "false"
        }, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


def fetch_defi_tvl(coin_id: str) -> dict:
    """Fetch TVL data from DeFiLlama."""
    # Map CoinGecko IDs to DeFiLlama slugs
    slug_map = {
        "uniswap": "uniswap", "aave": "aave", "maker": "makerdao",
        "compound-governance-token": "compound", "curve-dao-token": "curve",
        "lido-dao": "lido", "rocket-pool": "rocket-pool",
        "yearn-finance": "yearn-finance", "synthetix-network-token": "synthetix",
        "balancer": "balancer", "sushi": "sushi", "dydx": "dydx",
        "gmx": "gmx", "gains-network": "gains-network",
        "pendle": "pendle", "ethena": "ethena",
        "hyperliquid": "hyperliquid", "jupiter-exchange-solana": "jupiter",
        "morpho": "morpho", "euler": "euler", "radiant-capital": "radiant",
    }
    slug = slug_map.get(coin_id)
    if not slug:
        return {}
    try:
        r = requests.get(f"https://api.llama.fi/protocol/{slug}",
                         headers=HEADERS, timeout=10)
        if r.status_code == 200:
            d = r.json()
            tvl = d.get("tvl", [])
            current_tvl = tvl[-1]["totalLiquidityUSD"] if tvl else None
            prev_tvl    = tvl[-31]["totalLiquidityUSD"] if len(tvl) >= 31 else None
            tvl_change_30d = None
            if current_tvl and prev_tvl and prev_tvl > 0:
                tvl_change_30d = round((current_tvl/prev_tvl - 1)*100, 1)
            return {
                "tvl": round(current_tvl/1e6, 1) if current_tvl else None,  # in $M
                "tvl_change_30d": tvl_change_30d,
                "fees_24h": round(d.get("fees24h", 0)/1e3, 1) if d.get("fees24h") else None,  # $K
                "revenue_24h": round(d.get("revenue24h", 0)/1e3, 1) if d.get("revenue24h") else None,
            }
    except Exception:
        pass
    return {}


# ── Scoring ───────────────────────────────────────────────────────────────────
def score_crypto(market: dict, detail: dict, tvl_data: dict) -> dict:
    """
    Scores a coin across 4 dimensions:
      fundamental_score  – NVT proxy, fee/mcap, TVL/mcap
      momentum_score     – price RS vs BTC, volume surge, trend
      onchain_score      – dev activity, community growth, TVL growth
      gem_score          – composite with anticipation weighting
    """
    def safe(d, *keys, default=None):
        val = d
        for k in keys:
            if not isinstance(val, dict): return default
            val = val.get(k)
        return val if val is not None else default

    mcap      = safe(market, "market_cap") or 0
    vol_24h   = safe(market, "total_volume") or 0
    price     = safe(market, "current_price") or 0
    chg_1h    = safe(market, "price_change_percentage_1h_in_currency")
    chg_24h   = safe(market, "price_change_percentage_24h_in_currency")
    chg_7d    = safe(market, "price_change_percentage_7d_in_currency")
    chg_30d   = safe(market, "price_change_percentage_30d_in_currency")
    ath_pct   = safe(market, "ath_change_percentage")     # % below ATH
    atl_pct   = safe(market, "atl_change_percentage")     # % above ATL
    circ_sup  = safe(market, "circulating_supply") or 0
    total_sup = safe(market, "total_supply") or circ_sup
    max_sup   = safe(market, "max_supply")
    rank      = safe(market, "market_cap_rank") or 999

    # Developer data
    dev       = safe(detail, "developer_data") or {}
    commits_4w= safe(dev, "commit_count_4_weeks") or 0
    stars     = safe(dev, "stars") or 0
    forks     = safe(dev, "forks") or 0
    issues    = safe(dev, "total_issues") or 0
    closed_is = safe(dev, "closed_issues") or 0
    prs_merged= safe(dev, "pull_requests_merged") or 0
    contributors = safe(dev, "pull_request_contributors") or 0

    # Community data
    comm      = safe(detail, "community_data") or {}
    twitter_f = safe(comm, "twitter_followers") or 0
    reddit_sub= safe(comm, "reddit_subscribers") or 0

    # TVL data (from DeFiLlama)
    tvl           = tvl_data.get("tvl")          # $M
    tvl_change_30d= tvl_data.get("tvl_change_30d")
    fees_24h      = tvl_data.get("fees_24h")      # $K
    revenue_24h   = tvl_data.get("revenue_24h")   # $K

    # Derived metrics
    vol_mcap_ratio = vol_24h / mcap if mcap > 0 else 0
    supply_inf     = (circ_sup / total_sup) if total_sup > 0 else 1  # 1 = fully circulating
    fee_mcap_ratio = (revenue_24h * 365 * 1000 / mcap) if revenue_24h and mcap > 0 else None
    tvl_mcap_ratio = (tvl * 1e6 / mcap) if tvl and mcap > 0 else None

    # ── Fundamental score (0-100) ──────────────────────────────────────────
    f_score = 0; f_hits = 0

    # Market rank (lower = more established)
    if rank <= 10:   f_score += 30
    elif rank <= 50:  f_score += 22
    elif rank <= 100: f_score += 14
    elif rank <= 200: f_score += 7
    f_hits += 1

    # Fee/Revenue vs Market Cap (protocol revenue yield)
    if fee_mcap_ratio is not None:
        if   fee_mcap_ratio > 0.20: f_score += 30
        elif fee_mcap_ratio > 0.10: f_score += 22
        elif fee_mcap_ratio > 0.05: f_score += 14
        elif fee_mcap_ratio > 0.01: f_score += 6
        f_hits += 1

    # TVL / Market Cap (undervalued DeFi)
    if tvl_mcap_ratio is not None:
        if   tvl_mcap_ratio > 2.0:  f_score += 25
        elif tvl_mcap_ratio > 1.0:  f_score += 18
        elif tvl_mcap_ratio > 0.5:  f_score += 11
        elif tvl_mcap_ratio > 0.2:  f_score += 5
        f_hits += 1

    # Supply inflation (low inflation = store of value)
    if supply_inf > 0.90: f_score += 20; f_hits += 1
    elif supply_inf > 0.70: f_score += 12; f_hits += 1
    elif supply_inf > 0.50: f_score += 5; f_hits += 1

    # Volume/MCap ratio (healthy trading activity: 0.05-0.30)
    if 0.05 <= vol_mcap_ratio <= 0.30:
        f_score += 15; f_hits += 1
    elif 0.01 <= vol_mcap_ratio < 0.05:
        f_score += 7; f_hits += 1

    fund = round(min(100, f_score / max(f_hits, 1) * 3.5))

    # ── Momentum score (0-100) ─────────────────────────────────────────────
    m_score = 0; m_hits = 0

    if chg_7d is not None:
        if   chg_7d >  20: m_score += 30
        elif chg_7d >  10: m_score += 22
        elif chg_7d >   3: m_score += 14
        elif chg_7d >  -3: m_score += 8
        elif chg_7d > -10: m_score += 3
        m_hits += 1

    if chg_30d is not None:
        if   chg_30d >  40: m_score += 30
        elif chg_30d >  20: m_score += 22
        elif chg_30d >   5: m_score += 14
        elif chg_30d >  -5: m_score += 8
        elif chg_30d > -20: m_score += 3
        m_hits += 1

    if chg_24h is not None:
        if   chg_24h >  5: m_score += 15
        elif chg_24h >  2: m_score += 10
        elif chg_24h > -2: m_score += 7
        elif chg_24h > -5: m_score += 3
        m_hits += 1

    # Distance from ATH (buying opportunity when 50-80% below ATH)
    if ath_pct is not None:
        if   -80 < ath_pct < -50: m_score += 20  # deep value zone
        elif -50 < ath_pct < -30: m_score += 14
        elif -30 < ath_pct < -10: m_score += 8
        m_hits += 1

    momt = round(min(100, m_score / max(m_hits, 1) * 3.0))

    # ── On-chain / Dev score (0-100) ───────────────────────────────────────
    o_score = 0; o_hits = 0

    # Developer activity (4-week commits)
    if commits_4w > 0:
        if   commits_4w > 200: o_score += 30
        elif commits_4w > 100: o_score += 22
        elif commits_4w > 50:  o_score += 14
        elif commits_4w > 20:  o_score += 7
        o_hits += 1

    # GitHub stars (ecosystem interest)
    if stars > 0:
        if   stars > 10000: o_score += 20
        elif stars > 5000:  o_score += 14
        elif stars > 1000:  o_score += 8
        elif stars > 200:   o_score += 3
        o_hits += 1

    # Contributors (decentralised dev = moat)
    if contributors > 0:
        if   contributors > 100: o_score += 20
        elif contributors > 50:  o_score += 14
        elif contributors > 20:  o_score += 8
        elif contributors > 5:   o_score += 3
        o_hits += 1

    # TVL growth (protocol gaining traction)
    if tvl_change_30d is not None:
        if   tvl_change_30d >  30: o_score += 25
        elif tvl_change_30d >  10: o_score += 18
        elif tvl_change_30d >   0: o_score += 10
        elif tvl_change_30d > -10: o_score += 4
        o_hits += 1

    # Community size
    if twitter_f > 0:
        if   twitter_f > 1e6:  o_score += 15
        elif twitter_f > 200e3: o_score += 10
        elif twitter_f > 50e3:  o_score += 5
        o_hits += 1

    onchain = round(min(100, o_score / max(o_hits, 1) * 3.2))

    # ── Gem score composite ────────────────────────────────────────────────
    # Weights: fundamental 40% + momentum 30% + onchain 30%
    gem = round(0.40*fund + 0.30*momt + 0.30*onchain)

    # Style classification
    if fund >= 60 and momt >= 50:     style = "gem"
    elif fund >= 60 and momt < 40:    style = "value"
    elif momt >= 65 and fund < 40:    style = "momentum"
    elif onchain >= 60 and fund >= 40:style = "quality"
    elif momt < 30 and fund < 30:     style = "speculative"
    else:                             style = "neutral"

    return {
        "fundamental_score": fund,
        "momentum_score":    momt,
        "onchain_score":     onchain,
        "gem_score":         gem,
        "style":             style,
        "price":       round(price, 6) if price < 1 else round(price, 2),
        "chg_1h":      round(chg_1h, 2)  if chg_1h  is not None else None,
        "chg_24h":     round(chg_24h, 2) if chg_24h is not None else None,
        "chg_7d":      round(chg_7d, 2)  if chg_7d  is not None else None,
        "chg_30d":     round(chg_30d, 2) if chg_30d is not None else None,
        "mcap_m":      round(mcap/1e6, 1) if mcap else None,
        "vol_24h_m":   round(vol_24h/1e6, 1) if vol_24h else None,
        "vol_mcap":    round(vol_mcap_ratio*100, 1) if vol_mcap_ratio else None,
        "ath_pct":     round(ath_pct, 1) if ath_pct is not None else None,
        "rank":        rank,
        "supply_pct":  round(supply_inf*100, 1) if supply_inf else None,
        "tvl":         tvl,
        "tvl_chg_30d": tvl_change_30d,
        "fee_mcap":    round(fee_mcap_ratio*100, 2) if fee_mcap_ratio else None,
        "tvl_mcap":    round(tvl_mcap_ratio, 2) if tvl_mcap_ratio else None,
        "commits_4w":  commits_4w,
        "stars":       stars,
        "contributors":contributors,
        "twitter_f":   round(twitter_f/1e3, 1) if twitter_f else None,
        "fee_yield":   round(fee_mcap_ratio*100, 3) if fee_mcap_ratio else None,
        # raw for anticipation calc
        "_chg_7d_raw":  chg_7d,
        "_chg_30d_raw": chg_30d,
        "_vol_raw":     vol_24h,
        "_mcap_raw":    mcap,
    }


# ── Crypto anticipation signals v2 ───────────────────────────────────────────
def calc_crypto_anticipation(coin_id: str, market: dict, scores: dict) -> dict:
    """
    New anticipation signals:
      funding_rate     – perpetual funding rate (negative = bullish)
      funding_signal   – bullish/neutral/bearish
      btc_rs_30d       – return vs BTC over 30 days (alpha)
      vol_accel        – volume acceleration (recent vs prior)
      trends_score     – Google Trends 0-100
      signals_heatmap  – per-signal colour map
      anticipation_score – composite 0-100
    """
    result = {
        "funding_rate":      None,
        "funding_signal":    "neutral",
        "btc_rs_30d":        None,
        "vol_accel":         None,
        "trends_score":      None,
        "anticipation_score": 0,
        "signals_heatmap":   {},
    }

    symbol = (market.get("symbol") or coin_id).upper()

    # ── 1. Funding rate (OKX public API) ─────────────────────────────────
    try:
        fd = get_funding_rate(symbol)
        result["funding_rate"]   = fd["rate"]
        result["funding_signal"] = fd["signal"]
    except Exception:
        pass

    # ── 2. BTC-relative strength (30D) ───────────────────────────────────
    try:
        coin_30d = scores.get("chg_30d")
        btc_prices = get_btc_prices_30d()
        if btc_prices and len(btc_prices) >= 2:
            btc_30d = (btc_prices[-1]/btc_prices[0]-1)*100
            if coin_30d is not None:
                result["btc_rs_30d"] = round(coin_30d - btc_30d, 1)
    except Exception:
        pass

    # ── 3. Volume acceleration ────────────────────────────────────────────
    try:
        # CoinGecko gives us vol_24h and mcap; use vol/mcap ratio trend as proxy
        vol_mcap = scores.get("vol_mcap")
        chg_7d   = scores.get("_chg_7d_raw") or scores.get("chg_7d")
        chg_30d  = scores.get("_chg_30d_raw") or scores.get("chg_30d")
        if vol_mcap is not None:
            # High vol/mcap with positive price = accumulation
            if   vol_mcap > 20: result["vol_accel"] = 80
            elif vol_mcap > 10: result["vol_accel"] = 60
            elif vol_mcap > 5:  result["vol_accel"] = 40
            else:               result["vol_accel"] = 20
    except Exception:
        pass

    # ── 4. Google Trends ──────────────────────────────────────────────────
    try:
        name = market.get("name") or coin_id
        kw   = name.split()[0]
        result["trends_score"] = get_trends_score(kw)
    except Exception:
        pass

    # ── 5. Composite anticipation score ──────────────────────────────────
    a = 0; h = 0

    # Funding rate (negative = shorts paying longs = contrarian long)
    if result["funding_signal"] == "bullish":  a += 25; h += 1
    elif result["funding_signal"] == "neutral": a += 10; h += 1
    elif result["funding_signal"] == "bearish": a +=  3; h += 1

    # BTC RS (alpha generation)
    if result["btc_rs_30d"] is not None:
        a += (25 if result["btc_rs_30d"]>20 else
              18 if result["btc_rs_30d"]>10 else
              10 if result["btc_rs_30d"]>3  else
               5 if result["btc_rs_30d"]>0  else 0)
        h += 1

    # Momentum from base scores
    m = scores.get("momentum_score", 0)
    a += m * 0.25; h += 1

    # Volume acceleration
    if result["vol_accel"] is not None:
        a += result["vol_accel"] * 0.20; h += 1

    # ATH distance (deep value bonus)
    ath = scores.get("ath_pct")
    if ath is not None and ath < -50:
        a += 20; h += 1
    elif ath is not None and ath < -30:
        a += 12; h += 1

    # TVL growth
    tvl_chg = scores.get("tvl_chg_30d")
    if tvl_chg is not None and tvl_chg > 20:  a += 20; h += 1
    elif tvl_chg is not None and tvl_chg > 5: a += 10; h += 1

    # Google Trends
    ts = result["trends_score"]
    if ts is not None:
        a += 15 if ts>=70 else 8 if ts>=45 else 0
        h += 1

    # Dev activity
    commits = scores.get("commits_4w", 0)
    if commits > 100: a += 15; h += 1
    elif commits > 30: a += 8; h += 1

    result["anticipation_score"] = round(min(100, a/max(h,1)*1.6)) if h else 0

    # ── 6. Signal heatmap ─────────────────────────────────────────────────
    def tl(val, g, w):
        if val is None: return "grey"
        return "green" if val>=g else "amber" if val>=w else "red"

    result["signals_heatmap"] = {
        "Funding":   "green" if result["funding_signal"]=="bullish" else
                     "amber" if result["funding_signal"]=="neutral" else "red",
        "BTC RS 30D":tl(result["btc_rs_30d"], 10, 0),
        "Momentum":  tl(scores.get("momentum_score"), 60, 40),
        "Vol Surge": tl(result["vol_accel"], 60, 40),
        "ATH Dist":  "green" if ath and ath<-50 else "amber" if ath and ath<-30 else "grey",
        "TVL Growth":tl(tvl_chg, 20, 5),
        "Dev Act.":  tl(commits, 100, 30),
        "Trends":    tl(result["trends_score"], 70, 45),
    }

    return result


def sanitize(obj):
    if isinstance(obj, dict):  return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [sanitize(v) for v in obj]
    if isinstance(obj, float) and (obj != obj or obj == float('inf') or obj == float('-inf')):
        return None
    try:
        import numpy as np
        if isinstance(obj, (np.floating, np.integer)):
            v = obj.item()
            return None if isinstance(v, float) and (v!=v or abs(v)==float('inf')) else v
        if isinstance(obj, np.bool_): return bool(obj)
    except Exception: pass
    return obj


def fetch_coin(coin_id: str, market_data: dict) -> dict:
    """Build a full record for one coin using pre-fetched market data."""
    base = {
        "id": coin_id, "symbol": "?", "name": coin_id,
        "error": None,
        "fundamental_score":0,"momentum_score":0,"onchain_score":0,"gem_score":0,
        "anticipation_score":0,
        "style":"speculative","price":None,"chg_24h":None,"chg_7d":None,
        "chg_30d":None,"mcap_m":None,"vol_24h_m":None,"rank":999,
        "ath_pct":None,"tvl":None,"commits_4w":0,"twitter_f":None,
        "funding_rate":None,"funding_signal":"neutral",
        "btc_rs_30d":None,"vol_accel":None,"trends_score":None,
        "signals_heatmap":{},
    }

    market = market_data.get(coin_id)
    if not market:
        base["error"] = "Not found on CoinGecko"
        return base

    base["symbol"] = (market.get("symbol") or coin_id).upper()
    base["name"]   = market.get("name") or coin_id
    base["image"]  = market.get("image","")

    scores = score_crypto(market, {}, {})
    anti   = calc_crypto_anticipation(coin_id, market, scores)

    # Recompute gem score with anticipation bonus
    gem_final = round(scores["gem_score"]*0.65 + anti["anticipation_score"]*0.35)

    base.update(scores)
    base.update(anti)
    base["gem_score"] = gem_final
    # clean internal keys
    for k in ["_chg_7d_raw","_chg_30d_raw","_vol_raw","_mcap_raw"]:
        base.pop(k, None)
    return base


def fetch_coin_full(coin_id: str, market_data: dict) -> dict:
    """Full detail fetch including dev stats, TVL and anticipation — used for detail panel."""
    market   = market_data.get(coin_id, {})
    detail   = fetch_coin_detail(coin_id)
    tvl_data = fetch_defi_tvl(coin_id)
    base = {
        "id": coin_id,
        "symbol": (market.get("symbol") or coin_id).upper(),
        "name":   market.get("name") or coin_id,
        "image":  market.get("image",""),
        "error":  None,
    }
    scores = score_crypto(market, detail, tvl_data)
    anti   = calc_crypto_anticipation(coin_id, market, scores)
    gem_final = round(scores["gem_score"]*0.65 + anti["anticipation_score"]*0.35)
    base.update(scores)
    base.update(anti)
    base["gem_score"] = gem_final
    for k in ["_chg_7d_raw","_chg_30d_raw","_vol_raw","_mcap_raw"]:
        base.pop(k, None)
    return base


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    groups = {
        "🟠 OKX / EUR":        [k for k in UNIVERSE if k.startswith("OKX")],
        "⛓ Layer 1 & 2":      [k for k in UNIVERSE if k.startswith("Layer")],
        "🏦 DeFi":             [k for k in UNIVERSE if k.startswith("DeFi")],
        "🤖 AI & Data":        [k for k in UNIVERSE if k.startswith("AI")],
        "🌍 RWA & Infra":      [k for k in UNIVERSE if k.startswith("RWA") or k.startswith("Infrastructure")],
        "🎮 Gaming":           [k for k in UNIVERSE if k.startswith("Gaming")],
        "💎 Gems & Narratives":[k for k in UNIVERSE if k.startswith("Mid Cap") or k.startswith("Narrative")],
        "⚙️ Custom":           ["Custom"],
    }
    return render_template("index.html", universe=UNIVERSE, groups=groups)

@app.route("/api/scan", methods=["POST"])
def scan():
    data    = request.json
    coin_ids = list(dict.fromkeys([c.strip().lower() for c in data.get("coins",[]) if c.strip()]))[:120]
    if not coin_ids:
        return jsonify({"results":[], "total":0})

    # Batch fetch market data first (efficient — 1 API call per 100 coins)
    market_data = fetch_markets_batch(coin_ids)

    # Score all coins
    results = [fetch_coin(cid, market_data) for cid in coin_ids]

    mode = data.get("mode","gem")
    sort_key = {"fundamental":"fundamental_score","momentum":"momentum_score",
                "onchain":"onchain_score"}.get(mode,"gem_score")
    results.sort(key=lambda x: x.get(sort_key,0), reverse=True)

    return jsonify(sanitize({"results":results,"total":len(results)}))

@app.route("/api/detail/<coin_id>")
def detail(coin_id):
    market_data = fetch_markets_batch([coin_id])
    return jsonify(sanitize(fetch_coin_full(coin_id, market_data)))

@app.route("/api/universe")
def get_universe():
    return jsonify(UNIVERSE)

if __name__ == "__main__":
    total = sum(len(v) for v in UNIVERSE.values())
    print("🚀  Crypto GARP Scanner v1")
    print(f"    {total} coins · {len(UNIVERSE)} lists")
    print("    Data: CoinGecko + DeFiLlama (free, no API key)")
    print("    → http://127.0.0.1:5001")
    app.run(debug=False, port=5001, threaded=True)
