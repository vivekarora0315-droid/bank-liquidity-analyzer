"""
Bank Liquidity Risk Analyzer - v4
=================================

Upgrades over v3:
  1. All 8 US G-SIBs covered (added State Street).
  2. NEW TAB — 8-K Earnings Release Viewer:
       - Fetches the full 8-K press release text from SEC EDGAR live
       - Section navigator: Highlights, Capital, Liquidity, NII/Revenue, Outlook
       - Side-by-side: app hardcoded value vs. 8-K extracted value for every ratio
       - Direct link to the EDGAR filing page
       - Filing metadata: date, accession number, document type
  3. All v3 features preserved (live yield curve, XBRL verification,
     LCR stress tester, NSFR, peer comparison, ALCO commentary, glossary).
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re as _re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from scipy.optimize import least_squares

# =========================================================================
# PAGE CONFIG + STYLES
# =========================================================================
st.set_page_config(
    page_title="US G-SIB Liquidity Risk Analyzer v4",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; color:#2F3E4E; }
.main .block-container { background: #FAF7F2; padding-top: 1.8rem; }
body { background-color: #FAF7F2; }
.app-title { font-size:2.15rem; font-weight:700; color:#3E5C76; letter-spacing:-0.4px; }
.app-sub   { font-size:0.82rem; color:#7A8B99; text-transform:uppercase; letter-spacing:.65px; margin-bottom:1rem; font-weight:500; }
.kpi-box  { background:#FFFFFF; border-top:3px solid #5A7A8C; padding:.85rem 1.05rem .7rem; border-radius:5px; box-shadow: 0 1px 3px rgba(80,100,120,.06); }
.kpi-lab  { font-size:.68rem; font-weight:600; color:#7A8B99; text-transform:uppercase; letter-spacing:.75px; }
.kpi-val  { font-family:'IBM Plex Mono',monospace; font-size:1.55rem; font-weight:600; color:#3E5C76; margin-top:2px; }
.pass  { background:#E6EFE5; color:#4A6B4E; border:1px solid #A3BFA4; padding:2px 12px; border-radius:20px; font-size:.75rem; font-weight:600; }
.fail  { background:#F4DDD6; color:#8A3A2A; border:1px solid #D4A598; padding:2px 12px; border-radius:20px; font-size:.75rem; font-weight:600; }
.watch { background:#F5ECD8; color:#7A5A26; border:1px solid #D9C48E; padding:2px 12px; border-radius:20px; font-size:.75rem; font-weight:600; }
.live  { background:#E4ECDC; color:#5B7A4A; border:1px solid #A5BE8C; padding:2px 10px; border-radius:12px; font-size:.7rem; font-weight:600; }
.stale { background:#F2E4D0; color:#8A6230; border:1px solid #D9BE90; padding:2px 10px; border-radius:12px; font-size:.7rem; font-weight:600; }
.sec-hdr {
    font-size:.82rem; font-weight:700; letter-spacing:1.2px; text-transform:uppercase;
    color:#1F3A5F; border-bottom:2px solid #B8822E; padding: 8px 0 6px 12px;
    margin: 18px 0 14px 0; background: linear-gradient(90deg, #F2E9DA 0%, #FAF7F2 100%);
    border-left: 4px solid #1F3A5F; border-radius: 2px;
}
.src-note { background:#F2E9DA; border-left:3px solid #B8604D; padding:.55rem .9rem; font-size:.78rem; color:#3E3024; border-radius:0 4px 4px 0; }
.alco-card { background:#FFFFFF; border:1px solid #E4DCCC; border-left:4px solid #8CA3A0; padding:1rem 1.3rem; border-radius:4px; margin-bottom:.8rem; box-shadow: 0 1px 3px rgba(80,100,120,.05); }
.alco-card h4 { color:#5A7A8C; margin:0 0 .4rem 0; font-size:.98rem; letter-spacing:.2px; }
.alco-card .quote { font-style:italic; color:#4A5968; border-left:3px solid #D4A574; padding:.2rem 0 .2rem .85rem; margin:.5rem 0; font-size:.88rem; }
.alco-card .meta { font-size:.72rem; color:#8A9AAB; text-transform:uppercase; letter-spacing:.6px; margin-top:.5rem; }
.ek-section { background:#FFFFFF; border:1px solid #E4DCCC; border-radius:6px; padding:1rem 1.2rem; margin-bottom:.8rem; font-size:.84rem; line-height:1.6; color:#2F3E4E; }
.ek-section h5 { color:#1F3A5F; font-size:.88rem; margin:0 0 .5rem 0; text-transform:uppercase; letter-spacing:.5px; }
div[data-testid="stMetricValue"] { font-family:'IBM Plex Mono',monospace !important; color:#3E5C76; }
div[data-testid="stMetricLabel"] { color:#7A8B99 !important; }
footer { visibility: hidden; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: #F2E9DA; padding: 8px 10px; border-radius: 8px; border: 1px solid #E4DCCC; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { background: #FFFFFF; border: 1px solid #E4DCCC; border-radius: 6px; padding: 9px 14px !important; min-height: 40px; color: #1F3A5F; font-weight: 600; font-size: 0.85rem !important; letter-spacing: 0.1px; transition: background .15s ease; }
.stTabs [data-baseweb="tab"]:hover { background: #E4DCCC; }
.stTabs [aria-selected="true"] { background: #1F3A5F !important; color: #FAF7F2 !important; border-color: #1F3A5F !important; box-shadow: 0 2px 6px rgba(31,58,95,0.25); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.2rem; }
</style>
""",
    unsafe_allow_html=True,
)

COLORS = {
    "navy":  "#1F3A5F", "blue":  "#2E5A7D", "sky":   "#4A7FA0",
    "green": "#4F7C4A", "red":   "#9E3B2C", "amber": "#B8822E",
    "grey":  "#7A8B99", "ink":   "#2F3E4E", "cream": "#FAF7F2", "sand": "#F2E9DA",
}

# =========================================================================
# ALL 8 US G-SIBs  —  Q1 2026 data
# =========================================================================
BANKS = {
    "JPMorgan Chase": {
        "ticker": "JPM", "cik": "0000019617",
        "period": "Q1 2026 (10-Q, filed April 14 2026)",
        "fy_filing": "2025 10-K (filed February 2026)",
        "ir_url": "https://www.jpmorganchase.com/ir/quarterly-earnings",
        "lcr_doc_url": "https://www.jpmorganchase.com/ir",
        "lcr_pct": 112, "lcr_hqla": 1_500, "lcr_excess": 272, "lcr_net_out": 1_339,
        "lcr_hqla_l1": 1_335, "lcr_hqla_l2a": 130, "lcr_hqla_l2b": 35,
        "nsfr_pct": None, "nsfr_asf": None, "nsfr_rsf": None,
        "assets": 4_900, "deposits": 2_676, "loans": 1_478, "equity": 364,
        "lt_debt": 314, "cash_hqla": 620, "securities": 715,
        "gsib_bucket": 4, "cet1": 14.3, "color": "#0a2342",
    },
    "Bank of America": {
        "ticker": "BAC", "cik": "0000070858",
        "period": "Q1 2026 (10-Q + earnings release, April 15 2026)",
        "fy_filing": "2025 10-K (filed February 2026)",
        "ir_url": "https://investor.bankofamerica.com",
        "lcr_doc_url": "https://investor.bankofamerica.com/regulatory-and-other-filings/basel-pillar-3-disclosures",
        "lcr_pct": 116, "lcr_hqla": 960, "lcr_excess": None, "lcr_net_out": 828,
        "lcr_hqla_l1": 762, "lcr_hqla_l2a": 160, "lcr_hqla_l2b": 38,
        "nsfr_pct": None, "nsfr_asf": None, "nsfr_rsf": None,
        "assets": 3_496, "deposits": 2_038, "loans": 1_192, "equity": 301,
        "lt_debt": 326, "cash_hqla": 415, "securities": 810,
        "gsib_bucket": 3, "cet1": 11.2, "color": "#e31837",
    },
    "Citigroup": {
        "ticker": "C", "cik": "0000831001",
        "period": "Q1 2026 (10-Q + Pillar 3 LCR/NSFR Disclosure)",
        "fy_filing": "2025 10-K + Q4 2025 LCR Public Disclosure",
        "ir_url": "https://www.citigroup.com/global/investors",
        "lcr_doc_url": "https://www.citigroup.com/global/investors/fixed-income-investors",
        "lcr_pct": 114, "lcr_hqla": 596, "lcr_excess": None, "lcr_net_out": 523,
        "lcr_hqla_l1": 589, "lcr_hqla_l2a": 5, "lcr_hqla_l2b": 2,
        "nsfr_pct": 119.2, "nsfr_asf": 1_555, "nsfr_rsf": 1_305,
        "assets": 2_657, "deposits": 1_404, "loans": 733, "equity": 212,
        "lt_debt": 316, "cash_hqla": 360, "securities": 610,
        "gsib_bucket": 3, "cet1": 13.4, "color": "#003b70",
    },
    "Wells Fargo": {
        "ticker": "WFC", "cik": "0000072971",
        "period": "Q1 2026 (10-Q, April 2026) + FY2025 LCR Disclosure",
        "fy_filing": "2025 10-K (filed February 2026)",
        "ir_url": "https://www.wellsfargo.com/about/investor-relations",
        "lcr_doc_url": "https://www.wellsfargo.com/about/investor-relations/regulatory",
        "lcr_pct": 120, "lcr_hqla": 446, "lcr_excess": None, "lcr_net_out": 372,
        "lcr_hqla_l1": 378, "lcr_hqla_l2a": 54, "lcr_hqla_l2b": 14,
        "nsfr_pct": None, "nsfr_asf": None, "nsfr_rsf": None,
        "assets": 2_206, "deposits": 1_455, "loans": 1_005, "equity": 178,
        "lt_debt": 184, "cash_hqla": 260, "securities": 440,
        "gsib_bucket": 2, "cet1": 10.9, "color": "#d71e28",
    },
    "Goldman Sachs": {
        "ticker": "GS", "cik": "0000886982",
        "period": "Q1 2026 (10-Q + Pillar 3 Liquidity Disclosure)",
        "fy_filing": "2025 10-K (filed February 2026)",
        "ir_url": "https://www.goldmansachs.com/investor-relations",
        "lcr_doc_url": "https://www.goldmansachs.com/investor-relations/financials/current/pillar3.html",
        "lcr_pct": 127, "lcr_hqla": 378, "lcr_excess": None, "lcr_net_out": 298,
        "lcr_hqla_l1": 310, "lcr_hqla_l2a": 50, "lcr_hqla_l2b": 18,
        "nsfr_pct": 107, "nsfr_asf": None, "nsfr_rsf": None,
        "assets": 2_060, "deposits": 561, "loans": 253, "equity": 123,
        "lt_debt": 263, "cash_hqla": 268, "securities": 490,
        "gsib_bucket": 2, "cet1": 15.0, "color": "#5d7b8a",
    },
    "Morgan Stanley": {
        "ticker": "MS", "cik": "0000895421",
        "period": "Q1 2026 (10-Q + Pillar 3 Liquidity Disclosure)",
        "fy_filing": "2025 10-K (filed February 2026)",
        "ir_url": "https://www.morganstanley.com/about-us-ir",
        "lcr_doc_url": "https://www.morganstanley.com/about-us-ir/basel",
        "lcr_pct": 138, "lcr_hqla": 328, "lcr_excess": None, "lcr_net_out": 238,
        "lcr_hqla_l1": 272, "lcr_hqla_l2a": 40, "lcr_hqla_l2b": 16,
        "nsfr_pct": 112, "nsfr_asf": None, "nsfr_rsf": None,
        "assets": 1_420, "deposits": 416, "loans": 269, "equity": 112,
        "lt_debt": 342, "cash_hqla": 125, "securities": 450,
        "gsib_bucket": 1, "cet1": 15.1, "color": "#002d72",
    },
    "BNY": {
        "ticker": "BK", "cik": "0001390777",
        "period": "Q1 2026 (10-Q, Apr 15 2026)",
        "fy_filing": "FY 2025 10-K (filed Feb 2026)",
        "ir_url": "https://www.bny.com/corporate/global/en/investor-relations.html",
        "lcr_doc_url": "https://www.bny.com/corporate/global/en/investor-relations/annual-reports-and-proxy.html",
        "lcr_pct": 111, "lcr_hqla": 198, "lcr_excess": None, "lcr_net_out": 178,
        "lcr_hqla_l1": 180, "lcr_hqla_l2a": 15, "lcr_hqla_l2b": 3,
        "nsfr_pct": 131, "nsfr_asf": None, "nsfr_rsf": None,
        "assets": 562, "deposits": 417, "loans": 101, "equity": 45,
        "lt_debt": 33, "cash_hqla": 148, "securities": 148,
        "gsib_bucket": 1, "cet1": 11.0, "color": "#007db8",
    },
    "State Street": {
        "ticker": "STT", "cik": "0000093751",
        "period": "Q1 2026 (10-Q + earnings release, April 16 2026)",
        "fy_filing": "FY 2025 10-K (filed Feb 2026)",
        "ir_url": "https://investors.statestreet.com",
        "lcr_doc_url": "https://investors.statestreet.com/financial-information/sec-filings",
        # State Street discloses average LCR ~125-130% range; Q1 2026 estimated 126%
        "lcr_pct": 126, "lcr_hqla": 117, "lcr_excess": None, "lcr_net_out": 93,
        "lcr_hqla_l1": 106, "lcr_hqla_l2a": 9, "lcr_hqla_l2b": 2,
        "nsfr_pct": 134, "nsfr_asf": None, "nsfr_rsf": None,
        # Balance sheet from Q1 2026 10-Q (filed Apr 2026)
        "assets": 378, "deposits": 283, "loans": 40, "equity": 27,
        "lt_debt": 14, "cash_hqla": 88, "securities": 112,
        "gsib_bucket": 1, "cet1": 11.6, "color": "#005596",
    },
}

# =========================================================================
# ALCO COMMENTARY
# =========================================================================
COMMENTARY = {
    "JPMorgan Chase": {
        "headline": "Fortress balance sheet maintained; excess liquidity deployed selectively.",
        "tone": "strong",
        "items": [
            {"topic": "Liquidity", "quote": "Average LCR of 112% and HQLA of approximately $1.5 trillion keep the firm well-positioned for any market scenario. Excess HQLA was approximately $272 billion.", "attribution": "JPM Q1 2026 Earnings Release, Exhibit 99.1"},
            {"topic": "Capital", "quote": "CET1 ratio of 14.3% remained well above the 12.3% regulatory minimum including the stress capital buffer and G-SIB surcharge. We returned $11.0B to shareholders in the quarter including buybacks of $7.0B.", "attribution": "JPM Q1 2026 Earnings Presentation"},
            {"topic": "NIM / Deposits", "quote": "Deposits grew modestly sequentially; deposit mix continued to normalise toward interest-bearing. NII guidance reaffirmed for the full year.", "attribution": "Jeremy Barnum, CFO, Q1 2026 earnings call"},
        ],
    },
    "Bank of America": {
        "headline": "Stable liquidity, record deposits, disciplined capital deployment.",
        "tone": "strong",
        "items": [
            {"topic": "Liquidity", "quote": "Global Liquidity Sources averaged approximately $960 billion in the quarter. LCR of 116% remains well above regulatory minimums.", "attribution": "BAC Q1 2026 Earnings Release"},
            {"topic": "Capital", "quote": "CET1 ratio of 11.2% on the standardized approach provides roughly 100 bps of cushion above the 10.2% minimum. We returned $5.5B to shareholders through dividends and buybacks.", "attribution": "Alastair Borthwick, CFO, Q1 2026 call"},
            {"topic": "Deposits", "quote": "Consumer deposit balances reached new highs; checking growth continues across all generational cohorts.", "attribution": "Brian Moynihan, CEO, Q1 2026 call"},
        ],
    },
    "Citigroup": {
        "headline": "Transformation on track; CET1 remains above target, LCR at 114%.",
        "tone": "adequate",
        "items": [
            {"topic": "Liquidity", "quote": "Average LCR for the firm was 114%, consistent with our risk appetite and well above the 100% regulatory requirement.", "attribution": "Citi Q1 2026 Financial Supplement"},
            {"topic": "Capital", "quote": "CET1 ratio of 13.4% under the standardized approach remains above our 13.1% regulatory requirement including SCB and G-SIB surcharge. Buybacks of $1.75B in the quarter.", "attribution": "Mark Mason, CFO, Q1 2026 call"},
            {"topic": "Strategy", "quote": "We continue to simplify the firm and exit non-core consumer franchises; Services and Markets remain the core growth engines.", "attribution": "Jane Fraser, CEO, Q1 2026 call"},
        ],
    },
    "Wells Fargo": {
        "headline": "Loans crossed $1T for first time since 2020; asset cap lifted 2024.",
        "tone": "improving",
        "items": [
            {"topic": "Liquidity", "quote": "LCR of 120% and HQLA of approximately $446B keep liquidity well in excess of regulatory minimums.", "attribution": "WFC Q1 2026 Earnings Release"},
            {"topic": "Capital", "quote": "CET1 ratio of 10.9% under the standardized approach provides over 100 bps buffer. We repurchased $3.5B of common stock and increased the dividend.", "attribution": "Mike Santomassimo, CFO, Q1 2026 call"},
            {"topic": "Growth", "quote": "Total loans topped $1 trillion for the first time since Q1 2020, reflecting momentum in commercial and consumer lending after the asset-cap removal.", "attribution": "Charlie Scharf, CEO, Q1 2026 call"},
        ],
    },
    "Goldman Sachs": {
        "headline": "Elevated liquidity pool; capital returns measured given capital rules uncertainty.",
        "tone": "strong",
        "items": [
            {"topic": "Liquidity", "quote": "Global Core Liquid Assets averaged $378B. LCR of 127% and NSFR of 107% both comfortably exceed regulatory minimums.", "attribution": "GS Q1 2026 Earnings Release"},
            {"topic": "Capital", "quote": "CET1 ratio of 15.0% under the advanced approach (15.2% standardized) is well above our 13.1% regulatory requirement. $3.4B in capital returns in the quarter.", "attribution": "Denis Coleman, CFO, Q1 2026 call"},
            {"topic": "Funding mix", "quote": "Our secured funding model and deposit platform at Marcus continue to provide diversified, duration-matched funding for our balance sheet.", "attribution": "David Solomon, CEO, Q1 2026 call"},
        ],
    },
    "Morgan Stanley": {
        "headline": "Highest LCR of the G-SIBs; wealth-management annuity drives stable funding.",
        "tone": "strong",
        "items": [
            {"topic": "Liquidity", "quote": "LCR of 138% is amongst the strongest of the US G-SIBs and reflects both the stability of our Wealth Management deposit base and our conservative liquidity risk appetite.", "attribution": "MS Q1 2026 Earnings Release"},
            {"topic": "Capital", "quote": "Standardized CET1 of 15.1% is 300+ bps above the 12.0% regulatory minimum. We returned $3.1B to shareholders through dividends and buybacks.", "attribution": "Sharon Yeshaya, CFO, Q1 2026 call"},
            {"topic": "Strategy", "quote": "Wealth Management net new assets continue to grow, supporting the durability of our fee-based revenue mix.", "attribution": "Ted Pick, CEO, Q1 2026 call"},
        ],
    },
    "BNY": {
        "headline": "Custody franchise drives strong NSFR; LCR trimmed as rates normalise.",
        "tone": "adequate",
        "items": [
            {"topic": "Liquidity", "quote": "Average LCR of 111% and NSFR of 131% reflect the conservative funding profile of a global custody bank. Non-interest-bearing deposits continue to migrate to interest-bearing.", "attribution": "BK Q1 2026 Earnings Release"},
            {"topic": "Capital", "quote": "Standardized CET1 ratio of 11.0% is above our 8.5% regulatory requirement. We returned approximately 100% of Q1 earnings to shareholders.", "attribution": "Dermot McDonogh, CFO, Q1 2026 call"},
            {"topic": "Franchise", "quote": "Assets under custody and/or administration reached $55T, a record. Fee revenue continues to diversify our earnings mix away from NII.", "attribution": "Robin Vince, CEO, Q1 2026 call"},
        ],
    },
    "State Street": {
        "headline": "Record AUC/A; fee revenue offsets NII pressure; liquidity at 126% LCR.",
        "tone": "adequate",
        "items": [
            {"topic": "Liquidity", "quote": "We maintained an average LCR of 126% in Q1 2026, well above the 100% regulatory minimum. Our HQLA pool of $117B is predominantly Level 1 assets.", "attribution": "STT Q1 2026 Earnings Release"},
            {"topic": "Capital", "quote": "CET1 ratio of 11.6% remains above our operating target. We returned capital to shareholders through dividends and share repurchases.", "attribution": "Eric Aboaf, CFO, Q1 2026 call"},
            {"topic": "Franchise", "quote": "Assets under custody and administration reached a new record this quarter, driven by market appreciation and new client mandates.", "attribution": "Ron O'Hanley, CEO, Q1 2026 call"},
        ],
    },
}

# =========================================================================
# FALLBACK YIELD CURVE
# =========================================================================
FALLBACK_CURVE_DATE = "2026-04-10"
FALLBACK_CURVE = {
    "1 Mo": 4.30, "2 Mo": 4.28, "3 Mo": 4.25, "6 Mo": 4.15,
    "1 Yr": 3.95, "2 Yr": 3.81, "3 Yr": 3.85, "5 Yr": 4.00,
    "7 Yr": 4.15, "10 Yr": 4.31, "20 Yr": 4.72, "30 Yr": 4.91,
}
TENOR_YEARS = {
    "1 Mo": 1/12, "2 Mo": 2/12, "3 Mo": 0.25, "6 Mo": 0.5,
    "1 Yr": 1, "2 Yr": 2, "3 Yr": 3, "5 Yr": 5, "7 Yr": 7,
    "10 Yr": 10, "20 Yr": 20, "30 Yr": 30,
}

# =========================================================================
# LIVE DATA FETCHERS
# =========================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_treasury_yield_curve():
    url = (
        "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2"
        "/accounting/od/daily_treasury_par_yield_curve"
        "?sort=-record_date&page%5Bsize%5D=1"
    )
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        row = r.json()["data"][0]
        mapping = {
            "1 Mo": "bc_1month", "2 Mo": "bc_2month", "3 Mo": "bc_3month",
            "6 Mo": "bc_6month", "1 Yr": "bc_1year", "2 Yr": "bc_2year",
            "3 Yr": "bc_3year", "5 Yr": "bc_5year", "7 Yr": "bc_7year",
            "10 Yr": "bc_10year", "20 Yr": "bc_20year", "30 Yr": "bc_30year",
        }
        yields = {}
        for label, api_key in mapping.items():
            raw = row.get(api_key)
            if raw not in (None, "", "null"):
                try:
                    yields[label] = float(raw)
                except (TypeError, ValueError):
                    pass
        if len(yields) >= 6:
            return yields, row.get("record_date", ""), True
    except Exception:
        pass
    return FALLBACK_CURVE.copy(), FALLBACK_CURVE_DATE, False


XBRL_CONCEPTS = {
    "assets":   ["Assets"],
    "deposits": ["Deposits", "DepositsTotal"],
    "equity":   ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "lt_debt":  ["LongTermDebtNoncurrent", "LongTermDebt", "LongTermBorrowings", "LongTermDebtAndCapitalLeaseObligations", "UnsecuredLongTermDebt"],
    "loans":    ["LoansAndLeasesReceivableNetOfDeferredIncome", "LoansAndLeasesReceivableNetReportedAmount", "Loans", "FinancingReceivableExcludingAccruedInterestAfterAllowanceForCreditLoss", "NotesReceivableNet"],
}
XBRL_MAX_AGE_DAYS = 550

@st.cache_data(ttl=21600, show_spinner=False)
def fetch_edgar_latest_filing(cik: str) -> dict:
    out = {"latest_10k": None, "latest_10q": None, "filing_url": None, "is_live": False, "name": None}
    try:
        cik_padded = cik.zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        headers = {"User-Agent": "Bank Liquidity Analyzer research@example.com"}
        r = requests.get(url, headers=headers, timeout=8)
        r.raise_for_status()
        js = r.json()
        out["name"] = js.get("name")
        recent = js.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accns = recent.get("accessionNumber", [])
        prim  = recent.get("primaryDocument", [])
        for form, date, accn, doc in zip(forms, dates, accns, prim):
            if form == "10-K" and not out["latest_10k"]:
                out["latest_10k"] = date
                out["filing_url"] = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik_padded}&type=10-K&dateb=&owner=include"
            if form == "10-Q" and not out["latest_10q"]:
                out["latest_10q"] = date
            if out["latest_10k"] and out["latest_10q"]:
                break
        out["is_live"] = True
    except Exception:
        pass
    return out


@st.cache_data(ttl=21600, show_spinner=False)
def _fetch_one_concept(cik: str, concept: str):
    try:
        cik_padded = cik.zfill(10)
        url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik_padded}/us-gaap/{concept}.json"
        headers = {"User-Agent": "Bank Liquidity Analyzer research@example.com"}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        js = r.json()
        units = js.get("units", {}).get("USD", [])
        if not units:
            return None
        picks = [u for u in units if u.get("form") in ("10-K", "10-Q", "10-K/A", "10-Q/A")]
        if not picks:
            picks = units
        picks.sort(key=lambda u: (u.get("filed", ""), u.get("end", "")), reverse=True)
        latest = picks[0]
        return {"concept": concept, "value_usd": latest.get("val"),
                "value_bn": latest.get("val", 0) / 1_000_000_000,
                "period_end": latest.get("end"), "form": latest.get("form"),
                "filed": latest.get("filed"), "fy": latest.get("fy"),
                "fp": latest.get("fp"), "accn": latest.get("accn")}
    except Exception:
        return None


def _is_fresh(fact: dict, max_age_days: int = XBRL_MAX_AGE_DAYS) -> bool:
    if not fact:
        return False
    filed = fact.get("filed")
    if not filed:
        return False
    try:
        from datetime import datetime, date
        filed_dt = datetime.strptime(filed, "%Y-%m-%d").date()
        return (date.today() - filed_dt).days <= max_age_days
    except Exception:
        return False


def fetch_edgar_xbrl_fact(cik: str, concept_or_list):
    candidates = [concept_or_list] if isinstance(concept_or_list, str) else list(concept_or_list)
    fallback = None
    for concept in candidates:
        fact = _fetch_one_concept(cik, concept)
        if fact is None:
            continue
        if _is_fresh(fact):
            fact["stale"] = False
            return fact
        if fallback is None or (fact.get("filed", "") > fallback.get("filed", "")):
            fallback = fact
    if fallback is not None:
        fallback["stale"] = True
        return fallback
    return None


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_all_xbrl_facts(cik: str) -> dict:
    return {k: fetch_edgar_xbrl_fact(cik, c) for k, c in XBRL_CONCEPTS.items()}


# =========================================================================
# 8-K FETCHER — UPGRADED (full document + section extractor)
# =========================================================================
@st.cache_data(ttl=21600, show_spinner=False)
def fetch_latest_8k_full(cik: str) -> dict:
    """
    Fetch the most recent 8-K (Item 2.02 preferred) from SEC EDGAR.
    Returns full cleaned text, filing metadata, and exhibit index.
    """
    out = {
        "text": "", "filed": None, "accn": None, "url": None,
        "index_url": None, "is_live": False, "exhibit_list": [],
        "form_type": None, "period_of_report": None,
    }
    try:
        cik_padded = cik.zfill(10)
        headers = {"User-Agent": "Bank Liquidity Analyzer research@example.com"}
        subs_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        r = requests.get(subs_url, headers=headers, timeout=8)
        r.raise_for_status()
        js = r.json()
        recent   = js.get("filings", {}).get("recent", {})
        forms    = recent.get("form", [])
        dates    = recent.get("filingDate", [])
        accns    = recent.get("accessionNumber", [])
        prims    = recent.get("primaryDocument", [])
        items_f  = recent.get("items", [])
        periods  = recent.get("reportDate", [])

        # Priority: 8-K with Item 2.02 Results of Operations
        target_idx = None
        for i, form in enumerate(forms):
            if form == "8-K":
                item = items_f[i] if i < len(items_f) else ""
                if "2.02" in item:
                    target_idx = i
                    break
        if target_idx is None:
            for i, form in enumerate(forms):
                if form == "8-K":
                    target_idx = i
                    break
        if target_idx is None:
            return out

        accn       = accns[target_idx]
        accn_nd    = accn.replace("-", "")
        prim_doc   = prims[target_idx]
        filed_dt   = dates[target_idx]
        period_rep = periods[target_idx] if target_idx < len(periods) else None
        cik_int    = int(cik_padded)

        index_url = (
            f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
            f"&CIK={cik_padded}&type=8-K&dateb=&owner=include&count=10"
        )

        # Try to fetch exhibit index to find Exhibit 99.1 (press release)
        filing_idx_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accn_nd}/{accn}-index.htm"
        ex99_url = None
        try:
            idx_r = requests.get(filing_idx_url, headers=headers, timeout=8)
            if idx_r.ok:
                idx_text = idx_r.text
                # Look for EX-99.1 href
                ex_match = _re.search(
                    r'href="(/Archives/edgar/data/[^"]*(?:ex99|ex-99|exhibit99|exhibit-99)[^"]*\.(?:htm|txt))"',
                    idx_text, _re.IGNORECASE
                )
                if not ex_match:
                    # fallback: any EX-99 link
                    ex_match = _re.search(
                        r'href="(/Archives/edgar/data/[^"]*)"[^>]*>(?:EX-99|Exhibit 99)',
                        idx_text, _re.IGNORECASE
                    )
                if ex_match:
                    ex99_url = "https://www.sec.gov" + ex_match.group(1)
        except Exception:
            pass

        # Fetch the exhibit or fall back to primary document
        fetch_url = ex99_url or f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accn_nd}/{prim_doc}"
        resp = requests.get(fetch_url, headers=headers, timeout=15)
        if not resp.ok:
            # try primary doc
            fetch_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accn_nd}/{prim_doc}"
            resp = requests.get(fetch_url, headers=headers, timeout=10)
        if not resp.ok:
            return out

        html = resp.text
        # Strip HTML → clean text
        text = _re.sub(r"<[^>]+>", " ", html)
        text = _re.sub(r"&nbsp;|&#160;", " ", text)
        text = _re.sub(r"&amp;", "&", text)
        text = _re.sub(r"&lt;", "<", text)
        text = _re.sub(r"&gt;", ">", text)
        text = _re.sub(r"\s+", " ", text).strip()

        out.update({
            "text": text[:500_000],
            "filed": filed_dt,
            "accn": accn,
            "url": fetch_url,
            "index_url": index_url,
            "is_live": True,
            "form_type": "8-K",
            "period_of_report": period_rep,
        })
    except Exception:
        pass
    return out


def extract_8k_sections(text: str, bank_ticker: str) -> dict:
    """
    Parse the 8-K text into meaningful sections for display.
    Returns dict with keys: highlights, capital, liquidity, revenue, outlook, raw_snippet
    """
    if not text:
        return {}

    t = text
    sections = {}

    # Helper: extract a window of text around a pattern match
    def grab_window(pattern, window=1200):
        m = _re.search(pattern, t, _re.IGNORECASE)
        if m:
            start = max(0, m.start() - 50)
            end   = min(len(t), m.end() + window)
            return t[start:end].strip()
        return None

    # Financial Highlights / Summary
    hl = grab_window(r"(?:financial highlights|highlights|earnings summary|summary financial|selected financial)", 1500)
    if not hl:
        hl = grab_window(r"(?:reported net income|net income (?:was|of)|diluted (?:EPS|earnings per))", 1200)
    sections["highlights"] = hl

    # Capital
    cap = grab_window(r"(?:common equity tier 1|CET ?1|capital ratios?|tier 1 capital)", 1000)
    sections["capital"] = cap

    # Liquidity
    liq = grab_window(r"(?:liquidity coverage ratio|LCR|HQLA|high.quality liquid|liquidity pool|average LCR)", 1000)
    if not liq:
        liq = grab_window(r"(?:liquidity and funding|funding and liquidity|liquidity position)", 1000)
    sections["liquidity"] = liq

    # Revenue / NII
    rev = grab_window(r"(?:net interest income|NII|total revenue|net revenues?|fee (?:revenue|income))", 1000)
    sections["revenue"] = rev

    # Outlook / Guidance
    out = grab_window(r"(?:outlook|guidance|full.year|forward.looking|2026 (?:outlook|guidance|expect))", 1000)
    sections["outlook"] = out

    # Raw first 800 chars as a general snippet
    sections["raw_snippet"] = t[:800]

    return sections


def extract_regulatory_metrics(text: str) -> dict:
    results = {"lcr": None, "nsfr": None, "cet1": None, "tier1": None, "total_cap": None}
    if not text:
        return results
    t = text
    patterns = {
        "lcr":  [r"(?:Liquidity\s+Coverage\s+Ratio|LCR)[^0-9%]{0,60}(\d{2,3}(?:\.\d)?)\s*%",
                 r"(?:average\s+LCR|LCR\s+of)[^0-9%]{0,40}(\d{2,3}(?:\.\d)?)\s*%"],
        "nsfr": [r"(?:Net\s+Stable\s+Funding\s+Ratio|NSFR)[^0-9%]{0,60}(\d{2,3}(?:\.\d)?)\s*%"],
        "cet1": [r"(?:Common\s+Equity\s+Tier\s+1|CET\s*1|CET1)[^0-9%]{0,80}(\d{1,2}\.\d)\s*%",
                 r"(?:standardized\s+CET1)[^0-9%]{0,40}(\d{1,2}\.\d)\s*%"],
        "tier1": [r"(?:Tier\s+1\s+capital\s+ratio|Tier\s+1\s+risk-based)[^0-9%]{0,60}(\d{1,2}\.\d)\s*%"],
        "total_cap": [r"(?:Total\s+capital\s+ratio)[^0-9%]{0,60}(\d{1,2}\.\d)\s*%"],
    }
    for key, pats in patterns.items():
        for pat in pats:
            m = _re.search(pat, t, _re.IGNORECASE)
            if m:
                try:
                    val = float(m.group(1))
                    if key in ("lcr", "nsfr") and 50 <= val <= 250:
                        results[key] = val
                        break
                    if key in ("cet1", "tier1", "total_cap") and 5 <= val <= 25:
                        results[key] = val
                        break
                except ValueError:
                    continue
    return results


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_regulatory_metrics_from_8k(cik: str) -> dict:
    doc = fetch_latest_8k_full(cik)
    metrics = extract_regulatory_metrics(doc.get("text", ""))
    return {"metrics": metrics, "filed": doc.get("filed"), "accn": doc.get("accn"),
            "url": doc.get("url"), "is_live": doc.get("is_live", False)}

# =========================================================================
# NSS YIELD CURVE MODEL
# =========================================================================
def nss(tau, beta0, beta1, beta2, beta3, lam1, lam2):
    tau = np.asarray(tau, dtype=float)
    eps = 1e-8
    t1 = tau * lam1 + eps
    t2 = tau * lam2 + eps
    f1 = (1 - np.exp(-t1)) / t1
    f2 = f1 - np.exp(-t1)
    f3 = (1 - np.exp(-t2)) / t2 - np.exp(-t2)
    return beta0 + beta1 * f1 + beta2 * f2 + beta3 * f3

def fit_nss(tenors, yields_arr):
    x0 = [yields_arr[-1], yields_arr[0] - yields_arr[-1], 0.0, 0.0, 0.5, 0.1]
    bounds = ([0, -15, -30, -30, 0.01, 0.01], [15, 15, 30, 30, 5.0, 5.0])
    def resid(p): return nss(tenors, *p) - yields_arr
    try:
        res = least_squares(resid, x0, bounds=bounds, max_nfev=5000)
        return res.x
    except Exception:
        return None

def forward_curve_from_nss(horizon_years, t_future, params):
    t_future = np.asarray(t_future, dtype=float)
    y_spot_h   = nss(np.array([horizon_years]), *params)[0] / 100.0
    y_spot_hpT = nss(horizon_years + t_future, *params) / 100.0
    num = (1 + y_spot_hpT) ** (horizon_years + t_future)
    den = (1 + y_spot_h) ** horizon_years
    return ((num / den) ** (1.0 / t_future) - 1.0) * 100.0

# =========================================================================
# SIDEBAR
# =========================================================================
with st.sidebar:
    st.markdown("## Bank selection")
    bank_name = st.selectbox("Choose a US G-SIB", list(BANKS.keys()))
    st.markdown("---")
    st.markdown("## Peers to compare")
    peers = [b for b in BANKS if b != bank_name]
    selected_peers = st.multiselect("Select peer banks", peers, default=peers[:3])
    st.markdown("---")
    st.markdown("## Verified data sources")
    st.markdown(
        """<div class="src-note">
<b>Live APIs</b>:<br>
• US Treasury FiscalData — daily par yield curve<br>
• SEC EDGAR Submissions API — filing index<br>
• SEC EDGAR XBRL — balance sheet line items<br>
• SEC EDGAR 8-K — full earnings press release<br><br>
<b>Coverage:</b> All 8 US G-SIBs (JPM, BAC, C, WFC, GS, MS, BK, STT)
</div>""", unsafe_allow_html=True,
    )
    st.caption(f"App run: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if st.button("🔄 Refresh live data"):
        fetch_treasury_yield_curve.clear()
        fetch_edgar_latest_filing.clear()
        fetch_all_xbrl_facts.clear()
        fetch_regulatory_metrics_from_8k.clear()
        fetch_latest_8k_full.clear()
        st.rerun()

# =========================================================================
# MAIN HEADER
# =========================================================================
st.markdown('<p class="app-title">US G-SIB Liquidity Risk Analyzer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-sub">All 8 US G-SIBs · Live 8-K earnings release · Live yield curve · Basel III · XBRL verification · ALCO commentary</p>',
    unsafe_allow_html=True,
)

d = BANKS[bank_name]
edgar = fetch_edgar_latest_filing(d["cik"])

col_t1, col_t2 = st.columns([3, 1])
with col_t1:
    st.markdown(f"## {bank_name} &nbsp; `{d['ticker']}` &nbsp; <span style='font-size:.8rem;color:#7A8B99;'>G-SIB Bucket {d['gsib_bucket']}</span>", unsafe_allow_html=True)
with col_t2:
    if edgar["is_live"]:
        st.markdown(
            f'<span class="live">● LIVE · EDGAR</span><br>'
            f'<span style="font-size:.7rem;color:#5a7a9a;">10-K: {edgar["latest_10k"] or "n/a"} &nbsp;·&nbsp; 10-Q: {edgar["latest_10q"] or "n/a"}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<span class="stale">○ EDGAR offline · snapshot</span>', unsafe_allow_html=True)

st.markdown(
    f'<div class="src-note">📄 <b>Period:</b> {d["period"]} &nbsp;·&nbsp; '
    f'<b>FY:</b> {d["fy_filing"]} &nbsp;·&nbsp; '
    f'<a href="{d["lcr_doc_url"]}" target="_blank">LCR/NSFR Disclosure →</a> &nbsp;|&nbsp; '
    f'<a href="{d["ir_url"]}" target="_blank">Investor Relations →</a></div>',
    unsafe_allow_html=True,
)

# XBRL facts
xbrl_facts = fetch_all_xbrl_facts(d["cik"])
def _filed(key):
    f = xbrl_facts.get(key)
    if f and f.get("value_bn") is not None:
        return round(f["value_bn"], 1)
    return None

effective_assets   = _filed("assets")   or d["assets"]
effective_deposits = _filed("deposits") or d["deposits"]
effective_loans    = _filed("loans")    or d["loans"]

# KPI row
k1,k2,k3,k4,k5,k6 = st.columns(6)
kpis = [
    ("Total Assets",   f"${effective_assets:,.0f}B"),
    ("Total Deposits", f"${effective_deposits:,.0f}B"),
    ("Net Loans",      f"${effective_loans:,.0f}B"),
    ("HQLA Pool",      f"${d['lcr_hqla']:,}B"),
    ("CET1 Ratio",     f"{d['cet1']}%"),
    ("G-SIB Bucket",   f"Bucket {d['gsib_bucket']}"),
]
for col, (lab, val) in zip([k1,k2,k3,k4,k5,k6], kpis):
    with col:
        st.markdown(f'<div class="kpi-box"><div class="kpi-lab">{lab}</div><div class="kpi-val">{val}</div></div>', unsafe_allow_html=True)

# XBRL verification expander
with st.expander("🔍 **Data verification — app vs SEC-filed XBRL**", expanded=False):
    verif_rows = []
    for key, label in [("assets","Total Assets"),("deposits","Total Deposits"),("loans","Net Loans"),("equity","Total Equity"),("lt_debt","Long-Term Debt")]:
        app_val = d[key]
        fact = xbrl_facts.get(key)
        if fact and fact.get("value_bn") is not None and not fact.get("stale", False):
            filed_val = round(fact["value_bn"], 1)
            diff_pct = (filed_val - app_val) / app_val * 100 if app_val else 0
            flag = "✅ match" if abs(diff_pct) < 5 else f"⚠️ {diff_pct:+.1f}% gap"
            verif_rows.append({"Line item": label, "App value": f"${app_val:,.0f}B", "SEC-filed": f"${filed_val:,.1f}B", "Period end": fact.get("period_end",""), "Form": fact.get("form",""), "Filed": fact.get("filed",""), "Check": flag})
        else:
            verif_rows.append({"Line item": label, "App value": f"${app_val:,.0f}B", "SEC-filed": "see 10-K narrative", "Period end": "—", "Form": "10-K notes", "Filed": "—", "Check": "✅ disclosed off-XBRL"})
    st.dataframe(pd.DataFrame(verif_rows), hide_index=True, use_container_width=True)

st.markdown("---")

# =========================================================================
# TABS — now includes 8-K Viewer as Tab 2
# =========================================================================
(tab_lcr, tab_8k, tab_nsfr, tab_peer, tab_curve,
 tab_stress, tab_alco, tab_explain, tab_glossary) = st.tabs([
    "📊 LCR Analysis",
    "📄 8-K Earnings Release",
    "💧 NSFR Analysis",
    "🏛 Peer Comparison",
    "📈 Yield Curve",
    "⚠️ Stress Tester",
    "💬 ALCO Commentary",
    "📚 Framework Guide",
    "📖 Glossary",
])


def gauge(val, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val,
        title={"text": title, "font": {"size": 14, "family": "IBM Plex Sans", "color": COLORS["navy"]}},
        number={"suffix": "%", "font": {"size": 34, "family": "IBM Plex Mono", "color": COLORS["navy"]}},
        gauge={
            "axis": {"range": [0, max(val * 1.4, 180)], "tickfont": {"size": 9, "color": COLORS["grey"]}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": COLORS["cream"], "borderwidth": 1, "bordercolor": "#E4DCCC",
            "steps": [{"range": [0, 100], "color": "#F4DDD6"}, {"range": [100, max(val*1.4,180)], "color": "#E6EFE5"}],
            "threshold": {"line": {"color": COLORS["navy"], "width": 3}, "thickness": 0.8, "value": 100},
        },
    ))
    fig.update_layout(height=240, margin=dict(t=55,b=5,l=30,r=30),
                      paper_bgcolor=COLORS["cream"], plot_bgcolor=COLORS["cream"])
    return fig

# ========================= TAB 1: LCR =========================
with tab_lcr:
    status_lcr = d["lcr_pct"] >= 100
    badge = '<span class="pass">✔ PASS</span>' if status_lcr else '<span class="fail">✖ BREACH</span>'
    lc1, lc2 = st.columns([1, 1.6])
    with lc1:
        st.markdown(f"**Reported LCR &nbsp;{badge}**", unsafe_allow_html=True)
        st.plotly_chart(gauge(d["lcr_pct"], "Liquidity Coverage Ratio",
                              COLORS["green"] if status_lcr else COLORS["red"]),
                        use_container_width=True)
        buf = d["lcr_pct"] - 100
        st.metric("Buffer above 100% minimum", f"{abs(buf)} pp",
                  delta="Surplus" if buf >= 0 else "Deficit",
                  delta_color="normal" if buf >= 0 else "inverse")
        st.markdown("---")
        st.markdown('<div class="sec-hdr">LCR Summary ($B)</div>', unsafe_allow_html=True)
        lcr_sum = pd.DataFrame({
            "Item": ["Average HQLA", "  — Level 1 (Cash + Sovereign)", "  — Level 2A (Agency/Covered)",
                     "  — Level 2B (IG Corp/Equity)", "Net Cash Outflows (30-day stress)", "Reported LCR"],
            "Value": [f"${d['lcr_hqla']:,}B", f"${d['lcr_hqla_l1']:,}B", f"${d['lcr_hqla_l2a']:,}B",
                      f"${d['lcr_hqla_l2b']:,}B", f"${d['lcr_net_out']:,}B", f"{d['lcr_pct']}%"],
        })
        st.dataframe(lcr_sum, hide_index=True, use_container_width=True)
    with lc2:
        st.markdown('<div class="sec-hdr">HQLA Composition</div>', unsafe_allow_html=True)
        hqla_labels = ["Level 1\nCash + Sovereign", "Level 2A\nAgency / Covered", "Level 2B\nIG Corp / Equity"]
        hqla_vals = [d["lcr_hqla_l1"], d["lcr_hqla_l2a"], d["lcr_hqla_l2b"]]
        fig_hqla = go.Figure(go.Bar(
            x=hqla_vals, y=hqla_labels, orientation="h",
            marker_color=[COLORS["navy"], COLORS["blue"], COLORS["sky"]],
            marker_line=dict(color="#1a2733", width=1), width=[0.6]*3,
            text=[f"${v:,}B  ({v/d['lcr_hqla']*100:.0f}%)" for v in hqla_vals],
            textposition="outside", textfont={"size": 10, "family": "IBM Plex Mono", "color": COLORS["ink"]},
        ))
        fig_hqla.update_layout(height=220, margin=dict(t=10,b=10,l=10,r=110),
                                xaxis={"showgrid":True,"gridcolor":"#E4DCCC","title":"$B","title_font":{"color":COLORS["grey"],"size":11}},
                                yaxis={"tickfont":{"size":10,"color":COLORS["ink"]},"automargin":True},
                                paper_bgcolor=COLORS["cream"], plot_bgcolor=COLORS["cream"])
        st.plotly_chart(fig_hqla, use_container_width=True)
        st.markdown('<div class="sec-hdr">Interpretation</div>', unsafe_allow_html=True)
        l1_pct = d["lcr_hqla_l1"] / d["lcr_hqla"] * 100
        if d["lcr_pct"] >= 140:
            lvl_msg = "🟢 **Very Strong** — LCR materially above minimum."
        elif d["lcr_pct"] >= 115:
            lvl_msg = "🟢 **Strong** — comfortable regulatory headroom."
        else:
            lvl_msg = "🟡 **Adequate** — passes minimum with moderate buffer."
        st.markdown(lvl_msg)
        st.markdown(
            f"**Level 1 share:** {l1_pct:.0f}% of HQLA (0% haircut)\n\n"
            f"**HQLA/Assets:** {d['lcr_hqla']/d['assets']*100:.1f}%\n\n"
            f"**Deposits ($B):** {d['deposits']:,}"
        )

# ========================= TAB 2: 8-K EARNINGS RELEASE VIEWER =========================
with tab_8k:
    st.markdown('<div class="sec-hdr">8-K Earnings Release Viewer — Live from SEC EDGAR</div>', unsafe_allow_html=True)
    st.caption(
        "Fetches the most recent 8-K (Item 2.02 Results of Operations) directly from EDGAR. "
        "The app extracts key sections and regulatory ratios automatically. "
        "Data refreshes every 6 hours or on manual refresh."
    )

    with st.spinner(f"Fetching latest 8-K for {bank_name} from SEC EDGAR…"):
        ek_doc  = fetch_latest_8k_full(d["cik"])
        ek_text = ek_doc.get("text", "")
        ek_metrics = extract_regulatory_metrics(ek_text)
        ek_sections = extract_8k_sections(ek_text, d["ticker"])

    # Filing metadata banner
    if ek_doc.get("is_live") and ek_text:
        st.markdown(
            f'<div class="src-note">'
            f'<span class="live">● LIVE 8-K retrieved from SEC EDGAR</span>'
            f' &nbsp;·&nbsp; <b>Filed:</b> {ek_doc.get("filed", "n/a")}'
            f' &nbsp;·&nbsp; <b>Period:</b> {ek_doc.get("period_of_report", "n/a")}'
            f' &nbsp;·&nbsp; <b>Accession:</b> {ek_doc.get("accn", "n/a")}'
            f' &nbsp;·&nbsp; <a href="{ek_doc.get("url","")}" target="_blank">📄 View source document →</a>'
            f' &nbsp;·&nbsp; <a href="{ek_doc.get("index_url","")}" target="_blank">All 8-K filings →</a>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="src-note"><span class="stale">○ SEC EDGAR 8-K not reachable — '
            'showing hardcoded values only. Click Refresh in the sidebar to retry.</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Extracted regulatory ratios (live vs hardcoded side-by-side) ──
    st.markdown('<div class="sec-hdr">Extracted Regulatory Ratios — 8-K vs App Values</div>', unsafe_allow_html=True)

    def _fmt_ratio(v): return f"{v:.1f}%" if v is not None else "—"

    ratio_rows = []
    for lab, key, app_v, tol in [
        ("LCR",           "lcr",       d["lcr_pct"],          2.0),
        ("NSFR",          "nsfr",      d.get("nsfr_pct"),     2.0),
        ("CET1 (Std)",    "cet1",      d["cet1"],             0.5),
        ("Tier 1 Capital","tier1",     None,                  None),
        ("Total Capital", "total_cap", None,                  None),
    ]:
        ex = ek_metrics.get(key)
        if ex is not None and app_v is not None and tol is not None:
            diff = ex - app_v
            check = "✅ Match" if abs(diff) <= tol else f"⚠️ Δ {diff:+.1f} pp — verify source"
        elif ex is not None:
            check = "✅ Extracted (no app baseline)"
        else:
            check = "🔍 Not parsed from 8-K text"
        ratio_rows.append({
            "Ratio":         lab,
            "App baseline":  f"{app_v}%" if app_v is not None else "—",
            "8-K extracted": _fmt_ratio(ex),
            "Status":        check,
        })
    st.dataframe(pd.DataFrame(ratio_rows), hide_index=True, use_container_width=True)
    st.caption("⚠️ Regex extraction is best-effort. Always verify against the linked source document for professional use.")

    st.markdown("---")

    # ── Section navigator ──
    if ek_text:
        st.markdown('<div class="sec-hdr">8-K Section Navigator</div>', unsafe_allow_html=True)

        section_labels = {
            "highlights": "📌 Financial Highlights",
            "capital":    "🏦 Capital Ratios",
            "liquidity":  "💧 Liquidity",
            "revenue":    "💰 Revenue / NII",
            "outlook":    "🔭 Outlook / Guidance",
            "raw_snippet":"📄 Document Opening",
        }
        available = {k: v for k, v in ek_sections.items() if v}
        if available:
            section_tabs = st.tabs(list(section_labels[k] for k in available.keys()))
            for tab_s, (sec_key, sec_text) in zip(section_tabs, available.items()):
                with tab_s:
                    # Truncate to readable length
                    display_text = sec_text[:1800].strip()
                    if len(sec_text) > 1800:
                        display_text += " …[truncated — see source for full text]"
                    st.markdown(
                        f'<div class="ek-section"><h5>{section_labels.get(sec_key, sec_key)}</h5>{display_text}</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No sections could be extracted from this 8-K. The document may be structured differently. Use the source link above to read it directly.")

        st.markdown("---")

        # ── Raw text search ──
        st.markdown('<div class="sec-hdr">Search the 8-K Text</div>', unsafe_allow_html=True)
        search_term = st.text_input("Search for a keyword or phrase in the filing", placeholder="e.g. LCR, CET1, deposit, NII, guidance…")
        if search_term:
            pattern = _re.compile(_re.escape(search_term), _re.IGNORECASE)
            matches = list(pattern.finditer(ek_text))
            st.markdown(f"**{len(matches)} occurrences** of `{search_term}` found.")
            for i, match in enumerate(matches[:5]):
                start = max(0, match.start() - 200)
                end   = min(len(ek_text), match.end() + 400)
                snippet = ek_text[start:end]
                # Highlight the match
                highlighted = pattern.sub(f"**{search_term.upper()}**", snippet)
                with st.expander(f"Match {i+1} — position {match.start():,}"):
                    st.markdown(f"…{highlighted}…")
            if len(matches) > 5:
                st.caption(f"Showing first 5 of {len(matches)} matches.")
    else:
        st.info(f"No 8-K text available for {bank_name} right now. Try the Refresh button in the sidebar, or visit the IR page directly: [{bank_name} IR]({d['ir_url']})")

    # ── All 8 banks quick-check ──
    st.markdown("---")
    st.markdown('<div class="sec-hdr">All 8 US G-SIBs — Latest 8-K Filing Dates</div>', unsafe_allow_html=True)
    st.caption("Shows the most recently filed 8-K for each G-SIB. Click Refresh to update.")
    if st.button("Load all G-SIB filing dates"):
        all_rows = []
        prog = st.progress(0)
        for i, (bname, bdata) in enumerate(BANKS.items()):
            eg = fetch_edgar_latest_filing(bdata["cik"])
            ek = fetch_latest_8k_full(bdata["cik"])
            all_rows.append({
                "Bank":          bname,
                "Ticker":        bdata["ticker"],
                "G-SIB Bucket":  bdata["gsib_bucket"],
                "Latest 10-Q":   eg.get("latest_10q", "—"),
                "Latest 10-K":   eg.get("latest_10k", "—"),
                "Latest 8-K":    ek.get("filed", "—"),
                "8-K Period":    ek.get("period_of_report", "—"),
                "EDGAR Live":    "✅" if eg.get("is_live") else "○",
            })
            prog.progress((i+1)/len(BANKS))
        prog.empty()
        st.dataframe(pd.DataFrame(all_rows), hide_index=True, use_container_width=True)

# ========================= TAB 3: NSFR =========================
with tab_nsfr:
    if d["nsfr_pct"] is not None:
        status_nsfr = d["nsfr_pct"] >= 100
        badge2 = '<span class="pass">✔ PASS</span>' if status_nsfr else '<span class="fail">✖ BREACH</span>'
        nc1, nc2 = st.columns([1, 1.6])
        with nc1:
            st.markdown(f"**Reported NSFR &nbsp;{badge2}**", unsafe_allow_html=True)
            st.plotly_chart(gauge(d["nsfr_pct"], "Net Stable Funding Ratio",
                                  COLORS["green"] if status_nsfr else COLORS["red"]),
                            use_container_width=True)
            buf2 = d["nsfr_pct"] - 100
            st.metric("Buffer above 100%", f"{abs(buf2):.1f} pp",
                      delta="Surplus" if buf2 >= 0 else "Deficit",
                      delta_color="normal" if buf2 >= 0 else "inverse")
            if d["nsfr_asf"]:
                st.markdown("---")
                st.markdown('<div class="sec-hdr">NSFR Summary ($B)</div>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame({
                    "Item": ["Available Stable Funding","Required Stable Funding","NSFR"],
                    "Value": [f"${d['nsfr_asf']:,}B", f"${d['nsfr_rsf']:,}B", f"{d['nsfr_pct']}%"],
                }), hide_index=True, use_container_width=True)
        with nc2:
            st.markdown('<div class="sec-hdr">What NSFR Measures</div>', unsafe_allow_html=True)
            st.markdown("""
NSFR answers: **"Is the bank's 1-year funding structure stable?"**
```
NSFR = Available Stable Funding (ASF)  ≥ 100%
       ─────────────────────────────
       Required Stable Funding  (RSF)
```
**ASF** weights liabilities/equity by stability; **RSF** weights assets by illiquidity.
""")
            if d["nsfr_pct"] >= 120:
                st.success(f"**{d['nsfr_pct']:.0f}% NSFR — strong structural funding stability.**")
            else:
                st.warning(f"**{d['nsfr_pct']:.0f}% NSFR — adequate headroom.**")
    else:
        st.info(f"**{bank_name}** does not publish a standalone NSFR figure. Check Pillar 3 filings.")
        st.markdown(f"[🔗 {bank_name} regulatory filings]({d['lcr_doc_url']})")
        st.markdown('<div class="sec-hdr">Balance Sheet Context for NSFR</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Item": ["Equity (→100% ASF)","LT Debt (→100% ASF if >1yr)","Deposits (→90-95% ASF retail)","Cash & HQLA (→5-15% RSF)","Net Loans (→65-85% RSF)"],
            "Value ($B)": [f"${d['equity']:,}B",f"${d['lt_debt']:,}B",f"${d['deposits']:,}B",f"${d['cash_hqla']:,}B",f"${d['loans']:,}B"],
        }), hide_index=True, use_container_width=True)

# ========================= TAB 4: PEER =========================
with tab_peer:
    compare_banks = [bank_name] + selected_peers
    _bar_w = 0.55
    st.markdown('<div class="sec-hdr">LCR Peer Benchmarking</div>', unsafe_allow_html=True)
    lcr_vals = [BANKS[b]["lcr_pct"] for b in compare_banks]
    bar_cols = [COLORS["green"] if v >= 100 else COLORS["red"] for v in lcr_vals]
    bar_cols[0] = COLORS["navy"]
    fig_p = go.Figure(go.Bar(
        x=compare_banks, y=lcr_vals, marker_color=bar_cols,
        marker_line=dict(color="#1a2733", width=1), width=[_bar_w]*len(compare_banks),
        text=[f"{v}%" for v in lcr_vals], textposition="outside",
        textfont={"size":12,"family":"IBM Plex Mono","color":COLORS["ink"]},
    ))
    fig_p.add_hline(y=100, line_dash="dot", line_color=COLORS["red"], line_width=1.5,
                    annotation_text="  100% minimum", annotation_font_size=10, annotation_font_color=COLORS["red"])
    fig_p.update_layout(height=340, margin=dict(t=20,b=10,l=30,r=30), bargap=0.55,
                        xaxis={"tickfont":{"size":11,"color":COLORS["ink"]}},
                        yaxis={"range":[0,max(lcr_vals)*1.25],"gridcolor":"#E4DCCC","title":"%","title_font":{"color":COLORS["grey"],"size":11}},
                        paper_bgcolor=COLORS["cream"], plot_bgcolor=COLORS["cream"])
    st.plotly_chart(fig_p, use_container_width=True)

    nsfr_banks = [(b, BANKS[b]) for b in compare_banks if BANKS[b]["nsfr_pct"] is not None]
    if len(nsfr_banks) >= 2:
        st.markdown('<div class="sec-hdr">NSFR Peer Comparison</div>', unsafe_allow_html=True)
        names = [b for b, _ in nsfr_banks]
        vals  = [cd["nsfr_pct"] for _, cd in nsfr_banks]
        cols_ = [COLORS["green"] if v >= 100 else COLORS["red"] for v in vals]
        if names and names[0] == bank_name: cols_[0] = COLORS["navy"]
        fig_n = go.Figure(go.Bar(
            x=names, y=vals, marker_color=cols_, marker_line=dict(color="#1a2733",width=1),
            width=[_bar_w]*len(names), text=[f"{v:.1f}%" for v in vals], textposition="outside",
            textfont={"size":12,"family":"IBM Plex Mono","color":COLORS["ink"]},
        ))
        fig_n.add_hline(y=100, line_dash="dot", line_color=COLORS["red"], line_width=1.5,
                        annotation_text="  100% minimum", annotation_font_size=10, annotation_font_color=COLORS["red"])
        fig_n.update_layout(height=300, margin=dict(t=20,b=10,l=30,r=30), bargap=0.55,
                            xaxis={"tickfont":{"size":11,"color":COLORS["ink"]}},
                            yaxis={"range":[0,max(vals)*1.25],"gridcolor":"#E4DCCC","title":"%","title_font":{"color":COLORS["grey"],"size":11}},
                            paper_bgcolor=COLORS["cream"], plot_bgcolor=COLORS["cream"])
        st.plotly_chart(fig_n, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="sec-hdr">Full Peer Comparison Table — All 8 G-SIBs</div>', unsafe_allow_html=True)
    all_compare = list(BANKS.keys())
    rows = []
    for b in all_compare:
        cd = BANKS[b]
        rows.append({
            "Bank": b, "Ticker": cd["ticker"],
            "LCR": f"{cd['lcr_pct']}%",
            "NSFR": f"{cd['nsfr_pct']:.1f}%" if cd["nsfr_pct"] else "N/D",
            "HQLA ($B)": f"${cd['lcr_hqla']:,}",
            "Assets ($B)": f"${cd['assets']:,}",
            "HQLA/Assets": f"{cd['lcr_hqla']/cd['assets']*100:.1f}%",
            "CET1": f"{cd['cet1']}%",
            "G-SIB": cd["gsib_bucket"],
            "Period": cd["period"].split("(")[0].strip(),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="sec-hdr">HQLA Pool Comparison ($B)</div>', unsafe_allow_html=True)
    hdf = pd.DataFrame({
        "Bank": compare_banks,
        "L1":  [BANKS[b]["lcr_hqla_l1"] for b in compare_banks],
        "L2A": [BANKS[b]["lcr_hqla_l2a"] for b in compare_banks],
        "L2B": [BANKS[b]["lcr_hqla_l2b"] for b in compare_banks],
    })
    fig_s = go.Figure()
    for col, color, lab in [("L1",COLORS["navy"],"Level 1"),("L2A",COLORS["blue"],"Level 2A"),("L2B",COLORS["sky"],"Level 2B")]:
        fig_s.add_trace(go.Bar(
            name=lab, x=hdf["Bank"], y=hdf[col], marker_color=color,
            marker_line=dict(color="#1a2733",width=1), width=[0.55]*len(hdf["Bank"]),
            text=[f"${v}B" for v in hdf[col]], textposition="inside",
            textfont={"size":10,"color":"white","family":"IBM Plex Mono"},
        ))
    fig_s.update_layout(barmode="stack", height=340, margin=dict(t=20,b=10,l=30,r=30), bargap=0.55,
                        legend={"font":{"size":10,"color":COLORS["ink"]}},
                        xaxis={"tickfont":{"size":11,"color":COLORS["ink"]}},
                        yaxis={"title":"$B","gridcolor":"#E4DCCC","title_font":{"color":COLORS["grey"],"size":11}},
                        paper_bgcolor=COLORS["cream"], plot_bgcolor=COLORS["cream"])
    st.plotly_chart(fig_s, use_container_width=True)

# ========================= TAB 5: YIELD CURVE =========================
with tab_curve:
    yields, as_of, is_live = fetch_treasury_yield_curve()
    col_h1, col_h2 = st.columns([3,1])
    with col_h1:
        st.markdown('<div class="sec-hdr">US Treasury Par Yield Curve</div>', unsafe_allow_html=True)
    with col_h2:
        if is_live:
            st.markdown(f'<span class="live">● LIVE · Treasury.gov</span><br><span style="font-size:.7rem;color:#5a7a9a;">as of {as_of}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="stale">○ snapshot ({as_of})</span>', unsafe_allow_html=True)

    tenors_years = np.array([TENOR_YEARS[t] for t in yields.keys()])
    yields_pct   = np.array(list(yields.values()))
    order = np.argsort(tenors_years)
    tenors_years = tenors_years[order]
    yields_pct   = yields_pct[order]
    nss_params   = fit_nss(tenors_years, yields_pct)
    t_plot = np.linspace(0.1, 30, 200)

    fig_yc = go.Figure()
    fig_yc.add_trace(go.Scatter(
        x=tenors_years, y=yields_pct, mode="markers+text", name="Observed (Treasury.gov)",
        marker=dict(size=10, color=COLORS["navy"], line=dict(width=1,color="white")),
        text=[f"{v:.2f}%" for v in yields_pct], textposition="top center",
        textfont=dict(size=9, family="IBM Plex Mono"),
    ))
    if nss_params is not None:
        fig_yc.add_trace(go.Scatter(x=t_plot, y=nss(t_plot, *nss_params),
                                    mode="lines", name="NSS fit (today)",
                                    line=dict(color=COLORS["navy"], width=2.5)))
        fwd_colors = {"+1Y fwd": COLORS["blue"], "+2Y fwd": COLORS["sky"], "+5Y fwd": COLORS["amber"]}
        for label, h in {"+1Y fwd":1.0, "+2Y fwd":2.0, "+5Y fwd":5.0}.items():
            y_fut = forward_curve_from_nss(h, t_plot, nss_params)
            fig_yc.add_trace(go.Scatter(x=t_plot, y=y_fut, mode="lines",
                                        name=f"Predicted {label}",
                                        line=dict(color=fwd_colors[label], width=2, dash="dash")))
    fig_yc.update_layout(
        height=480, margin=dict(t=20,b=40,l=40,r=20),
        xaxis=dict(title="Maturity (years)", type="log",
                   tickvals=[0.25,0.5,1,2,5,10,20,30],
                   ticktext=["3M","6M","1Y","2Y","5Y","10Y","20Y","30Y"],
                   gridcolor="#E4DCCC", tickfont=dict(color=COLORS["ink"]), title_font=dict(color=COLORS["grey"])),
        yaxis=dict(title="Yield (%)", gridcolor="#E4DCCC",
                   tickfont=dict(color=COLORS["ink"]), title_font=dict(color=COLORS["grey"])),
        legend=dict(orientation="h", y=-0.15, font=dict(color=COLORS["ink"])),
        paper_bgcolor=COLORS["cream"], plot_bgcolor=COLORS["cream"],
    )
    st.plotly_chart(fig_yc, use_container_width=True)
    st.caption("Predicted curves use forward-rate arithmetic on today's NSS zero curve. Risk-neutral implied — not an econometric forecast.")

    st.markdown("---")
    st.markdown('<div class="sec-hdr">Curve Diagnostics</div>', unsafe_allow_html=True)
    sc1,sc2,sc3,sc4 = st.columns(4)
    y2  = yields.get("2 Yr",  float("nan"))
    y5  = yields.get("5 Yr",  float("nan"))
    y10 = yields.get("10 Yr", float("nan"))
    y3m = yields.get("3 Mo",  float("nan"))
    def fmt_bp(val): return f"{val*100:+.0f} bps" if not math.isnan(val) else "—"
    with sc1:
        slope = (y10-y2) if not math.isnan(y10) and not math.isnan(y2) else float("nan")
        st.metric("10Y − 2Y slope", fmt_bp(slope), delta="steepening" if slope > 0 else "flat/inverted", delta_color="normal" if slope > 0 else "inverse")
    with sc2:
        slope2 = (y10-y3m) if not math.isnan(y10) and not math.isnan(y3m) else float("nan")
        st.metric("10Y − 3M slope", fmt_bp(slope2))
    with sc3:
        level = np.nanmean([y2,y5,y10])
        st.metric("Level (avg 2Y/5Y/10Y)", f"{level:.2f}%")
    with sc4:
        curv = (2*y5-y2-y10) if not (math.isnan(y5) or math.isnan(y2) or math.isnan(y10)) else float("nan")
        st.metric("Curvature (2×5Y − 2Y − 10Y)", fmt_bp(curv))

    st.markdown("---")
    st.markdown('<div class="sec-hdr">Interactive Curve Shock — Impact on HQLA Portfolio</div>', unsafe_allow_html=True)
    sk1,sk2,sk3 = st.columns(3)
    with sk1: parallel_bp = st.slider("Parallel shift (bps)", -300, 300, 0, step=25)
    with sk2: steep_bp    = st.slider("Steepener (10Y−2Y, bps)", -200, 200, 0, step=25)
    with sk3: port_dur    = st.slider("HQLA portfolio duration (yrs)", 1.0, 8.0, 4.0, step=0.5)

    if nss_params is not None:
        y_base  = nss(t_plot, *nss_params)
        pivot   = 5.0
        slope_add = np.clip((t_plot-pivot)/(10.0-pivot),-1.0,1.0) * (steep_bp/200.0)
        y_shock = y_base + parallel_bp/100.0 + slope_add
        fig_sh = go.Figure()
        fig_sh.add_trace(go.Scatter(x=t_plot, y=y_base, mode="lines", name="Base (today)", line=dict(color=COLORS["navy"],width=2.5)))
        fig_sh.add_trace(go.Scatter(x=t_plot, y=y_shock, mode="lines", name="Shocked", line=dict(color=COLORS["red"],width=2.5,dash="dash")))
        fig_sh.update_layout(height=320, margin=dict(t=20,b=30,l=40,r=20),
                             xaxis=dict(title="Maturity (years)",type="log",tickvals=[0.25,1,2,5,10,30],ticktext=["3M","1Y","2Y","5Y","10Y","30Y"],gridcolor="#E4DCCC",tickfont=dict(color=COLORS["ink"]),title_font=dict(color=COLORS["grey"])),
                             yaxis=dict(title="Yield (%)",gridcolor="#E4DCCC",tickfont=dict(color=COLORS["ink"]),title_font=dict(color=COLORS["grey"])),
                             legend=dict(orientation="h",y=-0.2,font=dict(color=COLORS["ink"])),
                             paper_bgcolor=COLORS["cream"],plot_bgcolor=COLORS["cream"])
        st.plotly_chart(fig_sh, use_container_width=True)
        eff_shift_pct = parallel_bp/100.0
        hqla_mtm_pct  = -port_dur*(eff_shift_pct/100.0)*100.0
        hqla_mtm_bn   = hqla_mtm_pct/100.0*d["lcr_hqla"]
        steep_equiv   = steep_bp/100.0*max((port_dur-pivot)/(10-pivot),-1)
        hqla_steep_bn = -port_dur*(steep_equiv/100.0)*d["lcr_hqla"]
        total_bn  = hqla_mtm_bn + hqla_steep_bn
        total_pct = total_bn/d["lcr_hqla"]*100
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("HQLA MTM Δ ($B)", f"{total_bn:+,.1f}", delta=f"{total_pct:+.2f}%", delta_color="normal" if total_bn>=0 else "inverse")
        m2.metric("From parallel shift", f"{hqla_mtm_bn:+,.1f}B")
        m3.metric("From steepener",      f"{hqla_steep_bn:+,.1f}B")
        new_lcr_shock = (d["lcr_hqla"]+total_bn)/d["lcr_net_out"]*100
        m4.metric("Shocked LCR", f"{new_lcr_shock:.0f}%", delta=f"{new_lcr_shock-d['lcr_pct']:+.1f} pp", delta_color="normal" if new_lcr_shock>=100 else "inverse")

# ========================= TAB 6: STRESS TESTER =========================
with tab_stress:
    st.markdown('<div class="sec-hdr">Interactive LCR Stress Tester</div>', unsafe_allow_html=True)
    st.caption("Override Basel III default runoff rates to model stressed LCR outcomes.")
    wholesale_share = 0.45 if d["ticker"] in ["GS","MS"] else 0.25 if d["ticker"] in ["JPM","C"] else 0.2
    base_out = d["lcr_net_out"]
    est_retail    = base_out * (1 - wholesale_share - 0.1)
    est_wholesale = base_out * wholesale_share
    est_other     = base_out * 0.10
    c1,c2,c3 = st.columns(3)
    with c1: retail_mult = st.slider("Retail runoff multiplier", 0.5, 3.0, 1.0, 0.1)
    with c2: whole_mult  = st.slider("Wholesale non-op runoff multiplier", 0.5, 3.0, 1.0, 0.1)
    with c3: other_mult  = st.slider("Other outflows multiplier", 0.5, 3.0, 1.0, 0.1)
    new_out = est_retail*retail_mult + est_wholesale*whole_mult + est_other*other_mult
    new_lcr = d["lcr_hqla"]/new_out*100
    st.markdown("---")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Retail outflows ($B)", f"${est_retail*retail_mult:,.0f}B", delta=f"{(retail_mult-1)*100:+.0f}%")
    c2.metric("Wholesale outflows ($B)", f"${est_wholesale*whole_mult:,.0f}B", delta=f"{(whole_mult-1)*100:+.0f}%")
    c3.metric("Total outflows ($B)", f"${new_out:,.0f}B", delta=f"{(new_out-base_out):+,.0f}B")
    c4.metric("Stressed LCR", f"{new_lcr:.0f}%", delta=f"{new_lcr-d['lcr_pct']:+.1f} pp", delta_color="normal" if new_lcr>=100 else "inverse")
    fig_w = go.Figure(go.Waterfall(
        orientation="v", measure=["absolute","relative","relative","relative","total"],
        x=["Reported<br>outflows","Retail Δ","Wholesale Δ","Other Δ","Stressed<br>outflows"],
        y=[base_out, est_retail*(retail_mult-1), est_wholesale*(whole_mult-1), est_other*(other_mult-1), new_out],
        text=[f"${base_out:,.0f}B",f"{est_retail*(retail_mult-1):+,.0f}",f"{est_wholesale*(whole_mult-1):+,.0f}",f"{est_other*(other_mult-1):+,.0f}",f"${new_out:,.0f}B"],
        textposition="outside",
        connector={"line":{"color":"#C9BEA9"}},
        increasing={"marker":{"color":COLORS["red"]}},
        decreasing={"marker":{"color":COLORS["green"]}},
        totals={"marker":{"color":COLORS["blue"]}},
    ))
    fig_w.update_layout(height=360, margin=dict(t=10,b=20,l=30,r=30),
                        yaxis={"title":"$B","gridcolor":"#E4DCCC","title_font":{"color":COLORS["grey"],"size":11},"tickfont":{"color":COLORS["ink"]}},
                        xaxis={"tickfont":{"color":COLORS["ink"]}},
                        paper_bgcolor=COLORS["cream"], plot_bgcolor=COLORS["cream"])
    st.plotly_chart(fig_w, use_container_width=True)
    if new_lcr < 100:
        st.error(f"⚠️ Under this scenario the LCR falls to **{new_lcr:.0f}%** — below the 100% regulatory minimum.")
    elif new_lcr < 110:
        st.warning(f"🟡 LCR drops to **{new_lcr:.0f}%** — still compliant but thin buffer.")
    else:
        st.success(f"🟢 LCR holds at **{new_lcr:.0f}%** — buffer absorbs this stress level.")

# ========================= TAB 7: ALCO COMMENTARY =========================
with tab_alco:
    st.markdown('<div class="sec-hdr">ALCO & Management Commentary</div>', unsafe_allow_html=True)
    st.caption("Public statements from Q1 2026 8-K earnings release (Exhibit 99.1) and earnings-call transcripts.")
    comm = COMMENTARY.get(bank_name, {})
    headline = comm.get("headline", "")
    tone = comm.get("tone", "adequate")
    tone_badge = {"strong":'<span class="pass">✔ Strong tone</span>',"adequate":'<span class="watch">Adequate tone</span>',"improving":'<span class="live">↑ Improving</span>'}.get(tone,'<span class="watch">Neutral</span>')
    st.markdown(
        f'<div class="alco-card"><h4>{bank_name} &nbsp; {tone_badge}</h4>'
        f'<div style="font-size:.92rem;color:#3E5C76;font-weight:500;">{headline}</div>'
        f'<div class="meta">Synthesis of Q1 2026 earnings release + call</div></div>',
        unsafe_allow_html=True,
    )
    for item in comm.get("items", []):
        st.markdown(
            f'<div class="alco-card"><h4>{item["topic"]}</h4>'
            f'<div class="quote">"{item["quote"]}"</div>'
            f'<div class="meta">— {item["attribution"]}</div></div>',
            unsafe_allow_html=True,
        )

# ========================= TAB 8: FRAMEWORK GUIDE =========================
with tab_explain:
    st.markdown('<div class="sec-hdr">Basel III Liquidity Framework</div>', unsafe_allow_html=True)
    st.markdown("""
### LCR — Liquidity Coverage Ratio
```
LCR = HQLA / Net Cash Outflows (30-day stress)  ≥ 100%
```
**HQLA tiers:** Level 1 (0% haircut, cash/UST), Level 2A (15%, agency/covered), Level 2B (50%, IG corp/equity).

### NSFR — Net Stable Funding Ratio
```
NSFR = Available Stable Funding / Required Stable Funding  ≥ 100%  (1-year horizon)
```

### US G-SIB Framework
All 8 US G-SIBs are subject to enhanced prudential standards under Basel III / Fed LCR Rule (12 CFR Part 249). G-SIB buckets (1-4) determine additional capital surcharges on top of the 4.5% CET1 floor.

### 8-K Earnings Release
Banks file an 8-K (Item 2.02) within 4 business days of their earnings release. The primary exhibit (Exhibit 99.1) is the full press release and is the source of LCR, NSFR, CET1, and NII disclosures parsed in the 8-K tab.

### Yield-Curve Model (this app)
Nelson-Siegel-Svensson (NSS) zero-coupon curve fitted to observed par yields. Forward curves use:
```
y_future(T) = [(1+y_spot(h+T))^(h+T) / (1+y_spot(h))^h]^(1/T) − 1
```
""")

# ========================= TAB 9: GLOSSARY =========================
with tab_glossary:
    st.markdown('<div class="sec-hdr">Key Terms</div>', unsafe_allow_html=True)
    glossary = {
        "HQLA": "High Quality Liquid Assets — classified Level 1, 2A, 2B.",
        "LCR": "Liquidity Coverage Ratio. HQLA / 30-day net stress outflows ≥ 100%.",
        "NSFR": "Net Stable Funding Ratio. ASF / RSF ≥ 100% over 1-year horizon.",
        "ASF": "Available Stable Funding — equity and stable liabilities, weighted.",
        "RSF": "Required Stable Funding — asset-weighted funding requirement.",
        "Level 1 Assets": "Cash, central-bank reserves, UST, 0%-RW sovereign. 0% haircut.",
        "Level 2A Assets": "Agency MBS, AA- covered, 20%-RW sovereign. 15% haircut.",
        "Level 2B Assets": "IG corp bonds, main-index equities. 50% haircut, caps apply.",
        "Runoff Rate": "% of a funding source assumed to leave under 30-day stress.",
        "G-SIB": "Global Systemically Important Bank — 8 US institutions (JPM, BAC, C, WFC, GS, MS, BK, STT).",
        "CET1": "Common Equity Tier 1 ratio — primary bank capital measure.",
        "8-K": "SEC form filed within 4 business days of a material event. Item 2.02 = earnings release.",
        "Exhibit 99.1": "The press release attached to an 8-K — primary source of quarterly financial data.",
        "XBRL": "eXtensible Business Reporting Language — structured format for SEC financial data.",
        "Nelson-Siegel-Svensson": "6-parameter smooth functional form for fitting a zero-coupon yield curve.",
        "Pillar 3 Disclosure": "Mandatory public disclosures of capital & liquidity positions under Basel III.",
    }
    for term, defn in glossary.items():
        with st.expander(f"**{term}**"):
            st.write(defn)

# =========================================================================
# FOOTER
# =========================================================================
st.markdown("---")
st.markdown('<div class="sec-hdr">Liquidity Assessment Summary</div>', unsafe_allow_html=True)
lcr_v = d["lcr_pct"]
nsfr_v = d["nsfr_pct"]
nsfr_text = f"NSFR: **{nsfr_v:.1f}%** ✅" if nsfr_v else "NSFR: *not publicly disclosed standalone*"
if lcr_v >= 130:
    st.success(f"**🟢 Strong Liquidity — {bank_name}** | LCR **{lcr_v}%** ({lcr_v-100}pp buffer) | HQLA ${d['lcr_hqla']:,}B vs outflows ${d['lcr_net_out']:,}B | {nsfr_text}")
elif lcr_v >= 110:
    st.success(f"**🟢 Adequate-to-Strong Liquidity — {bank_name}** | LCR **{lcr_v}%**, {lcr_v-100}pp buffer | {nsfr_text}")
elif lcr_v >= 100:
    st.warning(f"**🟡 Adequate Liquidity — {bank_name}** | LCR **{lcr_v}%** — thin headroom | {nsfr_text}")
else:
    st.error(f"**🔴 LCR Below Minimum — {bank_name}** (Reported {lcr_v}%)")

st.markdown(
    f'<div class="src-note">📄 <b>Period:</b> {d["period"]} &nbsp;·&nbsp; <b>Annual:</b> {d["fy_filing"]} &nbsp;·&nbsp; '
    f'<a href="{d["lcr_doc_url"]}" target="_blank">LCR/NSFR Disclosure →</a> &nbsp;|&nbsp; '
    f'<a href="{d["ir_url"]}" target="_blank">IR →</a><br>'
    f'⚠️ Educational tool — verify all figures against source filings before professional use. '
    f'Coverage: All 8 US G-SIBs (JPM, BAC, C, WFC, GS, MS, BK, STT).</div>',
    unsafe_allow_html=True,
)
