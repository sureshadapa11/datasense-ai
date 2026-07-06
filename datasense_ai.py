import os, io, json, re
import streamlit as st
import pandas as pd
import numpy as np
import anthropic
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

class NamedBytesIO(io.BytesIO):
    """BytesIO with a .name attribute so load_dataframe can detect file type."""
    def __init__(self, data, name): super().__init__(data); self.name = name

try:
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass

MODEL = "claude-sonnet-4-6"

st.set_page_config(page_title="DataSense AI", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300..900;1,14..32,300..900&display=swap');

*, html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ── MAIN BACKGROUND ── */
.stApp { background: #ffffff; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1400px !important; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e8edf2 !important;
    min-width: 240px !important;
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }
section[data-testid="stSidebar"] * { color: #475569 !important; }
section[data-testid="stSidebar"] hr { border-color: #e2e8f0 !important; margin: 8px 0 !important; }
section[data-testid="stSidebar"] .stSuccess {
    background: rgba(5,150,105,0.07) !important;
    border: 1px solid rgba(5,150,105,0.25) !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] .stSuccess p { color: #059669 !important; font-size: 12px !important; }

/* ── NAV BUTTONS (horizontal bar) ── */
.stButton > button[kind="secondary"] {
    font-size: 11px !important;
    padding: 0.38rem 0.25rem !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    min-width: 0 !important;
}

/* ── MAIN AREA BUTTONS ── */
.stButton > button {
    background: #fff !important;
    color: #334155 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 0.45rem 0.75rem !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
.stButton > button:hover {
    background: #f8fafc !important;
    border-color: #cbd5e1 !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06) !important;
}

/* ── COLUMNS — allow flex shrink without clipping content ── */
[data-testid="stColumns"] > div { min-width: 0 !important; }
[data-testid="stColumn"] { min-width: 0 !important; }

/* ── FILE UPLOADER — full card restyle ── */
div[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
    margin-top: -3px !important;
}
section[data-testid="stFileUploaderDropzone"] {
    background: #f8f7ff !important;
    border: 2px dashed #c4b5fd !important;
    border-top: none !important;
    border-radius: 0 0 20px 20px !important;
    padding: 0.5rem 2rem 2rem !important;
    transition: background 0.2s, border-color 0.2s !important;
    min-height: 80px !important;
}
section[data-testid="stFileUploaderDropzone"]:hover {
    background: #f0eeff !important;
    border-color: #5b4bff !important;
}
section[data-testid="stFileUploaderDropzone"] > div {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0 !important;
    padding: 0.75rem 0 !important;
}
/* Browse files button — filled violet pill */
section[data-testid="stFileUploaderDropzone"] button {
    background: #5b4bff !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 0.6rem 2rem !important;
    box-shadow: 0 4px 14px rgba(91,75,255,0.28) !important;
    transition: all 0.18s ease !important;
    min-width: 160px !important;
}
section[data-testid="stFileUploaderDropzone"] button:hover {
    background: #4c3de0 !important;
    box-shadow: 0 6px 20px rgba(91,75,255,0.38) !important;
    transform: translateY(-1px) !important;
}
/* Hide the collapsed uploader label entirely */
div[data-testid="stFileUploader"] label { display: none !important; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: #fff !important; border-radius: 10px !important;
    padding: 3px !important; border: 1px solid #e2e8f0 !important;
    gap: 2px !important; box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}
.stTabs [data-baseweb="tab"] { color: #64748b !important; border-radius: 8px !important; font-size: 13px !important; font-weight: 500 !important; padding: 6px 14px !important; }
.stTabs [aria-selected="true"] { background: #0f172a !important; color: #fff !important; }

/* ── KPI CARDS ── */
.kpi-card {
    background: #fff;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    border: 1px solid #edf0f7;
    position: relative;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(15,23,42,0.05);
    transition: box-shadow 0.2s;
    height: 100%;
}
.kpi-card:hover { box-shadow: 0 4px 16px rgba(15,23,42,0.08); }
.kpi-bar {
    position: absolute; top: 0; left: 0;
    width: 3px; height: 100%;
    border-radius: 12px 0 0 12px;
}
.kpi-label {
    font-size: 10.5px; font-weight: 600; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 8px;
}
.kpi-value { font-size: 26px; font-weight: 700; color: #0d1f3c; line-height: 1.1; margin-bottom: 4px; }
.kpi-sub { font-size: 12px; color: #94a3b8; }
.kpi-up { color: #10b981 !important; font-weight: 600; }
.kpi-dn { color: #ef4444 !important; font-weight: 600; }

/* ── INSIGHT CARDS ── */
.insight-card {
    background: linear-gradient(135deg, #111827 0%, #0f172a 100%);
    border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 0.75rem;
    border: 1px solid #1e2d45;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.insight-tag {
    display: inline-block; border-radius: 20px; padding: 2px 10px;
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.07em; margin-bottom: 10px;
}
.insight-title { font-size: 15px; font-weight: 700; color: #f1f5f9; margin-bottom: 6px; }
.insight-text { font-size: 13px; line-height: 1.75; color: #94a3b8; }

/* ── AI BOX ── */
.ai-box {
    background: #fff;
    border: 1px solid #e4e0ff;
    border-left: 3px solid #5b4bff;
    border-radius: 0 12px 12px 0;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 1px 4px rgba(91,75,255,0.06);
}
.ai-box-title {
    font-size: 10.5px; font-weight: 700; color: #5b4bff;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 10px;
    display: flex; align-items: center; gap: 6px;
}
.ai-box-text { font-size: 13.5px; color: #1e293b; line-height: 1.8; }

/* ── SECTION HEADERS ── */
.sec-head {
    font-size: 11px; font-weight: 700; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin: 1.75rem 0 1rem;
    display: flex; align-items: center; gap: 8px;
    flex-wrap: wrap; word-break: break-word;
}
.sec-head::before {
    content: ''; display: block; width: 3px; height: 14px;
    background: #5b4bff;
    border-radius: 2px; flex-shrink: 0;
}

/* ── BADGES ── */
.badge { display: inline-block; border-radius: 6px; padding: 2px 8px; font-size: 11px; font-weight: 600; }
.badge-green { background: #dcfce7; color: #166534; }
.badge-amber { background: #fef3c7; color: #92400e; }
.badge-red   { background: #fee2e2; color: #991b1b; }
.badge-blue  { background: #dbeafe; color: #1e40af; }

/* ── STAT ROW ── */
.stat-row { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 0.5rem; align-items: center; }
.stat-pill {
    background: #f1f5f9; border: 1px solid #e2e8f0;
    border-radius: 100px; padding: 3px 10px;
    font-size: 11.5px; color: #64748b;
}
.stat-pill b { color: #334155; font-weight: 600; }

/* ── PAGE HEADER ── */
.page-header {
    background: #fff; border-radius: 12px; padding: 1.1rem 1.5rem;
    border: 1px solid #edf0f7; margin-bottom: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    display: flex; align-items: center; justify-content: space-between;
}

/* ── CHAT BUBBLES ── */
.chat-user {
    background: linear-gradient(135deg, #1e40af, #2563eb);
    color: #fff; border-radius: 16px 16px 4px 16px;
    padding: 0.75rem 1rem; margin: 0.5rem 0;
    font-size: 13.5px; line-height: 1.6; max-width: 85%; margin-left: auto;
    box-shadow: 0 2px 8px rgba(37,99,235,0.25);
}
.chat-ai {
    background: #fff; border: 1px solid #e2e8f0;
    color: #1e293b; border-radius: 4px 16px 16px 16px;
    padding: 0.75rem 1rem; margin: 0.5rem 0;
    font-size: 13.5px; line-height: 1.75; max-width: 92%;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}

/* ── FORM INPUTS ── */
.stTextInput > div > div > input {
    background: #fff !important; border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important; color: #0d1f3c !important;
    font-size: 13px !important; padding: 0.5rem 0.75rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #5b4bff !important;
    box-shadow: 0 0 0 3px rgba(91,75,255,0.1) !important;
}

/* ── DOWNLOAD BUTTON ── */
.stDownloadButton > button {
    background: #fff !important; color: #334155 !important;
    border: 1px solid #e2e8f0 !important; border-radius: 8px !important;
    font-size: 13px !important; font-weight: 500 !important;
    white-space: nowrap !important; overflow: hidden !important;
    text-overflow: ellipsis !important; min-width: 0 !important;
}
.stDownloadButton > button:hover {
    background: #f8fafc !important; border-color: #cbd5e1 !important;
}

/* ── EXPANDERS ── */
.streamlit-expanderHeader {
    background: #f8fafc !important; border-radius: 8px !important;
    font-size: 13px !important; font-weight: 600 !important; color: #334155 !important;
}
details { border: 1px solid #e2e8f0 !important; border-radius: 10px !important; margin-bottom: 8px !important; }

/* ── DATAFRAME ── */
.stDataFrame { border-radius: 10px !important; overflow: hidden !important; }

/* ── HEADINGS ── */
h1, h2, h3, h4 { color: #0d1f3c !important; }

/* ── SECTION LABELS (small-caps Tellius style) ── */
.slabel {
    display: block;
    font-size: 11px; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #5b4bff;
    margin-bottom: 10px;
}

/* ── INFO/SUCCESS/ERROR ── */
.stAlert { border-radius: 10px !important; }

/* ── PREVENT HORIZONTAL TEXT CLIP ── */
p, li,
.insight-text, .insight-title,
.ai-box-text, .kpi-sub {
    overflow-wrap: break-word !important;
    white-space: normal !important;
    max-width: 100% !important;
}
/* Ensure markdown paragraph containers are fully visible */
[data-testid="stMarkdownContainer"] {
    overflow: visible !important;
    max-width: 100% !important;
}
[data-testid="stMarkdownContainer"] p {
    overflow: visible !important;
    white-space: normal !important;
    overflow-wrap: break-word !important;
}

/* ── HIDE SIDEBAR ── */
section[data-testid="stSidebar"] { display:none !important; }
[data-testid="collapsedControl"] { display:none !important; }

/* ── PRIMARY BUTTON (outlined pill CTA style) ── */
.stButton > button[kind="primary"] {
    background: transparent !important;
    color: #5b4bff !important;
    border: 2px solid #5b4bff !important;
    border-radius: 100px !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    padding: 0.55rem 1.75rem !important;
    letter-spacing: 0.01em !important;
    box-shadow: none !important;
    transition: all 0.18s ease !important;
}
.stButton > button[kind="primary"]:hover {
    background: #5b4bff !important;
    color: #ffffff !important;
    box-shadow: 0 4px 16px rgba(91,75,255,0.25) !important;
}
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
COLORS = ['#2563eb','#10b981','#f59e0b','#ef4444','#8b5cf6','#06b6d4','#f97316','#84cc16','#ec4899','#14b8a6']

DATASET_CONFIG = {
    "sales": {
        "icon": "🛒", "label": "Sales Analytics", "color": "#2563eb",
        "keywords": ["revenue","sales","product","region","order","customer","discount","profit","units","sold"],
        "key_metrics": ["revenue","units_sold","profit","discount_pct"],
        "insights_prompt": (
            "You are a sales analyst. For this sales dataset provide SPECIFIC insights:\n"
            "1. Revenue performance: total, growth rate, best/worst period\n"
            "2. Product performance: top 3 products, their revenue share, any declining products\n"
            "3. Regional analysis: top region, underperforming regions, concentration risk\n"
            "4. Salesperson performance: top performer, gap between best and worst\n"
            "5. Discount impact: average discount, correlation with revenue\n"
            "6. Three specific actionable recommendations with expected impact\n"
            "Use EXACT numbers from the data. Make every insight specific to THIS dataset."
        ),
    },
    "hr": {
        "icon": "👥", "label": "HR Analytics", "color": "#7c3aed",
        "keywords": ["employee","salary","department","hire","leave","performance","headcount","staff","payroll","role"],
        "key_metrics": ["salary","bonus","leave_days"],
        "insights_prompt": (
            "You are an HR analyst. For this HR dataset provide SPECIFIC insights:\n"
            "1. Salary analysis: avg salary, range, highest/lowest paying departments\n"
            "2. Performance distribution: % excellent/good/average/poor performers\n"
            "3. Department headcount: largest and smallest departments\n"
            "4. Gender pay gap analysis if gender column exists\n"
            "5. Leave days: average, which department takes most leave\n"
            "6. Tenure analysis: when were most people hired\n"
            "7. Three HR recommendations based on the data\n"
            "Use EXACT numbers. Be specific to THIS dataset."
        ),
    },
    "finance": {
        "icon": "💰", "label": "Finance Analytics", "color": "#059669",
        "keywords": ["expense","budget","cashflow","profit","loss","asset","liability","income","cost","invoice"],
        "key_metrics": ["revenue","expense","profit","budget"],
        "insights_prompt": (
            "You are a finance analyst. For this finance dataset provide SPECIFIC insights:\n"
            "1. Revenue vs expense: total revenue, total expense, net profit\n"
            "2. Budget vs actual: are we over or under budget? By how much?\n"
            "3. Profit margins: calculate and assess\n"
            "4. Category breakdown: which category has highest spend\n"
            "5. Trends: is the business growing or declining?\n"
            "6. Cash flow health assessment\n"
            "7. Three financial recommendations\n"
            "Use EXACT numbers. Be specific to THIS dataset."
        ),
    },
    "marketing": {
        "icon": "📣", "label": "Marketing Analytics", "color": "#db2777",
        "keywords": ["campaign","click","impression","conversion","lead","channel","ctr","cpc","roas","audience"],
        "key_metrics": ["revenue","spend","conversions","roas","ctr"],
        "insights_prompt": (
            "You are a marketing analyst. For this marketing dataset provide SPECIFIC insights:\n"
            "1. Overall ROAS and ROI: are campaigns profitable?\n"
            "2. Best performing channel: highest revenue, best ROAS\n"
            "3. Worst performing channel: lowest ROI, needs attention\n"
            "4. Campaign performance: which campaign drives most conversions\n"
            "5. CTR benchmarks: which channels have best/worst CTR\n"
            "6. Spend efficiency: cost per conversion by channel\n"
            "7. Three marketing optimization recommendations\n"
            "Use EXACT numbers. Be specific to THIS dataset."
        ),
    },
    "inventory": {
        "icon": "📦", "label": "Inventory Analytics", "color": "#d97706",
        "keywords": ["stock","inventory","sku","warehouse","supplier","reorder","quantity","item","shelf"],
        "key_metrics": ["quantity","stock","reorder"],
        "insights_prompt": (
            "You are an inventory analyst. Provide SPECIFIC insights:\n"
            "1. Stock levels: total items, low stock alerts\n"
            "2. Supplier analysis: top suppliers, concentration risk\n"
            "3. Warehouse utilization\n"
            "4. Reorder analysis: items needing reorder\n"
            "5. Three inventory recommendations\n"
            "Use EXACT numbers."
        ),
    },
    "generic": {
        "icon": "📊", "label": "Data Analytics", "color": "#475569",
        "keywords": [],
        "key_metrics": [],
        "insights_prompt": (
            "You are a data analyst. For this dataset provide SPECIFIC insights:\n"
            "1. Data overview: key patterns you notice\n"
            "2. Top findings from numeric columns (use actual column names and values)\n"
            "3. Category distributions: what stands out\n"
            "4. Trends or anomalies in the data\n"
            "5. Three actionable recommendations based on this specific data\n"
            "Use EXACT numbers and column names from the dataset."
        ),
    },
}

AI_PAGE_PROMPTS = {
    "exec": "You are a senior analyst. Write an executive summary with: (1) top 3 headline metrics with exact values, (2) biggest risk or issue, (3) top opportunity, (4) three specific action recommendations. Use EXACT numbers. Be crisp and direct.",
    "trends": "You are a trend analyst. Analyze time patterns: (1) overall growth rate first to last period, (2) best and worst months with exact values, (3) MoM acceleration or deceleration, (4) seasonality patterns if visible, (5) forecast for next period. Use EXACT numbers from the data.",
    "categories": "You are a segmentation analyst. Analyze categories/segments: (1) which segment dominates and by how much, (2) which is underperforming and why, (3) concentration risk — is too much dependent on one segment, (4) quick wins to improve bottom segments. Use EXACT numbers.",
    "anomalies": "You are a data quality analyst. Identify: (1) any suspicious values or outliers with exact values, (2) missing data patterns, (3) data integrity issues, (4) whether the data looks reliable, (5) recommendations to improve data quality. Use EXACT numbers.",
}

# ── DATA LOADING ──────────────────────────────────────────────────────────────
def load_dataframe(f):
    key = f"df_{f.name}"
    if key in st.session_state:
        return st.session_state[key]
    f.seek(0)
    ext = os.path.splitext(f.name)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(f)
    else:
        df = None
        for hr in [0, 1, 2]:
            try:
                f.seek(0)
                tmp = pd.read_excel(f, header=hr)
                tmp = tmp.loc[:, ~tmp.columns.str.contains('^Unnamed', na=False)]
                tmp = tmp.dropna(how='all').dropna(axis=1, how='all')
                if len(tmp.columns) > 1 and len(tmp) > 1:
                    df = tmp; break
            except: continue
        if df is None:
            f.seek(0); df = pd.read_excel(f)
    df.columns = [str(c).strip().replace('\n', ' ') for c in df.columns]
    df = df.dropna(how='all').dropna(axis=1, how='all').reset_index(drop=True)
    st.session_state[key] = df
    return df

@st.cache_data
def analyze_columns(df):
    result = {"numeric": [], "categorical": [], "date": [], "text": []}
    id_names = {'id','index','no','num','number','#','sr','sr.','row','seq','employee_id'}
    for col in df.columns:
        s = df[col].dropna()
        if len(s) == 0: continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            result["date"].append(col); continue
        if pd.api.types.is_numeric_dtype(df[col]):
            is_id = (str(col).lower().strip() in id_names or
                     (df[col].nunique() == len(df) and str(col).lower().strip().endswith('id')))
            if not is_id: result["numeric"].append(col)
            continue
        try:
            conv = pd.to_datetime(s.astype(str), errors='coerce')
            if conv.notna().sum() / len(s) > 0.7:
                result["date"].append(col); continue
        except: pass
        if s.nunique() <= 30 or s.nunique() / len(s) <= 0.3:
            result["categorical"].append(col)
        else:
            result["text"].append(col)
    return result

@st.cache_data
def detect_type(df):
    cols = ' '.join(df.columns.tolist()).lower()
    scores = {k: sum(1 for w in v["keywords"] if w in cols)
              for k, v in DATASET_CONFIG.items() if k != "generic"}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "generic"

def fmt(v):
    try:
        if pd.isna(v): return "—"
        v = float(v)
        if abs(v) >= 1e9:  return f"{v/1e9:.1f}B"
        if abs(v) >= 1e6:  return f"{v/1e6:.1f}M"
        if abs(v) >= 1000: return f"{v:,.0f}"
        if abs(v) >= 1:    return f"{v:,.1f}"
        return f"{v:.3f}"
    except: return "—"

def fmt_plain(v):
    try:
        if pd.isna(v): return "—"
        v = float(v)
        if abs(v) >= 1e6:  return f"{v/1e6:.1f}M"
        if abs(v) >= 1000: return f"{v:,.0f}"
        if abs(v) >= 1:    return f"{v:,.1f}"
        return f"{v:.3f}"
    except: return "—"

# ── DATASET-SPECIFIC SMART INSIGHTS ──────────────────────────────────────────
@st.cache_data
def compute_smart_insights(df, _col_analysis, dtype):
    col_analysis = _col_analysis
    """Generate dataset-specific insight cards based on actual column names and values."""
    insights = []
    nums = col_analysis["numeric"]
    cats = col_analysis["categorical"]
    dates = col_analysis["date"]

    if dtype == "sales":
        # Revenue insight
        rev_col = next((c for c in nums if 'revenue' in c.lower()), nums[0] if nums else None)
        if rev_col:
            total = df[rev_col].sum()
            avg   = df[rev_col].mean()
            insights.append({
                "tag": "💰 Revenue", "color": "#2563eb",
                "title": f"Total {rev_col.replace('_',' ').title()}",
                "text": f"Total: {fmt(total)} across {len(df):,} transactions. Average per transaction: {fmt(avg)}.",
                "number": fmt(total), "number_color": "#2563eb"
            })
        # Top product
        prod_col = next((c for c in cats if 'product' in c.lower()), None)
        if prod_col and rev_col:
            top = df.groupby(prod_col)[rev_col].sum().idxmax()
            top_val = df.groupby(prod_col)[rev_col].sum().max()
            share = top_val / df[rev_col].sum() * 100
            insights.append({
                "tag": "🏆 Top Product", "color": "#10b981",
                "title": f"Best Seller: {top}",
                "text": f"{top} generates {fmt(top_val)} ({share:.1f}% of total revenue). Focus marketing here.",
                "number": f"{share:.0f}%", "number_color": "#10b981"
            })
        # Region gap
        reg_col = next((c for c in cats if 'region' in c.lower()), None)
        if reg_col and rev_col:
            grp = df.groupby(reg_col)[rev_col].sum()
            gap = grp.max() - grp.min()
            insights.append({
                "tag": "🌍 Regional Gap", "color": "#f59e0b",
                "title": f"Top vs Bottom Region: {fmt(gap)} gap",
                "text": f"{grp.idxmax()} leads with {fmt(grp.max())}. {grp.idxmin()} is lowest at {fmt(grp.min())}. Consider reallocating resources.",
                "number": fmt(gap), "number_color": "#f59e0b"
            })
        # Discount impact
        disc_col = next((c for c in nums if 'discount' in c.lower()), None)
        if disc_col:
            avg_disc = df[disc_col].mean()
            insights.append({
                "tag": "🏷️ Discounting", "color": "#8b5cf6",
                "title": f"Average Discount: {avg_disc:.1f}%",
                "text": f"Discounts range from {df[disc_col].min():.0f}% to {df[disc_col].max():.0f}%. Review if high discounts are eroding margins.",
                "number": f"{avg_disc:.1f}%", "number_color": "#8b5cf6"
            })

    elif dtype == "hr":
        # Salary insight
        sal_col = next((c for c in nums if 'salary' in c.lower()), nums[0] if nums else None)
        if sal_col:
            avg_sal = df[sal_col].mean()
            max_sal = df[sal_col].max()
            min_sal = df[sal_col].min()
            insights.append({
                "tag": "💰 Salary", "color": "#7c3aed",
                "title": "Salary Distribution",
                "text": f"Average salary: {fmt_plain(avg_sal)}. Range: {fmt_plain(min_sal)} — {fmt_plain(max_sal)}. Gap of {fmt_plain(max_sal-min_sal)} between highest and lowest paid.",
                "number": fmt_plain(avg_sal), "number_color": "#7c3aed"
            })
        # Department with highest salary
        dept_col = next((c for c in cats if 'department' in c.lower() or 'dept' in c.lower()), None)
        if dept_col and sal_col:
            dept_avg = df.groupby(dept_col)[sal_col].mean()
            insights.append({
                "tag": "🏢 Department Pay", "color": "#2563eb",
                "title": f"Highest Paying: {dept_avg.idxmax()}",
                "text": f"{dept_avg.idxmax()} avg salary: {fmt_plain(dept_avg.max())}. {dept_avg.idxmin()} is lowest at {fmt_plain(dept_avg.min())}.",
                "number": fmt_plain(dept_avg.max()), "number_color": "#2563eb"
            })
        # Performance distribution
        perf_col = next((c for c in cats if 'performance' in c.lower()), None)
        if perf_col:
            perf_dist = df[perf_col].value_counts()
            top_perf  = perf_dist.index[0]
            top_pct   = perf_dist.iloc[0] / len(df) * 100
            insights.append({
                "tag": "⭐ Performance", "color": "#10b981",
                "title": f"Most Common Rating: {top_perf}",
                "text": f"{top_pct:.0f}% of employees rated '{top_perf}'. " +
                        " | ".join([f"{k}: {v}" for k,v in perf_dist.items()]),
                "number": f"{top_pct:.0f}%", "number_color": "#10b981"
            })
        # Leave days
        leave_col = next((c for c in nums if 'leave' in c.lower()), None)
        if leave_col:
            avg_leave = df[leave_col].mean()
            insights.append({
                "tag": "🏖️ Leave", "color": "#f59e0b",
                "title": f"Avg Leave Days: {avg_leave:.1f}",
                "text": f"Employees take an average of {avg_leave:.1f} leave days. Range: {df[leave_col].min():.0f} to {df[leave_col].max():.0f} days.",
                "number": f"{avg_leave:.1f}", "number_color": "#f59e0b"
            })

    elif dtype == "finance":
        rev_col  = next((c for c in nums if 'revenue' in c.lower()), None)
        exp_col  = next((c for c in nums if 'expense' in c.lower() or 'cost' in c.lower()), None)
        prof_col = next((c for c in nums if 'profit' in c.lower()), None)
        bud_col  = next((c for c in nums if 'budget' in c.lower()), None)
        if rev_col and exp_col:
            total_rev = df[rev_col].sum()
            total_exp = df[exp_col].sum()
            margin = (total_rev - total_exp) / total_rev * 100 if total_rev > 0 else 0
            insights.append({
                "tag": "📊 P&L Summary", "color": "#059669",
                "title": f"Net Margin: {margin:.1f}%",
                "text": f"Total Revenue: {fmt(total_rev)} | Total Expenses: {fmt(total_exp)} | Net: {fmt(total_rev-total_exp)}",
                "number": f"{margin:.1f}%", "number_color": "#059669" if margin > 0 else "#ef4444"
            })
        if bud_col and exp_col:
            total_bud = df[bud_col].sum()
            total_exp2 = df[exp_col].sum()
            variance = total_bud - total_exp2
            pct = variance / total_bud * 100 if total_bud > 0 else 0
            insights.append({
                "tag": "📋 Budget vs Actual", "color": "#2563eb",
                "title": f"{'Under' if variance>0 else 'Over'} Budget by {fmt(abs(variance))}",
                "text": f"Budget: {fmt(total_bud)} | Actual Spend: {fmt(total_exp2)} | Variance: {fmt(variance)} ({pct:.1f}%)",
                "number": f"{abs(pct):.1f}%", "number_color": "#10b981" if variance > 0 else "#ef4444"
            })
        cat_col = next((c for c in cats if 'category' in c.lower()), None)
        if cat_col and exp_col:
            top_cat = df.groupby(cat_col)[exp_col].sum().idxmax()
            top_val = df.groupby(cat_col)[exp_col].sum().max()
            share   = top_val / df[exp_col].sum() * 100
            insights.append({
                "tag": "💸 Top Cost", "color": "#ef4444",
                "title": f"Biggest Expense: {top_cat}",
                "text": f"{top_cat} accounts for {fmt(top_val)} ({share:.1f}%) of total expenses. Consider cost reduction here.",
                "number": f"{share:.0f}%", "number_color": "#ef4444"
            })

    elif dtype == "marketing":
        roas_col  = next((c for c in nums if 'roas' in c.lower()), None)
        spend_col = next((c for c in nums if 'spend' in c.lower()), None)
        rev_col   = next((c for c in nums if 'revenue' in c.lower()), None)
        ctr_col   = next((c for c in nums if 'ctr' in c.lower()), None)
        conv_col  = next((c for c in nums if 'conversion' in c.lower()), None)
        chan_col  = next((c for c in cats if 'channel' in c.lower()), None)
        camp_col  = next((c for c in cats if 'campaign' in c.lower()), None)

        if roas_col:
            avg_roas = df[roas_col].mean()
            insights.append({
                "tag": "📈 ROAS", "color": "#db2777",
                "title": f"Average ROAS: {avg_roas:.2f}x",
                "text": f"For every 1 unit spent, you earn {avg_roas:.2f} back. Range: {df[roas_col].min():.2f}x — {df[roas_col].max():.2f}x.",
                "number": f"{avg_roas:.2f}x", "number_color": "#db2777" if avg_roas >= 3 else "#ef4444"
            })
        if chan_col and roas_col:
            best_chan = df.groupby(chan_col)[roas_col].mean().idxmax()
            best_roas = df.groupby(chan_col)[roas_col].mean().max()
            worst_chan = df.groupby(chan_col)[roas_col].mean().idxmin()
            insights.append({
                "tag": "🏆 Best Channel", "color": "#10b981",
                "title": f"{best_chan} leads with {best_roas:.2f}x ROAS",
                "text": f"{best_chan} is your top performer. {worst_chan} is lowest — review or pause spend there.",
                "number": f"{best_roas:.2f}x", "number_color": "#10b981"
            })
        if spend_col and conv_col:
            total_spend = df[spend_col].sum()
            total_conv  = df[conv_col].sum()
            cpa = total_spend / total_conv if total_conv > 0 else 0
            insights.append({
                "tag": "💸 Cost Per Conversion", "color": "#f59e0b",
                "title": f"CPA: {fmt_plain(cpa)}",
                "text": f"Total spend: {fmt(total_spend)} | Total conversions: {total_conv:,} | Avg cost per conversion: {fmt_plain(cpa)}",
                "number": fmt_plain(cpa), "number_color": "#f59e0b"
            })
        if ctr_col:
            raw_ctr = df[ctr_col].mean()
            avg_ctr = raw_ctr if raw_ctr > 1 else raw_ctr * 100
            min_ctr = df[ctr_col].min()
            max_ctr = df[ctr_col].max()
            min_display = min_ctr if min_ctr > 1 else min_ctr * 100
            max_display = max_ctr if max_ctr > 1 else max_ctr * 100
            insights.append({
                "tag": "👆 CTR", "color": "#8b5cf6",
                "title": f"Average CTR: {avg_ctr:.2f}%",
                "text": f"Click-through rate ranges from {min_display:.2f}% to {max_display:.2f}%. Industry avg is 2-5%.",
                "number": f"{avg_ctr:.2f}%", "number_color": "#8b5cf6"
            })

    else:
        # Generic insights for any dataset
        for i, nc in enumerate(nums[:4]):
            s = df[nc].dropna()
            insights.append({
                "tag": f"📊 {nc.replace('_',' ').title()}", "color": COLORS[i],
                "title": f"Total: {fmt_plain(s.sum())}",
                "text": f"Average: {fmt_plain(s.mean())} | Min: {fmt_plain(s.min())} | Max: {fmt_plain(s.max())} | Std Dev: {fmt_plain(s.std())}",
                "number": fmt_plain(s.sum()), "number_color": COLORS[i]
            })

    return insights

@st.cache_data
def compute_time_insights(df, _col_analysis):
    col_analysis = _col_analysis
    if not col_analysis["date"] or not col_analysis["numeric"]: return {}
    date_col = col_analysis["date"][0]
    num_col  = col_analysis["numeric"][0]
    df2 = df.copy()
    df2['_dt'] = pd.to_datetime(df2[date_col].astype(str), errors='coerce')
    df2 = df2.dropna(subset=['_dt'])
    df2['_m'] = df2['_dt'].dt.to_period('M')
    monthly = df2.groupby('_m')[num_col].sum().sort_index()
    if len(monthly) < 2: return {}
    growth = round((float(monthly.iloc[-1]) - float(monthly.iloc[0])) / abs(float(monthly.iloc[0])) * 100, 1) if monthly.iloc[0] != 0 else 0
    mom = monthly.pct_change().dropna()
    return {
        "periods": len(monthly), "growth": growth,
        "best_period": str(monthly.idxmax()),
        "worst_period": str(monthly.idxmin()),
        "best_val": float(monthly.max()),
        "worst_val": float(monthly.min()),
        "num_col": num_col, "date_col": date_col,
    }

@st.cache_data
def compute_advanced_stats(df, _col_analysis):
    col_analysis = _col_analysis
    stats = {}
    for nc in col_analysis["numeric"]:
        s = df[nc].dropna()
        if len(s) == 0: continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        stats[nc] = {
            "sum": s.sum(), "mean": s.mean(), "median": s.median(),
            "min": s.min(), "max": s.max(), "std": s.std(),
            "missing": int(df[nc].isna().sum()),
            "outliers": int(((s < q1-1.5*iqr) | (s > q3+1.5*iqr)).sum()),
            "q1": q1, "q3": q3,
        }
    return stats

@st.cache_data
def compute_correlations(df, _col_analysis):
    col_analysis = _col_analysis
    nums = col_analysis["numeric"]
    if len(nums) < 2: return []
    corr = df[nums].corr()
    pairs = []
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            r = corr.iloc[i,j]
            if not pd.isna(r):
                pairs.append({"col1": nums[i], "col2": nums[j], "r": round(float(r), 3)})
    pairs.sort(key=lambda x: abs(x["r"]), reverse=True)
    return pairs[:8]

# ── PLOTLY CHARTS ─────────────────────────────────────────────────────────────
BASE = dict(paper_bgcolor='white', plot_bgcolor='white',
            font=dict(family='Inter, sans-serif', size=12, color='#475569'),
            margin=dict(l=10, r=10, t=30, b=10))

def mk_layout(**kwargs):
    d = dict(**BASE)
    d.update(kwargs)
    return d

def plot_trend(df, col_analysis):
    dc = col_analysis["date"][0]
    ncs = col_analysis["numeric"][:4]
    df2 = df.copy()
    df2['_dt'] = pd.to_datetime(df2[dc].astype(str), errors='coerce')
    df2 = df2.dropna(subset=['_dt'])
    df2['_p'] = df2['_dt'].dt.to_period('M').dt.to_timestamp()
    fig = go.Figure()
    for i, nc in enumerate(ncs):
        agg = df2.groupby('_p')[nc].sum().reset_index()
        fig.add_trace(go.Scatter(
            x=agg['_p'], y=agg[nc], name=nc.replace('_',' ').title(),
            line=dict(color=COLORS[i], width=2.5),
            fill='tozeroy' if i == 0 else 'none',
            fillcolor='rgba(37,99,235,0.07)',
            mode='lines+markers', marker=dict(size=5)
        ))
    fig.update_layout(**mk_layout(height=360, showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.01, x=0),
        xaxis=dict(showgrid=False, linecolor='#e2e8f0'),
        yaxis=dict(gridcolor='#f1f5f9', linecolor='#e2e8f0')))
    return fig

def plot_bar(df, cat_col, num_col, top_n=12, horizontal=False):
    grp = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(top_n).reset_index()
    if horizontal:
        grp = grp.sort_values(num_col, ascending=True)
        fig = px.bar(grp, x=num_col, y=cat_col, orientation='h', color=cat_col,
                     color_discrete_sequence=COLORS)
    else:
        fig = px.bar(grp, x=cat_col, y=num_col, color=cat_col,
                     color_discrete_sequence=COLORS)
    fig.update_layout(**mk_layout(height=340, showlegend=False,
        xaxis=dict(showgrid=False, linecolor='#e2e8f0'),
        yaxis=dict(gridcolor='#f1f5f9', linecolor='#e2e8f0')))
    fig.update_traces(marker_line_width=0)
    return fig

def plot_pie(df, cat_col, num_col):
    grp = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(8).reset_index()
    fig = px.pie(grp, names=cat_col, values=num_col,
                 color_discrete_sequence=COLORS, hole=0.45)
    fig.update_layout(**mk_layout(height=320, showlegend=True,
        legend=dict(orientation='v', x=1.0, font=dict(size=11))))
    fig.update_traces(textposition='inside', textinfo='percent+label',
                      textfont_size=11)
    return fig

def plot_histogram(df, num_col):
    fig = px.histogram(df, x=num_col, nbins=20, color_discrete_sequence=['#2563eb'])
    fig.update_layout(**mk_layout(height=240, showlegend=False,
        xaxis=dict(showgrid=False, linecolor='#e2e8f0'),
        yaxis=dict(gridcolor='#f1f5f9', linecolor='#e2e8f0')))
    fig.update_traces(marker_line_width=0.5, marker_line_color='white')
    return fig

def plot_scatter(df, col1, col2):
    sample = df[[col1, col2]].dropna().head(500)
    fig = px.scatter(sample, x=col1, y=col2, color_discrete_sequence=['#2563eb'])
    fig.update_layout(**mk_layout(height=290, showlegend=False,
        xaxis=dict(showgrid=True, gridcolor='#f1f5f9', linecolor='#e2e8f0'),
        yaxis=dict(showgrid=True, gridcolor='#f1f5f9', linecolor='#e2e8f0')))
    fig.update_traces(marker=dict(size=7, opacity=0.55))
    return fig

def plot_heatmap(df, col_analysis):
    if not col_analysis["date"] or not col_analysis["categorical"] or not col_analysis["numeric"]:
        return None
    dc = col_analysis["date"][0]
    cat = col_analysis["categorical"][0]
    num = col_analysis["numeric"][0]
    df2 = df.copy()
    df2['_dt'] = pd.to_datetime(df2[dc].astype(str), errors='coerce')
    df2['_month'] = df2['_dt'].dt.strftime('%b')
    df2['_mnum']  = df2['_dt'].dt.month
    months_order = df2.drop_duplicates('_month').sort_values('_mnum')['_month'].tolist()
    pivot = pd.pivot_table(df2, values=num, index=cat, columns='_month',
                           aggfunc='sum', fill_value=0)
    pivot = pivot.reindex(columns=[m for m in months_order if m in pivot.columns])
    n_cats = len(pivot)
    fig = px.imshow(pivot, color_continuous_scale='Blues', aspect='auto', text_auto='.0f')
    fig.update_layout(**mk_layout(height=max(260, n_cats*52+80), showlegend=False))
    fig.update_coloraxes(showscale=False)
    return fig

# ── AI ────────────────────────────────────────────────────────────────────────
def get_ai_analysis(df_info, dtype, page_key="exec"):
    try:
        client = anthropic.Anthropic()
        base_prompt = DATASET_CONFIG[dtype]["insights_prompt"]
        page_prompt = AI_PAGE_PROMPTS.get(page_key, "")
        system = f"{base_prompt}\n\nAdditional focus: {page_prompt}" if page_prompt else base_prompt
        with client.messages.stream(
            model=MODEL, max_tokens=900,
            system=system,
            messages=[{"role": "user", "content": f"Dataset:\n{df_info}"}]
        ) as stream:
            for text in stream.text_stream:
                yield text
    except anthropic.AuthenticationError:
        yield "Auth error — check your API key."
    except Exception as e:
        yield f"Error: {e}"

def ask_claude(question, df_info, dtype):
    try:
        client = anthropic.Anthropic()
        with client.messages.stream(
            model=MODEL, max_tokens=800,
            system=(f"You are an expert {dtype} data analyst.\nDataset:\n{df_info}\n"
                    f"Give precise, specific answers using actual numbers from the data. Use bullet points."),
            messages=[{"role": "user", "content": question}]
        ) as stream:
            for text in stream.text_stream:
                yield text
    except anthropic.AuthenticationError:
        yield "Auth error — check your API key."
    except Exception as e:
        yield f"Error: {e}"

# ── REPORT GENERATOR ──────────────────────────────────────────────────────────
def generate_html_report(file_title, dtype, cfg, df, col_analysis, adv_stats, ss):
    import datetime as _dt
    now_str = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    total_missing = int(df.isnull().sum().sum())
    dupes = int(df.duplicated().sum())
    completeness = round((1 - df.isnull().sum().sum() / (df.shape[0]*df.shape[1])) * 100, 1)

    kpi_html = ""
    for nc in col_analysis["numeric"][:6]:
        s = df[nc].dropna()
        kpi_html += (
            f'<div class="kpi-card">'
            f'<div class="kpi-label">{nc.replace("_"," ").upper()}</div>'
            f'<div class="kpi-value">{fmt_plain(s.sum())}</div>'
            f'<div class="kpi-sub">Avg {fmt_plain(s.mean())} · Max {fmt_plain(s.max())}</div>'
            f'</div>'
        )

    charts_html = ""
    try:
        if col_analysis["date"] and col_analysis["numeric"]:
            fig_t = plot_trend(df, col_analysis)
            charts_html += "<h3 class='section-title'>Trend Chart</h3>" + fig_t.to_html(full_html=False, include_plotlyjs="cdn")
        if col_analysis["categorical"] and col_analysis["numeric"]:
            fig_p = plot_pie(df, col_analysis["categorical"][0], col_analysis["numeric"][0])
            charts_html += "<h3 class='section-title'>Category Breakdown</h3>" + fig_p.to_html(full_html=False, include_plotlyjs=False)
    except Exception:
        charts_html = "<p>Charts could not be generated.</p>"

    ai_html = ""
    ai_labels = {"ai_exec": "Executive Summary", "ai_trends": "Trend Analysis",
                 "ai_cat": "Category Insights", "ai_anomaly": "Data Quality Report"}
    for ai_key, label in ai_labels.items():
        txt = ss.get(ai_key, "")
        if txt:
            ai_html += (
                f'<div class="ai-box">'
                f'<div class="ai-title">{label}</div>'
                f'<p>{txt.replace(chr(10), "<br>")}</p>'
                f'</div>'
            )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DataSense AI Report — {file_title}</title>
<style>
body{{font-family:system-ui,sans-serif;background:#f8fafc;color:#0d1f3c;margin:0;padding:0}}
.header{{background:linear-gradient(135deg,#0f172a,#1e3a5f);color:#fff;padding:2rem 3rem}}
.header h1{{margin:0 0 6px;font-size:26px;font-weight:800}}
.header p{{margin:0;color:#94a3b8;font-size:13px}}
.badge{{background:rgba(91,75,255,0.3);color:#c4b5fd;border-radius:6px;padding:2px 10px;font-size:11px;font-weight:700;margin-right:6px}}
.content{{max-width:1100px;margin:0 auto;padding:2rem 3rem}}
.section-title{{font-size:11px;font-weight:700;color:#5b4bff;text-transform:uppercase;letter-spacing:0.1em;border-left:3px solid #5b4bff;padding-left:10px;margin:2rem 0 1rem}}
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(175px,1fr));gap:14px;margin-bottom:2rem}}
.kpi-card{{background:#fff;border-radius:12px;padding:1rem 1.2rem;border:1px solid #edf0f7;box-shadow:0 1px 4px rgba(0,0,0,0.04)}}
.kpi-label{{font-size:10px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:6px}}
.kpi-value{{font-size:24px;font-weight:800;color:#0d1f3c;line-height:1.1;margin-bottom:4px}}
.kpi-sub{{font-size:11px;color:#94a3b8}}
.stats-row{{display:flex;flex-wrap:wrap;gap:0;background:#fff;border-radius:10px;border:1px solid #edf0f7;margin-bottom:1.5rem;overflow:hidden}}
.stat-item{{padding:12px 20px;border-right:1px solid #e2e8f0}}
.stat-label{{font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.06em}}
.stat-val{{font-size:20px;font-weight:800;color:#0d1f3c}}
.ai-box{{background:#fff;border:1px solid #e4e0ff;border-left:3px solid #5b4bff;border-radius:0 12px 12px 0;padding:1rem 1.4rem;margin-bottom:1rem}}
.ai-title{{font-size:11px;font-weight:700;color:#5b4bff;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px}}
.ai-box p{{font-size:13px;color:#1e293b;line-height:1.8;margin:0}}
footer{{background:#0f172a;color:#475569;text-align:center;padding:1.5rem;font-size:12px;margin-top:3rem}}
</style>
</head>
<body>
<div class="header">
  <div style="margin-bottom:10px">
    <span class="badge">{cfg["icon"]} {cfg["label"]}</span>
    <span class="badge">Generated {now_str}</span>
  </div>
  <h1>{file_title}</h1>
  <p>{df.shape[0]:,} rows · {df.shape[1]} columns · DataSense AI Report</p>
</div>
<div class="content">
  <div class="section-title">Data Overview</div>
  <div class="stats-row">
    <div class="stat-item"><div class="stat-label">Rows</div><div class="stat-val">{df.shape[0]:,}</div></div>
    <div class="stat-item"><div class="stat-label">Columns</div><div class="stat-val">{df.shape[1]}</div></div>
    <div class="stat-item"><div class="stat-label">Completeness</div><div class="stat-val">{completeness}%</div></div>
    <div class="stat-item"><div class="stat-label">Missing</div><div class="stat-val">{total_missing:,}</div></div>
    <div class="stat-item"><div class="stat-label">Duplicates</div><div class="stat-val">{dupes:,}</div></div>
    <div class="stat-item"><div class="stat-label">Numeric Cols</div><div class="stat-val">{len(col_analysis["numeric"])}</div></div>
  </div>
  <div class="section-title">Key Performance Indicators</div>
  <div class="kpi-grid">{kpi_html}</div>
  <div class="section-title">Charts</div>
  {charts_html}
  {"<div class='section-title'>AI Insights</div>" + ai_html if ai_html else ""}
</div>
<footer>Generated by DataSense AI · {now_str} · Powered by Claude</footer>
</body>
</html>"""
    return html.encode("utf-8")

# ── FILE STATE ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if "stored_file" not in st.session_state:
    st.session_state["stored_file"] = None
if "show_uploader" not in st.session_state:
    st.session_state["show_uploader"] = False
if "df_modified" not in st.session_state:
    st.session_state["df_modified"] = None
if "df_modified_for" not in st.session_state:
    st.session_state["df_modified_for"] = None

# ══════════════════════════════════════════════════════════════════════════════
# WELCOME
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state["stored_file"]:
    # ── SVG icon definitions ──────────────────────────────────────────────────
    ICONS = {
        "cart":    '<svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><circle cx="9" cy="21" r="1"/><circle cx="20" cy="21" r="1"/><path d="M1 1h4l2.68 13.39a2 2 0 0 0 2 1.61h9.72a2 2 0 0 0 2-1.61L23 6H6"/></svg>',
        "trend":   '<svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>',
        "wallet":  '<svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><path d="M20 12V22H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v4"/><path d="M22 12h-4a2 2 0 0 0 0 4h4v-4z"/><circle cx="18" cy="14" r="1" fill="currentColor" stroke="none"/></svg>',
        "barchart":'<svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/><line x1="2" y1="20" x2="22" y2="20"/></svg>',
        "users":   '<svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
        "handshk": '<svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>',
        "megaph":  '<svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>',
        "target":  '<svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
        "package": '<svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><line x1="16.5" y1="9.4" x2="7.5" y2="4.21"/><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>',
        "boxes":   '<svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><path d="M2.97 12.92A2 2 0 0 0 2 14.63v3.24a2 2 0 0 0 .97 1.71l3 1.8a2 2 0 0 0 2.06 0L12 19v-5.5l-5-3-4.03 2.42Z"/><path d="m7 16.5-4.74-2.85"/><path d="m7 16.5 5-3"/><path d="M7 16.5v5.17"/><path d="M12 13.5V19l3.97 2.38a2 2 0 0 0 2.06 0l3-1.8a2 2 0 0 0 .97-1.71v-3.24a2 2 0 0 0-.97-1.71L17 10.5l-5 3Z"/><path d="m17 16.5-5-3"/><path d="m17 16.5 4.74-2.85"/><path d="M17 16.5v5.17"/><path d="M7.97 4.42A2 2 0 0 0 7 6.13v4.37l5 3 5-3V6.13a2 2 0 0 0-.97-1.71l-3-1.8a2 2 0 0 0-2.06 0l-3 1.8Z"/><path d="M12 8 7.26 5.15"/><path d="m12 8 4.74-2.85"/><path d="M12 13.5V8"/></svg>',
        "grid":    '<svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>',
        "linechart":'<svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    }

    def card(icon1_key, icon2_key, title, desc, bg, title_color, icon_color):
        i1 = ICONS[icon1_key].replace('stroke="currentColor"', f'stroke="{icon_color}"')
        i2 = ICONS[icon2_key].replace('stroke="currentColor"', f'stroke="{icon_color}"').replace('fill="currentColor"', f'fill="{icon_color}"')
        return f"""
        <div style='background:{bg};border-radius:16px;padding:1.4rem 1.5rem 1.3rem;
          border:1px solid rgba(0,0,0,0.06);height:190px;position:relative;overflow:hidden;
          display:flex;flex-direction:column;justify-content:flex-end;margin-bottom:14px;
          box-shadow:0 1px 3px rgba(0,0,0,0.04)'>
          <div style='position:absolute;top:14px;right:14px;opacity:0.55'>{i2}</div>
          <div style='margin-bottom:10px;opacity:0.85'>{i1}</div>
          <div style='font-size:17px;font-weight:700;color:{title_color};margin-bottom:5px;letter-spacing:-0.01em'>{title}</div>
          <div style='font-size:12px;color:#64748b;line-height:1.55'>{desc}</div>
        </div>"""

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── HERO (2-column: text left, product mockup right) ─────────────────────
    hero_l, hero_r = st.columns([1.05, 0.95])
    with hero_l:
        st.markdown("""
        <div style='padding-top:2rem'>
          <span class='slabel'>Powered by Claude AI</span>
          <h1 style='font-size:52px;font-weight:800;color:#0d1f3c;line-height:1.08;
            margin:0 0 20px;letter-spacing:-0.035em'>
            Instant Insights.<br><span style='color:#5b4bff'>Any Spreadsheet.</span>
          </h1>
          <p style='font-size:15.5px;color:#64748b;line-height:1.75;margin:0 0 2.5rem;max-width:420px'>
            Upload a CSV or Excel file. DataSense AI auto-detects your dataset type and
            builds a full analytics dashboard in seconds — KPIs, charts, trends &amp; AI narratives.
          </p>
          <!-- Stats bar -->
          <div style='display:flex;flex-wrap:wrap;gap:0;border:1px solid #e9ecf0;border-radius:12px;
            overflow:hidden;background:#fff;width:fit-content;box-shadow:0 1px 3px rgba(0,0,0,0.05)'>
            <div style='padding:10px 20px;border-right:1px solid #e9ecf0'>
              <div style='font-size:20px;font-weight:800;color:#0d1f3c;line-height:1'>9</div>
              <div style='font-size:11px;color:#94a3b8;margin-top:2px'>Analytics Views</div>
            </div>
            <div style='padding:10px 20px;border-right:1px solid #e9ecf0'>
              <div style='font-size:20px;font-weight:800;color:#0d1f3c;line-height:1'>6</div>
              <div style='font-size:11px;color:#94a3b8;margin-top:2px'>Dataset Types</div>
            </div>
            <div style='padding:10px 20px;border-right:1px solid #e9ecf0'>
              <div style='font-size:20px;font-weight:800;color:#5b4bff;line-height:1'>AI</div>
              <div style='font-size:11px;color:#94a3b8;margin-top:2px'>Claude Powered</div>
            </div>
            <div style='padding:10px 20px'>
              <div style='font-size:20px;font-weight:800;color:#0d1f3c;line-height:1'>0</div>
              <div style='font-size:11px;color:#94a3b8;margin-top:2px'>Setup Required</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with hero_r:
        st.markdown("""
        <div style='padding-top:0.25rem;display:flex;justify-content:center'>
          <div style='box-shadow:0 24px 64px rgba(79,70,229,0.13),0 4px 20px rgba(0,0,0,0.07);
            border-radius:18px;overflow:hidden;width:100%;border:1px solid #e2e8f0'>
            <svg viewBox="0 0 420 308" width="100%" xmlns="http://www.w3.org/2000/svg">
              <rect width="420" height="308" fill="#fff"/>
              <rect width="420" height="40" fill="#f8fafc"/>
              <rect y="39" width="420" height="1" fill="#e2e8f0"/>
              <circle cx="18" cy="20" r="5" fill="#fc5c5c"/>
              <circle cx="32" cy="20" r="5" fill="#fdbc2c"/>
              <circle cx="46" cy="20" r="5" fill="#34c759"/>
              <text x="210" y="24" text-anchor="middle" font-size="11" fill="#94a3b8" font-family="system-ui,sans-serif" font-weight="500">DataSense AI  ·  Sales Dashboard</text>
              <rect y="40" width="420" height="268" fill="#f8fafc"/>
              <rect x="12" y="52" width="122" height="56" rx="10" fill="#fff" stroke="#e2e8f0" stroke-width="1"/>
              <text x="22" y="68" font-size="8" fill="#3b82f6" font-family="system-ui,sans-serif" font-weight="700">REVENUE</text>
              <text x="22" y="87" font-size="19" fill="#1e3a8a" font-family="system-ui,sans-serif" font-weight="800">$23.8K</text>
              <text x="22" y="100" font-size="8" fill="#22c55e" font-family="system-ui,sans-serif" font-weight="600">up 12.4% vs last month</text>
              <rect x="142" y="52" width="122" height="56" rx="10" fill="#fff" stroke="#e2e8f0" stroke-width="1"/>
              <text x="152" y="68" font-size="8" fill="#16a34a" font-family="system-ui,sans-serif" font-weight="700">ORDERS</text>
              <text x="152" y="87" font-size="19" fill="#14532d" font-family="system-ui,sans-serif" font-weight="800">1,240</text>
              <text x="152" y="100" font-size="8" fill="#22c55e" font-family="system-ui,sans-serif" font-weight="600">up 8.1% vs last month</text>
              <rect x="272" y="52" width="136" height="56" rx="10" fill="#fff" stroke="#e2e8f0" stroke-width="1"/>
              <text x="282" y="68" font-size="8" fill="#7c3aed" font-family="system-ui,sans-serif" font-weight="700">GROWTH RATE</text>
              <text x="282" y="87" font-size="19" fill="#4c1d95" font-family="system-ui,sans-serif" font-weight="800">+18.4%</text>
              <text x="282" y="100" font-size="8" fill="#22c55e" font-family="system-ui,sans-serif" font-weight="600">up 3.2% vs last quarter</text>
              <text x="12" y="124" font-size="10" fill="#0f172a" font-family="system-ui,sans-serif" font-weight="700">Revenue Trend</text>
              <text x="272" y="124" font-size="10" fill="#0f172a" font-family="system-ui,sans-serif" font-weight="700">Top Products</text>
              <rect x="12" y="130" width="248" height="122" rx="10" fill="#fff" stroke="#e2e8f0" stroke-width="1"/>
              <line x1="22" y1="162" x2="250" y2="162" stroke="#f1f5f9" stroke-width="1"/>
              <line x1="22" y1="182" x2="250" y2="182" stroke="#f1f5f9" stroke-width="1"/>
              <line x1="22" y1="202" x2="250" y2="202" stroke="#f1f5f9" stroke-width="1"/>
              <line x1="22" y1="222" x2="250" y2="222" stroke="#f1f5f9" stroke-width="1"/>
              <defs>
                <linearGradient id="cg" x1="0" y1="140" x2="0" y2="242" gradientUnits="userSpaceOnUse">
                  <stop offset="0%" stop-color="#4f46e5" stop-opacity="0.18"/>
                  <stop offset="100%" stop-color="#4f46e5" stop-opacity="0.01"/>
                </linearGradient>
              </defs>
              <path d="M22 232 L60 218 L100 200 L140 184 L180 168 L220 152 L250 140 L250 242 L22 242 Z" fill="url(#cg)"/>
              <path d="M22 232 L60 218 L100 200 L140 184 L180 168 L220 152 L250 140" stroke="#4f46e5" stroke-width="2.2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
              <circle cx="22" cy="232" r="3" fill="#fff" stroke="#4f46e5" stroke-width="1.5"/>
              <circle cx="100" cy="200" r="3" fill="#fff" stroke="#4f46e5" stroke-width="1.5"/>
              <circle cx="180" cy="168" r="3" fill="#fff" stroke="#4f46e5" stroke-width="1.5"/>
              <circle cx="250" cy="140" r="4" fill="#4f46e5" stroke="#fff" stroke-width="1.5"/>
              <text x="22"  y="255" font-size="7" fill="#94a3b8" font-family="system-ui,sans-serif" text-anchor="middle">Jan</text>
              <text x="60"  y="255" font-size="7" fill="#94a3b8" font-family="system-ui,sans-serif" text-anchor="middle">Feb</text>
              <text x="100" y="255" font-size="7" fill="#94a3b8" font-family="system-ui,sans-serif" text-anchor="middle">Mar</text>
              <text x="140" y="255" font-size="7" fill="#94a3b8" font-family="system-ui,sans-serif" text-anchor="middle">Apr</text>
              <text x="180" y="255" font-size="7" fill="#94a3b8" font-family="system-ui,sans-serif" text-anchor="middle">May</text>
              <text x="220" y="255" font-size="7" fill="#94a3b8" font-family="system-ui,sans-serif" text-anchor="middle">Jun</text>
              <text x="250" y="255" font-size="7" fill="#94a3b8" font-family="system-ui,sans-serif" text-anchor="middle">Jul</text>
              <rect x="272" y="130" width="136" height="122" rx="10" fill="#fff" stroke="#e2e8f0" stroke-width="1"/>
              <text x="282" y="151" font-size="8" fill="#64748b" font-family="system-ui,sans-serif">Laptop Pro</text>
              <rect x="282" y="154" width="116" height="8" rx="3" fill="#e0e7ff"/>
              <rect x="282" y="154" width="105" height="8" rx="3" fill="#4f46e5"/>
              <text x="282" y="175" font-size="8" fill="#64748b" font-family="system-ui,sans-serif">Monitor X</text>
              <rect x="282" y="178" width="116" height="8" rx="3" fill="#e0e7ff"/>
              <rect x="282" y="178" width="86" height="8" rx="3" fill="#818cf8"/>
              <text x="282" y="199" font-size="8" fill="#64748b" font-family="system-ui,sans-serif">Keyboard</text>
              <rect x="282" y="202" width="116" height="8" rx="3" fill="#e0e7ff"/>
              <rect x="282" y="202" width="64" height="8" rx="3" fill="#a5b4fc"/>
              <text x="282" y="223" font-size="8" fill="#64748b" font-family="system-ui,sans-serif">Mouse Elite</text>
              <rect x="282" y="226" width="116" height="8" rx="3" fill="#e0e7ff"/>
              <rect x="282" y="226" width="44" height="8" rx="3" fill="#c7d2fe"/>
              <rect x="12" y="260" width="396" height="36" rx="10" fill="#faf5ff" stroke="#e9d5ff" stroke-width="1"/>
              <text x="42" y="275" font-size="8.5" fill="#6d28d9" font-family="system-ui,sans-serif" font-weight="700">AI Insight:</text>
              <text x="100" y="275" font-size="8.5" fill="#6d28d9" font-family="system-ui,sans-serif">Revenue grew 18.4% YoY — Laptop Pro drives 91% of category sales.</text>
              <text x="42" y="288" font-size="7.5" fill="#7c3aed" font-family="system-ui,sans-serif" opacity="0.65">Powered by Claude AI · Generated in real-time from your data</text>
            </svg>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── FEATURE HIGHLIGHTS ────────────────────────────────────────────────────
    st.markdown("<div style='height:72px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <span class='slabel'>What You Get</span>
    <h2 style='font-size:30px;font-weight:800;color:#0d1f3c;margin:0 0 6px;letter-spacing:-0.02em'>
      Everything you need to understand your data
    </h2>
    <p style='font-size:14px;color:#64748b;margin:0 0 28px'>
      No configuration. No code. Just upload and explore.
    </p>
    """, unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    FEAT = [
        ("#f5f3ff","#ede9fe","#5b4bff",
         '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/></svg>',
         "Auto-Detect Dataset",
         "Instantly classifies your data as Sales, HR, Finance, Marketing, Inventory, or Generic — no setup needed."),
        ("#f0fdf4","#dcfce7","#059669",
         '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>',
         "Smart Visualizations",
         "Auto-generates trend lines, bar charts, heatmaps and scatter plots — all tailored to your data domain."),
        ("#fffbeb","#fef3c7","#d97706",
         '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>',
         "AI-Powered Analysis",
         "Claude AI writes domain-specific insights grounded in your exact numbers and answers questions in plain English."),
    ]
    for col, (bg, ibg, ic, svg, title, desc) in zip([f1, f2, f3], FEAT):
        icon = svg.replace('stroke="currentColor"', f'stroke="{ic}"')
        with col:
            st.markdown(f"""
            <div style='background:{bg};border-radius:16px;padding:1.75rem 1.75rem 1.6rem;
              border:1px solid rgba(0,0,0,0.05);height:100%'>
              <div style='width:46px;height:46px;background:{ibg};border-radius:13px;
                display:flex;align-items:center;justify-content:center;margin-bottom:18px'>
                {icon}
              </div>
              <div style='font-size:15.5px;font-weight:700;color:#0d1f3c;margin-bottom:10px'>{title}</div>
              <div style='font-size:13.5px;color:#64748b;line-height:1.7'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    # ── DATASET TYPE SECTION HEADING ──────────────────────────────────────────
    st.markdown("""
    <div style='margin:80px 0 28px'>
      <span class='slabel'>Supported Datasets</span>
      <h2 style='font-size:32px;font-weight:800;color:#0d1f3c;margin:0 0 10px;letter-spacing:-0.025em'>
        Works with any dataset
      </h2>
      <p style='font-size:15px;color:#94a3b8;margin:0'>
        Upload your data and DataSense AI figures out everything else.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # 3×2 card grid
    r1c1, r1c2, r1c3 = st.columns(3)
    r2c1, r2c2, r2c3 = st.columns(3)

    with r1c1:
        st.markdown(card("cart",   "trend",   "Sales",   "Revenue, products, regions, salesperson performance", "#eef3fc", "#1e3a8a", "#3b5bdb"), unsafe_allow_html=True)
    with r1c2:
        st.markdown(card("wallet", "barchart","Finance", "Budget vs actual, P&L, expense categories, margins",  "#edfaf3", "#14532d", "#16a34a"), unsafe_allow_html=True)
    with r1c3:
        st.markdown(card("users",  "handshk", "HR",      "Headcount, salary bands, performance, leave analysis","#f3eeff", "#4c1d95", "#7c3aed"), unsafe_allow_html=True)
    with r2c1:
        st.markdown(card("megaph", "target",  "Marketing","ROAS, CTR, CPA, channel performance, conversions",   "#fff0eb", "#7c2d12", "#ea580c"), unsafe_allow_html=True)
    with r2c2:
        st.markdown(card("package","boxes",   "Inventory","Stock levels, suppliers, reorder alerts, utilization","#fef9eb", "#78350f", "#d97706"), unsafe_allow_html=True)
    with r2c3:
        st.markdown(card("linechart","grid",  "Generic", "Works with any tabular dataset automatically",        "#f3f4f6", "#1f2937", "#6b7280"), unsafe_allow_html=True)

    # How it works (full width)
    st.markdown("<div style='height:72px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#f8f7ff;border-radius:20px;padding:2.25rem 2.5rem 1.75rem;
      border:1px solid #e4e0ff;box-shadow:0 1px 4px rgba(91,75,255,0.06)'>
      <span class='slabel'>How It Works</span>
      <p style='font-size:22px;font-weight:800;color:#0d1f3c;margin:0 0 4px;letter-spacing:-0.02em'>
        From upload to insight in seconds
      </p>
    </div>""", unsafe_allow_html=True)

    hw_l, hw_c, hw_r = st.columns([1, 1.4, 1])

    with hw_l:
        st.markdown("""
        <div style='padding:0 0 0 0.5rem;display:flex;flex-direction:column;gap:32px;padding-top:8px'>
          <div style='display:flex;gap:10px;align-items:flex-start'>
            <span style='background:#ede9fe;color:#5b4bff;font-size:11px;font-weight:700;
              padding:3px 8px;border-radius:8px;flex-shrink:0;margin-top:2px'>01</span>
            <span style='font-size:13px;color:#334155;line-height:1.55'>
              <b style="color:#0d1f3c">Upload</b> any CSV or Excel file
            </span>
          </div>
          <div style='display:flex;gap:10px;align-items:flex-start'>
            <span style='background:#ede9fe;color:#5b4bff;font-size:11px;font-weight:700;
              padding:3px 8px;border-radius:8px;flex-shrink:0;margin-top:2px'>03</span>
            <span style='font-size:13px;color:#334155;line-height:1.55'>
              <b style="color:#0d1f3c">Explore</b> KPIs, charts &amp; trends
            </span>
          </div>
        </div>""", unsafe_allow_html=True)

    with hw_c:
            st.markdown("""
            <div style='display:flex;justify-content:center;align-items:center;padding:4px 0'>
            <svg width="180" height="132" viewBox="0 0 180 132" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect x="4" y="4" width="68" height="54" rx="10" fill="#EEF4FF" stroke="#C7D7FD" stroke-width="1.2"/>
              <path d="M38 34v-13M32 27l6-6 6 6" stroke="#4F76F6" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
              <rect x="22" y="44" width="30" height="3" rx="1.5" fill="#C7D7FD"/>
              <rect x="26" y="50" width="22" height="3" rx="1.5" fill="#C7D7FD"/>
              <rect x="108" y="4" width="68" height="54" rx="10" fill="#F0FDF4" stroke="#A7F3D0" stroke-width="1.2"/>
              <rect x="120" y="16" width="44" height="30" rx="4" fill="none" stroke="#34D399" stroke-width="1.4"/>
              <line x1="120" y1="26" x2="164" y2="26" stroke="#34D399" stroke-width="1.2"/>
              <line x1="142" y1="16" x2="142" y2="46" stroke="#34D399" stroke-width="1.2"/>
              <circle cx="153" cy="38" r="7" fill="#F0FDF4" stroke="#34D399" stroke-width="1.4"/>
              <line x1="158" y1="43" x2="162" y2="47" stroke="#34D399" stroke-width="1.8" stroke-linecap="round"/>
              <path d="M74 28 Q90 20 106 28" stroke="#CBD5E1" stroke-width="1.4" stroke-dasharray="3 2" fill="none"/>
              <polygon points="106,24 108,30 102,28" fill="#CBD5E1"/>
              <rect x="4" y="74" width="68" height="52" rx="10" fill="#FFF7ED" stroke="#FED7AA" stroke-width="1.2"/>
              <rect x="18" y="104" width="8" height="10" rx="1" fill="#FB923C"/>
              <rect x="30" y="96" width="8" height="18" rx="1" fill="#FB923C" opacity=".75"/>
              <rect x="42" y="88" width="8" height="26" rx="1" fill="#FB923C" opacity=".5"/>
              <polyline points="16,100 30,90 44,94 58,82" stroke="#EF4444" stroke-width="1.6" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
              <rect x="108" y="74" width="68" height="52" rx="10" fill="#EFF6FF" stroke="#BFDBFE" stroke-width="1.2"/>
              <rect x="124" y="82" width="32" height="28" rx="8" fill="none" stroke="#3B82F6" stroke-width="1.4"/>
              <circle cx="136" cy="95" r="2.5" fill="#3B82F6"/>
              <circle cx="148" cy="95" r="2.5" fill="#3B82F6"/>
              <path d="M133 103 Q140 107 148 103" stroke="#3B82F6" stroke-width="1.3" stroke-linecap="round" fill="none"/>
              <line x1="140" y1="82" x2="140" y2="78" stroke="#3B82F6" stroke-width="1.4" stroke-linecap="round"/>
              <circle cx="140" cy="76" r="2.5" fill="#3B82F6"/>
              <circle cx="162" cy="84" r="5" fill="#EFF6FF" stroke="#3B82F6" stroke-width="1.2"/>
              <text x="159.5" y="87" font-size="6" fill="#3B82F6" font-weight="bold">?</text>
              <line x1="38" y1="60" x2="38" y2="72" stroke="#CBD5E1" stroke-width="1.4" stroke-dasharray="3 2"/>
              <polygon points="34,72 38,78 42,72" fill="#CBD5E1"/>
              <path d="M106 100 Q90 108 74 100" stroke="#CBD5E1" stroke-width="1.4" stroke-dasharray="3 2" fill="none"/>
              <polygon points="74,96 72,102 78,100" fill="#CBD5E1"/>
              <line x1="142" y1="72" x2="142" y2="60" stroke="#CBD5E1" stroke-width="1.4" stroke-dasharray="3 2"/>
              <polygon points="138,60 142,54 146,60" fill="#CBD5E1"/>
            </svg>
            </div>""", unsafe_allow_html=True)

    with hw_r:
        st.markdown("""
        <div style='padding:0 0.5rem 0 0;display:flex;flex-direction:column;gap:32px;padding-top:8px'>
          <div style='display:flex;gap:10px;align-items:flex-start'>
            <span style='background:#ede9fe;color:#5b4bff;font-size:11px;font-weight:700;
              padding:3px 8px;border-radius:8px;flex-shrink:0;margin-top:2px'>02</span>
            <span style='font-size:13px;color:#334155;line-height:1.55'>
              <b style="color:#0d1f3c">Auto-detect</b> dataset type instantly
            </span>
          </div>
          <div style='display:flex;gap:10px;align-items:flex-start'>
            <span style='background:#ede9fe;color:#5b4bff;font-size:11px;font-weight:700;
              padding:3px 8px;border-radius:8px;flex-shrink:0;margin-top:2px'>04</span>
            <span style='font-size:13px;color:#334155;line-height:1.55'>
              <b style="color:#0d1f3c">Ask</b> any question via Data Agent
            </span>
          </div>
        </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION: 9 ANALYTICS VIEWS
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("<div style='height:88px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center;margin-bottom:48px'>
      <span class='slabel' style='display:inline-block'>9 Built-In Views</span>
      <h2 style='font-size:34px;font-weight:800;color:#0d1f3c;margin:8px 0 12px;letter-spacing:-0.025em'>
        Every angle your data needs
      </h2>
      <p style='font-size:15px;color:#64748b;margin:0 auto;max-width:520px;line-height:1.7'>
        No manual chart-building. DataSense AI generates all nine views automatically
        the moment you upload your file.
      </p>
    </div>
    """, unsafe_allow_html=True)

    VIEWS = [
        ("🏠", "Overview",      "#f5f3ff", "#5b4bff",
         "Top KPIs + trend chart at a glance. The executive view."),
        ("💡", "Smart Insights", "#f0fdf4", "#059669",
         "Domain-specific insight cards with exact metric values from your data."),
        ("📊", "KPI Report",     "#fffbeb", "#d97706",
         "Detailed statistics, distributions and value breakdowns per column."),
        ("📈", "Trends",         "#eff6ff", "#2563eb",
         "Time-series lines + monthly performance heatmaps across segments."),
        ("📦", "Categories",     "#fdf2f8", "#db2777",
         "Segment breakdowns — bar charts, pie charts and cross-tab analysis."),
        ("🔗", "Correlations",   "#f0fdf4", "#059669",
         "Metric relationships, scatter plots and correlation heatmap matrix."),
        ("🔍", "Anomalies",      "#fff7ed", "#ea580c",
         "Data quality report: outliers, missing values, duplicates flagged."),
        ("🤖", "AI Analysis",    "#f5f3ff", "#5b4bff",
         "Deep Claude-generated narrative tailored to your exact dataset type."),
        ("📋", "Data Table",     "#f8fafc", "#475569",
         "Searchable, filterable raw data view with one-click CSV export."),
    ]

    va, vb, vc = st.columns(3)
    for i, (icon, name, bg, color, desc) in enumerate(VIEWS):
        col = [va, vb, vc][i % 3]
        with col:
            st.markdown(f"""
            <div style='background:{bg};border-radius:14px;padding:1.4rem 1.5rem;
              border:1px solid rgba(0,0,0,0.05);margin-bottom:16px;
              display:flex;gap:14px;align-items:flex-start'>
              <div style='width:40px;height:40px;border-radius:10px;background:#fff;
                display:flex;align-items:center;justify-content:center;
                font-size:18px;flex-shrink:0;box-shadow:0 1px 4px rgba(0,0,0,0.07)'>
                {icon}
              </div>
              <div>
                <div style='font-size:14px;font-weight:700;color:{color};margin-bottom:4px'>
                  {name}
                </div>
                <div style='font-size:12.5px;color:#64748b;line-height:1.6'>{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION: DATA AGENT SPOTLIGHT
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("<div style='height:88px'></div>", unsafe_allow_html=True)
    ag_l, ag_r = st.columns([1.1, 0.9])
    with ag_l:
        st.markdown("""
        <div style='padding-top:1.5rem;padding-right:2rem'>
          <span class='slabel'>Data Agent</span>
          <h2 style='font-size:34px;font-weight:800;color:#0d1f3c;margin:8px 0 16px;
            letter-spacing:-0.025em;line-height:1.15'>
            Ask questions.<br>Get answers instantly.
          </h2>
          <p style='font-size:15px;color:#64748b;line-height:1.75;margin:0 0 28px'>
            The Data Agent translates plain-English questions into live pandas
            queries on your dataset — returning tables, charts and plain-English
            summaries in seconds. No SQL. No Python. Just ask.
          </p>
          <div style='display:flex;flex-direction:column;gap:14px'>
            <div style='display:flex;align-items:flex-start;gap:12px'>
              <div style='width:32px;height:32px;background:#ede9fe;border-radius:8px;
                display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:15px'>🔍</div>
              <div>
                <div style='font-size:13.5px;font-weight:600;color:#0d1f3c;margin-bottom:2px'>Filter &amp; search</div>
                <div style='font-size:12.5px;color:#64748b'>"Show me all orders above $10,000 in Q3"</div>
              </div>
            </div>
            <div style='display:flex;align-items:flex-start;gap:12px'>
              <div style='width:32px;height:32px;background:#dcfce7;border-radius:8px;
                display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:15px'>📊</div>
              <div>
                <div style='font-size:13.5px;font-weight:600;color:#0d1f3c;margin-bottom:2px'>Aggregate &amp; group</div>
                <div style='font-size:12.5px;color:#64748b'>"What is total revenue by region this year?"</div>
              </div>
            </div>
            <div style='display:flex;align-items:flex-start;gap:12px'>
              <div style='width:32px;height:32px;background:#fef3c7;border-radius:8px;
                display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:15px'>📈</div>
              <div>
                <div style='font-size:13.5px;font-weight:600;color:#0d1f3c;margin-bottom:2px'>Generate charts</div>
                <div style='font-size:12.5px;color:#64748b'>"Plot monthly sales trend as a line chart"</div>
              </div>
            </div>
            <div style='display:flex;align-items:flex-start;gap:12px'>
              <div style='width:32px;height:32px;background:#fee2e2;border-radius:8px;
                display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:15px'>🧮</div>
              <div>
                <div style='font-size:13.5px;font-weight:600;color:#0d1f3c;margin-bottom:2px'>Calculate stats</div>
                <div style='font-size:12.5px;color:#64748b'>"What is the average salary by department?"</div>
              </div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
    with ag_r:
        st.markdown("""
        <div style='background:linear-gradient(145deg,#0d1f3c,#1e3a5f);border-radius:20px;
          padding:1.5rem;box-shadow:0 24px 64px rgba(13,31,60,0.22)'>
          <div style='display:flex;align-items:center;gap:8px;margin-bottom:16px'>
            <div style='width:8px;height:8px;border-radius:50%;background:#ef4444'></div>
            <div style='width:8px;height:8px;border-radius:50%;background:#f59e0b'></div>
            <div style='width:8px;height:8px;border-radius:50%;background:#22c55e'></div>
            <span style='font-size:11px;color:#64748b;margin-left:6px;font-weight:500'>Data Agent</span>
          </div>
          <div style='display:flex;flex-direction:column;gap:12px'>
            <div style='background:rgba(91,75,255,0.2);border-radius:12px 12px 4px 12px;
              padding:10px 14px;align-self:flex-end;max-width:85%'>
              <p style='font-size:12.5px;color:#c7d2fe;margin:0;line-height:1.5'>
                Show me the top 5 products by revenue
              </p>
            </div>
            <div style='background:rgba(255,255,255,0.07);border-radius:4px 12px 12px 12px;
              padding:12px 14px;max-width:95%'>
              <p style='font-size:11px;color:#94a3b8;margin:0 0 8px;font-weight:600;
                text-transform:uppercase;letter-spacing:0.06em'>🤖 Data Agent</p>
              <p style='font-size:12.5px;color:#e2e8f0;margin:0 0 10px;line-height:1.55'>
                Here are the top 5 products by total revenue:
              </p>
              <div style='background:rgba(0,0,0,0.2);border-radius:8px;padding:8px 10px;
                font-size:11.5px;color:#94a3b8;font-family:monospace'>
                <div style='display:flex;justify-content:space-between;border-bottom:1px solid rgba(255,255,255,0.07);padding-bottom:5px;margin-bottom:5px'>
                  <span style='color:#c7d2fe;font-weight:600'>Product</span>
                  <span style='color:#c7d2fe;font-weight:600'>Revenue</span>
                </div>
                <div style='display:flex;justify-content:space-between;margin-bottom:3px'>
                  <span>Laptop Pro</span><span style='color:#34d399'>$84,200</span>
                </div>
                <div style='display:flex;justify-content:space-between;margin-bottom:3px'>
                  <span>Monitor X</span><span style='color:#34d399'>$62,100</span>
                </div>
                <div style='display:flex;justify-content:space-between;margin-bottom:3px'>
                  <span>Keyboard Elite</span><span style='color:#34d399'>$38,450</span>
                </div>
                <div style='display:flex;justify-content:space-between;margin-bottom:3px'>
                  <span>Mouse Pro</span><span style='color:#34d399'>$29,800</span>
                </div>
                <div style='display:flex;justify-content:space-between'>
                  <span>Webcam HD</span><span style='color:#34d399'>$21,600</span>
                </div>
              </div>
            </div>
            <div style='background:rgba(91,75,255,0.2);border-radius:12px 12px 4px 12px;
              padding:10px 14px;align-self:flex-end;max-width:85%'>
              <p style='font-size:12.5px;color:#c7d2fe;margin:0;line-height:1.5'>
                Now show it as a bar chart
              </p>
            </div>
            <div style='background:rgba(255,255,255,0.07);border-radius:4px 12px 12px 12px;
              padding:10px 14px;max-width:95%'>
              <p style='font-size:12px;color:#64748b;margin:0;line-height:1.5'>
                📊 Chart generated — Laptop Pro leads with 36% of top-5 revenue share.
              </p>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION: WHY DATASENSE AI — BENEFITS
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("<div style='height:88px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0d1f3c 0%,#1e3a5f 100%);
      border-radius:24px;padding:3.5rem 3rem 3rem;'>
      <div style='text-align:center;margin-bottom:40px'>
        <span class='slabel' style='color:#a78bfa'>Why DataSense AI</span>
        <h2 style='font-size:34px;font-weight:800;color:#f1f5f9;margin:8px 0 12px;
          letter-spacing:-0.025em'>
          Ditch the spreadsheet bottleneck
        </h2>
        <p style='font-size:15px;color:#94a3b8;margin:0 auto;max-width:500px;line-height:1.7'>
          Most teams spend hours building reports manually. DataSense AI delivers
          the same results in seconds — automatically.
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    ben_cols = st.columns(2)
    BENEFITS = [
        ("⏱️", "Hours → Seconds",
         "What used to take an analyst half a day — cleaning data, building charts, writing commentary — DataSense AI does in under 10 seconds.",
         "#f5f3ff", "#5b4bff"),
        ("🧠", "AI That Reads Your Data",
         "Claude AI doesn't just show charts — it reads your actual numbers and writes specific, grounded insights like a senior analyst would.",
         "#f0fdf4", "#059669"),
        ("🎯", "Domain-Specific Intelligence",
         "Sales, HR, Finance, Marketing, Inventory — each dataset type gets a tailored set of KPIs, charts and AI prompts built for that domain.",
         "#fffbeb", "#d97706"),
        ("💬", "Ask In Plain English",
         "No SQL, no Python, no formulas. Type a question in plain English and the Data Agent queries your dataset and returns results instantly.",
         "#eff6ff", "#2563eb"),
        ("📤", "Zero Setup, Zero Config",
         "Drop a CSV or Excel file. That's it. No schema mapping, no column renaming, no data transformation required before analysis.",
         "#fdf2f8", "#db2777"),
        ("🔒", "Your Data Stays Local",
         "Files are processed in your browser session only. Nothing is stored server-side. Your data never leaves your Streamlit session.",
         "#fff7ed", "#ea580c"),
    ]
    for i, (icon, title, desc, bg, color) in enumerate(BENEFITS):
        with ben_cols[i % 2]:
            st.markdown(f"""
            <div style='background:{bg};border-radius:16px;padding:1.5rem 1.75rem;
              margin-bottom:16px;border:1px solid rgba(0,0,0,0.05)'>
              <div style='font-size:24px;margin-bottom:10px'>{icon}</div>
              <div style='font-size:15px;font-weight:700;color:{color};margin-bottom:8px'>{title}</div>
              <div style='font-size:13.5px;color:#475569;line-height:1.7'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION: WHO IT'S FOR
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("<div style='height:88px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center;margin-bottom:48px'>
      <span class='slabel' style='display:inline-block'>Use Cases</span>
      <h2 style='font-size:34px;font-weight:800;color:#0d1f3c;margin:8px 0 12px;
        letter-spacing:-0.025em'>
        Built for everyone who works with data
      </h2>
      <p style='font-size:15px;color:#64748b;margin:0 auto;max-width:500px;line-height:1.7'>
        Whether you're a business analyst, team manager, or a founder
        — DataSense AI meets you where you are.
      </p>
    </div>
    """, unsafe_allow_html=True)

    wif1, wif2, wif3 = st.columns(3)
    PERSONAS = [
        ("👩‍💼", "Business Analysts",   "#f5f3ff", "#5b4bff", "#ede9fe",
         [("📁","Upload client data and generate a full report in seconds"),
          ("🔍","Surface anomalies and data quality issues automatically"),
          ("📊","Export charts and KPI summaries for presentations"),
          ("💬","Use Data Agent to answer ad-hoc stakeholder questions")]),
        ("📋", "Team Managers",        "#f0fdf4", "#059669", "#dcfce7",
         [("📈","Track team KPIs — revenue, headcount, performance"),
          ("🏆","Identify top performers and underperforming segments"),
          ("📅","Spot trends over time without needing a data team"),
          ("🎯","Get AI-written summaries you can paste straight into reports")]),
        ("🚀", "Founders & Operators", "#fffbeb", "#d97706", "#fef3c7",
         [("💰","Understand your sales funnel and revenue drivers instantly"),
          ("📦","Monitor inventory, margins and supplier performance"),
          ("🌍","Analyse regional performance and market penetration"),
          ("🤖","Get Claude AI to write the analysis narrative for you")]),
    ]
    for col, (icon, title, bg, color, ibg, points) in zip([wif1, wif2, wif3], PERSONAS):
        pts_html = "".join(
            f"<div style='display:flex;gap:10px;align-items:flex-start;margin-bottom:10px'>"
            f"<span style='font-size:14px;flex-shrink:0'>{p[0]}</span>"
            f"<span style='font-size:13px;color:#475569;line-height:1.55'>{p[1]}</span>"
            f"</div>" for p in points
        )
        with col:
            st.markdown(f"""
            <div style='background:{bg};border-radius:20px;padding:1.75rem 1.75rem 1.5rem;
              border:1px solid rgba(0,0,0,0.05);height:100%'>
              <div style='width:52px;height:52px;background:{ibg};border-radius:14px;
                display:flex;align-items:center;justify-content:center;
                font-size:24px;margin-bottom:16px'>
                {icon}
              </div>
              <div style='font-size:17px;font-weight:800;color:{color};
                margin-bottom:16px;letter-spacing:-0.01em'>{title}</div>
              {pts_html}
            </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION: COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("<div style='height:88px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center;margin-bottom:40px'>
      <span class='slabel' style='display:inline-block'>Comparison</span>
      <h2 style='font-size:34px;font-weight:800;color:#0d1f3c;margin:8px 0 12px;
        letter-spacing:-0.025em'>
        DataSense AI vs. doing it manually
      </h2>
    </div>
    <div style='overflow-x:auto'>
    <table style='width:100%;border-collapse:collapse;font-size:13.5px;'>
      <thead>
        <tr>
          <th style='text-align:left;padding:14px 20px;background:#f8f7ff;
            border-bottom:2px solid #e4e0ff;color:#0d1f3c;font-weight:700;
            border-radius:12px 0 0 0'>Task</th>
          <th style='text-align:center;padding:14px 20px;background:#f8f7ff;
            border-bottom:2px solid #e4e0ff;color:#64748b;font-weight:600'>
            Manual (Excel / Tableau)</th>
          <th style='text-align:center;padding:14px 20px;
            background:linear-gradient(135deg,#5b4bff,#7c3aed);
            border-bottom:2px solid #4c3de0;color:#fff;font-weight:700;
            border-radius:0 12px 0 0'>DataSense AI</th>
        </tr>
      </thead>
      <tbody>
        <tr style='border-bottom:1px solid #f1f5f9'>
          <td style='padding:13px 20px;color:#334155;font-weight:500'>Time to first insight</td>
          <td style='padding:13px 20px;text-align:center;color:#94a3b8'>2–4 hours</td>
          <td style='padding:13px 20px;text-align:center;font-weight:700;color:#5b4bff'>
            &lt; 10 seconds</td>
        </tr>
        <tr style='border-bottom:1px solid #f1f5f9;background:#fafafa'>
          <td style='padding:13px 20px;color:#334155;font-weight:500'>Requires technical skills</td>
          <td style='padding:13px 20px;text-align:center;color:#94a3b8'>SQL / Python / BI tool</td>
          <td style='padding:13px 20px;text-align:center;color:#059669;font-weight:700'>None</td>
        </tr>
        <tr style='border-bottom:1px solid #f1f5f9'>
          <td style='padding:13px 20px;color:#334155;font-weight:500'>Auto-detect dataset type</td>
          <td style='padding:13px 20px;text-align:center;color:#ef4444'>✗</td>
          <td style='padding:13px 20px;text-align:center;color:#059669;font-weight:700'>✓</td>
        </tr>
        <tr style='border-bottom:1px solid #f1f5f9;background:#fafafa'>
          <td style='padding:13px 20px;color:#334155;font-weight:500'>AI-written narrative insights</td>
          <td style='padding:13px 20px;text-align:center;color:#ef4444'>✗</td>
          <td style='padding:13px 20px;text-align:center;color:#059669;font-weight:700'>✓ Claude AI</td>
        </tr>
        <tr style='border-bottom:1px solid #f1f5f9'>
          <td style='padding:13px 20px;color:#334155;font-weight:500'>Plain-English data queries</td>
          <td style='padding:13px 20px;text-align:center;color:#ef4444'>✗</td>
          <td style='padding:13px 20px;text-align:center;color:#059669;font-weight:700'>✓ Data Agent</td>
        </tr>
        <tr style='border-bottom:1px solid #f1f5f9;background:#fafafa'>
          <td style='padding:13px 20px;color:#334155;font-weight:500'>Anomaly &amp; outlier detection</td>
          <td style='padding:13px 20px;text-align:center;color:#94a3b8'>Manual</td>
          <td style='padding:13px 20px;text-align:center;color:#059669;font-weight:700'>✓ Automatic</td>
        </tr>
        <tr>
          <td style='padding:13px 20px;color:#334155;font-weight:500'>Setup &amp; configuration</td>
          <td style='padding:13px 20px;text-align:center;color:#94a3b8'>Schema mapping, ETL</td>
          <td style='padding:13px 20px;text-align:center;color:#059669;font-weight:700'>Zero</td>
        </tr>
      </tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION: TECH STACK TRUST BAR
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("<div style='height:80px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center;margin-bottom:28px'>
      <span class='slabel' style='display:inline-block'>Powered By</span>
      <h3 style='font-size:20px;font-weight:700;color:#0d1f3c;margin:8px 0 0;
        letter-spacing:-0.01em'>
        Enterprise-grade technology, zero enterprise complexity
      </h3>
    </div>
    <div style='display:flex;flex-wrap:wrap;justify-content:center;gap:16px'>
      <div style='background:#fff;border:1px solid #e9ecf0;border-radius:12px;
        padding:14px 24px;display:flex;align-items:center;gap:10px;
        box-shadow:0 1px 3px rgba(0,0,0,0.04)'>
        <span style='font-size:22px'>🤖</span>
        <div>
          <div style='font-size:12px;font-weight:700;color:#0d1f3c'>Claude AI</div>
          <div style='font-size:11px;color:#94a3b8'>by Anthropic</div>
        </div>
      </div>
      <div style='background:#fff;border:1px solid #e9ecf0;border-radius:12px;
        padding:14px 24px;display:flex;align-items:center;gap:10px;
        box-shadow:0 1px 3px rgba(0,0,0,0.04)'>
        <span style='font-size:22px'>🐼</span>
        <div>
          <div style='font-size:12px;font-weight:700;color:#0d1f3c'>Pandas</div>
          <div style='font-size:11px;color:#94a3b8'>Data processing</div>
        </div>
      </div>
      <div style='background:#fff;border:1px solid #e9ecf0;border-radius:12px;
        padding:14px 24px;display:flex;align-items:center;gap:10px;
        box-shadow:0 1px 3px rgba(0,0,0,0.04)'>
        <span style='font-size:22px'>📊</span>
        <div>
          <div style='font-size:12px;font-weight:700;color:#0d1f3c'>Plotly</div>
          <div style='font-size:11px;color:#94a3b8'>Interactive charts</div>
        </div>
      </div>
      <div style='background:#fff;border:1px solid #e9ecf0;border-radius:12px;
        padding:14px 24px;display:flex;align-items:center;gap:10px;
        box-shadow:0 1px 3px rgba(0,0,0,0.04)'>
        <span style='font-size:22px'>⚡</span>
        <div>
          <div style='font-size:12px;font-weight:700;color:#0d1f3c'>Streamlit</div>
          <div style='font-size:11px;color:#94a3b8'>Web framework</div>
        </div>
      </div>
      <div style='background:#fff;border:1px solid #e9ecf0;border-radius:12px;
        padding:14px 24px;display:flex;align-items:center;gap:10px;
        box-shadow:0 1px 3px rgba(0,0,0,0.04)'>
        <span style='font-size:22px'>🔢</span>
        <div>
          <div style='font-size:12px;font-weight:700;color:#0d1f3c'>NumPy</div>
          <div style='font-size:11px;color:#94a3b8'>Numerical analysis</div>
        </div>
      </div>
      <div style='background:#fff;border:1px solid #e9ecf0;border-radius:12px;
        padding:14px 24px;display:flex;align-items:center;gap:10px;
        box-shadow:0 1px 3px rgba(0,0,0,0.04)'>
        <span style='font-size:22px'>📁</span>
        <div>
          <div style='font-size:12px;font-weight:700;color:#0d1f3c'>OpenPyXL</div>
          <div style='font-size:11px;color:#94a3b8'>Excel support</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION: FINAL CTA
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("<div style='height:88px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:linear-gradient(135deg,#5b4bff 0%,#7c3aed 100%);
      border-radius:24px;padding:4rem 3rem;text-align:center;
      box-shadow:0 24px 64px rgba(91,75,255,0.28)'>
      <span style='display:inline-block;background:rgba(255,255,255,0.15);
        border:1px solid rgba(255,255,255,0.25);border-radius:20px;
        padding:4px 14px;font-size:11.5px;font-weight:700;color:#e0d9ff;
        letter-spacing:0.08em;text-transform:uppercase;margin-bottom:20px'>
        Free · No sign-up · Instant
      </span>
      <h2 style='font-size:40px;font-weight:800;color:#fff;margin:0 0 16px;
        letter-spacing:-0.03em;line-height:1.1'>
        Ready to understand<br>your data?
      </h2>
      <p style='font-size:16px;color:rgba(255,255,255,0.75);margin:0 auto 36px;
        max-width:440px;line-height:1.7'>
        Drop any CSV or Excel file and get a full AI-powered analytics
        dashboard in under 10 seconds.
      </p>
      <div style='display:flex;justify-content:center;gap:32px;flex-wrap:wrap'>
        <div style='text-align:center'>
          <div style='font-size:28px;font-weight:800;color:#fff'>9</div>
          <div style='font-size:12px;color:rgba(255,255,255,0.6)'>Analytics views</div>
        </div>
        <div style='text-align:center'>
          <div style='font-size:28px;font-weight:800;color:#fff'>6</div>
          <div style='font-size:12px;color:rgba(255,255,255,0.6)'>Dataset types</div>
        </div>
        <div style='text-align:center'>
          <div style='font-size:28px;font-weight:800;color:#fff'>0</div>
          <div style='font-size:12px;color:rgba(255,255,255,0.6)'>Setup required</div>
        </div>
        <div style='text-align:center'>
          <div style='font-size:28px;font-weight:800;color:#fff'>∞</div>
          <div style='font-size:12px;color:rgba(255,255,255,0.6)'>Questions to ask</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── CTA BUTTONS (real Streamlit widgets, centered) ────────────────────────
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    _sample_path = os.path.join(_HERE, "sample_sales_data.csv")
    _cta_l, _cta_c1, _cta_c2, _cta_r = st.columns([1.2, 1.3, 1.3, 1.2])

    with _cta_c1:
        if st.button("📂  Upload your data", use_container_width=True, type="primary",
                     key="cta_upload_btn"):
            st.session_state["show_uploader"] = True
            st.rerun()

    with _cta_c2:
        if os.path.exists(_sample_path):
            if st.button("⚡  Try sample data", use_container_width=True, type="primary",
                         key="cta_sample_btn"):
                with open(_sample_path, "rb") as _f:
                    st.session_state["stored_file"] = {"bytes": _f.read(),
                                                       "name": "sample_sales_data.csv"}
                st.rerun()

    # ── UPLOADER REVEAL (shown only after clicking Upload button) ─────────────
    if st.session_state.get("show_uploader"):
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        _ul, _uc, _ur = st.columns([0.8, 2.4, 0.8])
        with _uc:
            # Header only — no merging with native dropzone
            st.markdown("""
            <div style='text-align:center;padding:0.5rem 0 1rem'>
              <div style='font-size:38px;line-height:1;margin-bottom:10px'>☁️</div>
              <div style='font-size:17px;font-weight:700;color:#0d1f3c;margin-bottom:4px'>
                Upload your data file
              </div>
              <div style='font-size:12.5px;color:#94a3b8;letter-spacing:0.04em'>
                CSV &nbsp;·&nbsp; XLSX &nbsp;·&nbsp; XLS &nbsp;·&nbsp; up to 200 MB
              </div>
            </div>""", unsafe_allow_html=True)
            # Native uploader with collapsed label — no custom text that overlaps button
            _new_file = st.file_uploader(
                "upload", type=["csv", "xlsx", "xls"],
                label_visibility="collapsed", key="cta_uploader"
            )
            if _new_file:
                _data = _new_file.read()
                st.session_state["stored_file"] = {"bytes": _data, "name": _new_file.name}
                st.session_state.pop("show_uploader", None)
                st.rerun()
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            _xl, _xc, _xr = st.columns([1, 1.4, 1])
            with _xc:
                if st.button("✕  Close", use_container_width=True, key="hide_uploader"):
                    st.session_state.pop("show_uploader", None)
                    st.rerun()

    st.markdown("<div style='height:56px'></div>", unsafe_allow_html=True)

    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
_stored = st.session_state["stored_file"]
_file   = NamedBytesIO(_stored["bytes"], _stored["name"])
try:
    df = load_dataframe(_file)
except Exception as e:
    st.error(f"Could not load file: {e}")
    if st.button("← Back to home"):
        st.session_state["stored_file"] = None; st.rerun()
    st.stop()

# Apply cleaned/modified df if it belongs to the current file
if (st.session_state.get("df_modified_for") == _stored["name"]
        and st.session_state.get("df_modified") is not None):
    df = st.session_state["df_modified"]

if st.session_state.get("last_file") != _stored["name"]:
    st.session_state["active_view"] = "overview"
    st.session_state["last_file"]   = _stored["name"]
    st.session_state["messages"]    = []
    st.session_state["df_modified"] = None
    st.session_state["df_modified_for"] = None
    for k in list(st.session_state.keys()):
        if k.startswith("ai_"): del st.session_state[k]

# Scroll to top of page whenever dashboard renders
st.markdown("""<script>
(function() {
    var main = window.parent.document.querySelector('[data-testid="stMain"]');
    if (main) main.scrollTop = 0;
})();
</script>""", unsafe_allow_html=True)

col_analysis  = analyze_columns(df)
dtype         = detect_type(df)
cfg           = DATASET_CONFIG[dtype]
adv_stats     = compute_advanced_stats(df, col_analysis)
correlations  = compute_correlations(df, col_analysis)
time_insights = compute_time_insights(df, col_analysis)
smart_insights= compute_smart_insights(df, col_analysis, dtype)

# ── TOP BAR: file info + actions ─────────────────────────────────────────────
file_title = (_stored["name"].replace(".csv","").replace(".xlsx","").replace(".xls","")
              .replace("_"," ").replace("-"," ").title())

tb1, tb2, tb3, tb4, tb5 = st.columns([4, 0.85, 0.95, 0.95, 0.8])
with tb1:
    _modified_badge = (' <span style="background:#fef3c7;border:1px solid #fde68a;color:#92400e;'
                       'border-radius:6px;padding:1px 7px;font-size:10px;font-weight:700">✏ Cleaned</span>'
                       if st.session_state.get("df_modified_for") == _stored["name"] else "")
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;padding:6px 0 2px;flex-wrap:wrap;min-width:0;overflow:hidden">'
        f'<div style="background:#f0eeff;border:1px solid #c4b5fd;border-radius:8px;padding:4px 10px;'
        f'display:flex;align-items:center;gap:5px;flex-shrink:0">'
        f'<span style="font-size:13px">{cfg["icon"]}</span>'
        f'<span style="color:#5b4bff;font-size:11px;font-weight:700;white-space:nowrap">{cfg["label"]}</span>'
        f'</div>'
        f'<span style="font-size:15px;font-weight:700;color:#0d1f3c;white-space:nowrap;overflow:hidden;'
        f'text-overflow:ellipsis;max-width:220px">{file_title}</span>'
        f'<span style="font-size:11.5px;color:#94a3b8;white-space:nowrap;flex-shrink:0">'
        f'{df.shape[0]:,} rows · {df.shape[1]} cols</span>'
        f'{_modified_badge}'
        f'</div>',
        unsafe_allow_html=True)
with tb2:
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇ CSV", csv, f"{file_title}.csv", "text/csv",
                       use_container_width=True, help="Download dataset as CSV")
with tb3:
    _report_bytes = generate_html_report(file_title, dtype, cfg, df, col_analysis, adv_stats, st.session_state)
    st.download_button("📄 Report", _report_bytes, f"{file_title}_report.html", "text/html",
                       use_container_width=True, help="Download full HTML report")
with tb4:
    if st.button("🔄 Refresh", use_container_width=True, help="Regenerate all AI insights"):
        for k in list(st.session_state.keys()):
            if k.startswith("ai_"): del st.session_state[k]
        st.rerun()
with tb5:
    if st.button("↩ Home", use_container_width=True, help="Back to home"):
        st.session_state["stored_file"] = None
        for k in list(st.session_state.keys()):
            if k.startswith(("df_","ai_","last_","messages","active_","df_modified")): del st.session_state[k]
        st.rerun()

# ── HORIZONTAL NAV ───────────────────────────────────────────────────────────
pages = [("🏠 Overview","overview"),("💡 Insights","smart"),
         ("📊 KPIs","kpis"),("📈 Trends","trends"),
         ("📦 Categories","categories"),("🔗 Correlations","correlations"),
         ("🔍 Anomalies","anomalies"),("🤖 AI Analysis","ai"),
         ("📋 Data Table","table"),("🤖 Data Agent","agent"),
         ("🧹 Clean","clean"),("🔮 Forecast","forecast")]

if not col_analysis["date"] or not col_analysis["numeric"]:
    pages = [p for p in pages if p[1] not in ("trends", "forecast")]
if not col_analysis["categorical"]:
    pages = [p for p in pages if p[1] != "categories"]
if len(col_analysis["numeric"]) < 2:
    pages = [p for p in pages if p[1] != "correlations"]

# If the stored active_view was removed (e.g. data has no dates), reset to overview
_valid_keys = {key for _, key in pages}
active_view_check = st.session_state.get("active_view", "overview")
if active_view_check not in _valid_keys:
    st.session_state["active_view"] = "overview"
    active_view_check = "overview"

# Split nav into rows of max 6 so labels never clip
_ROW_SIZE = 6
_nav_rows = [pages[i:i+_ROW_SIZE] for i in range(0, len(pages), _ROW_SIZE)]
for _nav_row in _nav_rows:
    _nav_cols = st.columns(len(_nav_row))
    for (label, key), col in zip(_nav_row, _nav_cols):
        with col:
            if active_view_check == key:
                st.markdown(
                    f'<div style="background:#f0eeff;border-bottom:2px solid #5b4bff;'
                    f'border-radius:8px 8px 0 0;padding:0.42rem 0.3rem;text-align:center;'
                    f'font-size:11.5px;font-weight:700;color:#5b4bff;white-space:nowrap;'
                    f'overflow:hidden;text-overflow:ellipsis">{label}</div>',
                    unsafe_allow_html=True)
            else:
                if st.button(label, key=f"pg_{key}", use_container_width=True):
                    st.session_state["active_view"] = key
                    st.rerun()

st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:2px 0 12px'>", unsafe_allow_html=True)

active_view = st.session_state.get("active_view", "overview")

df_info = (f"Dataset: {_stored['name']} | Type: {dtype} | {df.shape[0]} rows x {df.shape[1]} cols\n"
           f"Columns: {', '.join(df.columns.tolist())}\n"
           f"Numeric columns: {', '.join(col_analysis['numeric'])}\n"
           f"Categorical columns: {', '.join(col_analysis['categorical'])}\n"
           f"Date columns: {', '.join(col_analysis['date'])}\n"
           f"Statistics:\n{df.describe().to_string()}\n"
           f"Sample data:\n{df.head(5).to_string()}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if active_view == "overview":
    # Top KPIs
    if col_analysis["numeric"]:
        accent = [cfg["color"],"#10b981","#f59e0b","#8b5cf6"]
        kcols  = st.columns(min(len(col_analysis["numeric"]), 4))
        for i, nc in enumerate(col_analysis["numeric"][:4]):
            s = df[nc].dropna()
            with kcols[i]:
                st.markdown(
                    f'<div class="kpi-card">'
                    f'<div class="kpi-bar" style="background:{accent[i%4]}"></div>'
                    f'<div class="kpi-label">{nc.replace("_"," ").title()}</div>'
                    f'<div class="kpi-value">{fmt_plain(s.sum())}</div>'
                    f'<div class="kpi-sub">Avg {fmt_plain(s.mean())} · Max {fmt_plain(s.max())}</div>'
                    f'</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    oc1, oc2 = st.columns([3, 2])
    with oc1:
        st.markdown('<div class="sec-head">Trend Overview — How Your Key Metrics Move</div>', unsafe_allow_html=True)
        try:
            if col_analysis["date"] and col_analysis["numeric"]:
                st.plotly_chart(plot_trend(df, col_analysis), use_container_width=True)
            elif col_analysis["categorical"] and col_analysis["numeric"]:
                cat = col_analysis["categorical"][0]; num = col_analysis["numeric"][0]
                st.plotly_chart(plot_bar(df, cat, num, horizontal=df[cat].nunique()>6), use_container_width=True)
            else:
                st.info("Upload data with numeric columns to see charts.")
        except Exception as e:
            st.error(f"Chart error: {e}")

    with oc2:
        st.markdown('<div class="sec-head">Breakdown — What Drives the Numbers</div>', unsafe_allow_html=True)
        try:
            if col_analysis["categorical"] and col_analysis["numeric"]:
                cat = col_analysis["categorical"][0]; num = col_analysis["numeric"][0]
                st.plotly_chart(plot_pie(df, cat, num), use_container_width=True)
        except Exception as e:
            st.error(f"Chart error: {e}")

    # AI Summary
    st.markdown('<div class="sec-head">Executive Summary — Highlights, Risks & Actions</div>', unsafe_allow_html=True)
    if "ai_exec" not in st.session_state:
        st.markdown('<div class="ai-box-title" style="font-size:11px;font-weight:700;color:#5b4bff;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">🤖 Executive Summary</div>', unsafe_allow_html=True)
        st.session_state["ai_exec"] = st.write_stream(get_ai_analysis(df_info, dtype, "exec"))
    else:
        st.markdown(f'<div class="ai-box"><div class="ai-box-title">🤖 {cfg["label"]} — Executive Summary</div>'
                    f'<div class="ai-box-text">{st.session_state["ai_exec"].replace(chr(10),"<br>")}</div></div>',
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SMART INSIGHTS (dataset-specific)
# ══════════════════════════════════════════════════════════════════════════════
elif active_view == "smart":
    st.markdown(f'<div class="sec-head">{cfg["icon"]} Smart Insights — {cfg["label"]}</div>', unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:#64748b;font-size:13px;margin-bottom:1.5rem;"
        f"white-space:normal;overflow:visible;display:block;width:100%;max-width:100%'>"
        f"Generate dataset-specific insight cards based on actual column names and values — "
        f"tailored to your <b style='color:#334155'>{dtype}</b> dataset.</p>",
        unsafe_allow_html=True)

    if smart_insights:
        for ins in smart_insights:
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(
                    f'<div class="insight-card">'
                    f'<span class="insight-tag" style="background:rgba(255,255,255,0.12);color:#e2e8f0">{ins["tag"]}</span>'
                    f'<div class="insight-title">{ins["title"]}</div>'
                    f'<div class="insight-text">{ins["text"]}</div>'
                    f'</div>', unsafe_allow_html=True)
            with c2:
                st.markdown(
                    f'<div class="kpi-card" style="text-align:center;height:100%">'
                    f'<div class="kpi-bar" style="background:{ins["color"]}"></div>'
                    f'<div style="font-size:28px;font-weight:800;color:{ins["color"]};margin-top:12px">{ins["number"]}</div>'
                    f'<div class="kpi-sub">{ins["tag"]}</div>'
                    f'</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

    # Second row: charts that match the dataset type
    if col_analysis["categorical"] and col_analysis["numeric"]:
        st.markdown('<div class="sec-head">Segment Analysis — Performance by Every Category</div>', unsafe_allow_html=True)
        cats = col_analysis["categorical"][:4]
        num  = col_analysis["numeric"][0]
        for i in range(0, len(cats), 2):
            batch = cats[i:i+2]
            cols  = st.columns(len(batch))
            for j, cat in enumerate(batch):
                with cols[j]:
                    st.markdown(f"**{cat.replace('_',' ').title()} by {num.replace('_',' ').title()}**")
                    try:
                        n = df[cat].nunique()
                        st.plotly_chart(plot_bar(df, cat, num, horizontal=n>5), use_container_width=True)
                    except Exception as e:
                        st.error(str(e))

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: KPI REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif active_view == "kpis":
    st.markdown('<div class="sec-head">KPI Scorecard — Full Stats for Every Metric</div>', unsafe_allow_html=True)
    if not col_analysis["numeric"]:
        st.info("No numeric columns detected.")
    else:
        for batch_start in range(0, len(col_analysis["numeric"]), 4):
            batch = col_analysis["numeric"][batch_start:batch_start+4]
            cols  = st.columns(len(batch))
            for j, nc in enumerate(batch):
                s = df[nc].dropna()
                with cols[j]:
                    st.markdown(
                        f'<div class="kpi-card">'
                        f'<div class="kpi-bar" style="background:{COLORS[j%len(COLORS)]}"></div>'
                        f'<div class="kpi-label">{nc.replace("_"," ").title()}</div>'
                        f'<div class="kpi-value" style="font-size:22px">{fmt_plain(s.sum())}</div>'
                        f'<div class="kpi-sub" style="margin-top:10px">'
                        f'Avg: <b>{fmt_plain(s.mean())}</b><br>'
                        f'Median: <b>{fmt_plain(s.median())}</b><br>'
                        f'Min: <b>{fmt_plain(s.min())}</b> · Max: <b>{fmt_plain(s.max())}</b><br>'
                        f'Std Dev: <b>{fmt_plain(s.std())}</b> · Missing: <b>{int(df[nc].isna().sum())}</b>'
                        f'</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="sec-head">Value Distributions — Spread, Skew & Frequency</div>', unsafe_allow_html=True)
        for batch_start in range(0, len(col_analysis["numeric"]), 3):
            batch = col_analysis["numeric"][batch_start:batch_start+3]
            cols  = st.columns(len(batch))
            for j, nc in enumerate(batch):
                with cols[j]:
                    st.markdown(f"**{nc.replace('_',' ').title()}**")
                    try: st.plotly_chart(plot_histogram(df, nc), use_container_width=True)
                    except Exception as e: st.error(str(e))

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TRENDS
# ══════════════════════════════════════════════════════════════════════════════
elif active_view == "trends":
    st.markdown('<div class="sec-head">Time Series Analysis</div>', unsafe_allow_html=True)
    if not col_analysis["date"] or not col_analysis["numeric"]:
        st.info("Trend analysis requires at least one date and one numeric column.")
    else:
        if time_insights:
            ti = time_insights
            nc_label = ti["num_col"].replace("_"," ").title()
            t1,t2,t3,t4 = st.columns(4)
            tup = "▲" if ti["growth"]>0 else "▼"
            tcls = "kpi-up" if ti["growth"]>0 else "kpi-dn"
            with t1:
                st.markdown(f'<div class="kpi-card"><div class="kpi-bar" style="background:{cfg["color"]}"></div><div class="kpi-label">Overall Growth</div><div class="kpi-value"><span class="{tcls}">{tup} {ti["growth"]:+.1f}%</span></div><div class="kpi-sub">First to last period</div></div>', unsafe_allow_html=True)
            with t2:
                st.markdown(f'<div class="kpi-card"><div class="kpi-bar" style="background:#10b981"></div><div class="kpi-label">Best Period</div><div class="kpi-value" style="font-size:18px">{ti["best_period"]}</div><div class="kpi-sub">{fmt_plain(ti["best_val"])} {nc_label}</div></div>', unsafe_allow_html=True)
            with t3:
                st.markdown(f'<div class="kpi-card"><div class="kpi-bar" style="background:#ef4444"></div><div class="kpi-label">Worst Period</div><div class="kpi-value" style="font-size:18px">{ti["worst_period"]}</div><div class="kpi-sub">{fmt_plain(ti["worst_val"])} {nc_label}</div></div>', unsafe_allow_html=True)
            with t4:
                st.markdown(f'<div class="kpi-card"><div class="kpi-bar" style="background:#f59e0b"></div><div class="kpi-label">Periods</div><div class="kpi-value">{ti["periods"]}</div><div class="kpi-sub">Monthly periods</div></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        try: st.plotly_chart(plot_trend(df, col_analysis), use_container_width=True)
        except Exception as e: st.error(f"Error: {e}")

        if col_analysis["categorical"]:
            st.markdown('<div class="sec-head">Heatmap — Monthly Performance by Segment</div>', unsafe_allow_html=True)
            try:
                _hm = plot_heatmap(df, col_analysis)
                if _hm: st.plotly_chart(_hm, use_container_width=True)
            except Exception as e: st.error(f"Heatmap error: {e}")

        if "ai_trends" not in st.session_state:
            st.markdown('<div class="ai-box-title" style="font-size:11px;font-weight:700;color:#5b4bff;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">🤖 Trend Analysis</div>', unsafe_allow_html=True)
            st.session_state["ai_trends"] = st.write_stream(get_ai_analysis(df_info, dtype, "trends"))
        else:
            st.markdown(f'<div class="ai-box"><div class="ai-box-title">🤖 Trend Analysis</div>'
                        f'<div class="ai-box-text">{st.session_state["ai_trends"].replace(chr(10),"<br>")}</div></div>',
                        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CATEGORIES
# ══════════════════════════════════════════════════════════════════════════════
elif active_view == "categories":
    st.markdown('<div class="sec-head">Segment Performance — Winners, Laggards & Risk</div>', unsafe_allow_html=True)
    if not col_analysis["categorical"] or not col_analysis["numeric"]:
        st.info("Category analysis requires categorical and numeric columns.")
    else:
        for cat in col_analysis["categorical"][:4]:
            num_col = col_analysis["numeric"][0]
            grp = df.groupby(cat)[num_col].sum().sort_values(ascending=False)
            total = grp.sum()
            top3_share = grp.head(3).sum() / total * 100 if total > 0 else 0
            risk = "High" if top3_share>70 else "Medium" if top3_share>50 else "Low"
            badge = "badge-red" if risk=="High" else "badge-amber" if risk=="Medium" else "badge-green"

            st.markdown(f'<div class="sec-head">{cat.replace("_"," ").title()} — {df[cat].nunique()} unique values</div>', unsafe_allow_html=True)
            cc1,cc2,cc3,cc4 = st.columns(4)
            with cc1:
                st.markdown(f'<div class="kpi-card"><div class="kpi-bar" style="background:#10b981"></div><div class="kpi-label">Top Performer</div><div class="kpi-value" style="font-size:16px">{grp.index[0]}</div><div class="kpi-sub">{fmt_plain(grp.iloc[0])}</div></div>', unsafe_allow_html=True)
            with cc2:
                st.markdown(f'<div class="kpi-card"><div class="kpi-bar" style="background:#ef4444"></div><div class="kpi-label">Lowest</div><div class="kpi-value" style="font-size:16px">{grp.index[-1]}</div><div class="kpi-sub">{fmt_plain(grp.iloc[-1])}</div></div>', unsafe_allow_html=True)
            with cc3:
                st.markdown(f'<div class="kpi-card"><div class="kpi-bar" style="background:#f59e0b"></div><div class="kpi-label">Top 3 Share</div><div class="kpi-value">{top3_share:.1f}%</div><div class="kpi-sub">of total</div></div>', unsafe_allow_html=True)
            with cc4:
                st.markdown(f'<div class="kpi-card"><div class="kpi-bar" style="background:#8b5cf6"></div><div class="kpi-label">Concentration Risk</div><div class="kpi-value" style="font-size:18px"><span class="badge {badge}">{risk}</span></div></div>', unsafe_allow_html=True)

            bc1, bc2 = st.columns([3,2])
            with bc1:
                try:
                    n = df[cat].nunique()
                    st.plotly_chart(plot_bar(df, cat, num_col, horizontal=n>6), use_container_width=True)
                except Exception as e: st.error(str(e))
            with bc2:
                try: st.plotly_chart(plot_pie(df, cat, num_col), use_container_width=True)
                except Exception as e: st.error(str(e))

        if "ai_cat" not in st.session_state:
            st.markdown('<div class="ai-box-title" style="font-size:11px;font-weight:700;color:#5b4bff;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">🤖 Category Analysis</div>', unsafe_allow_html=True)
            st.session_state["ai_cat"] = st.write_stream(get_ai_analysis(df_info, dtype, "categories"))
        else:
            st.markdown(f'<div class="ai-box"><div class="ai-box-title">🤖 Category Analysis</div>'
                        f'<div class="ai-box-text">{st.session_state["ai_cat"].replace(chr(10),"<br>")}</div></div>',
                        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CORRELATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif active_view == "correlations":
    st.markdown('<div class="sec-head">Correlation Analysis — Which Metrics Move Together</div>', unsafe_allow_html=True)
    if len(col_analysis["numeric"]) < 2:
        st.info("Correlation analysis requires at least 2 numeric columns.")
    else:
        if correlations:
            for pair in correlations[:6]:
                r = pair["r"]
                strength = "Strong" if abs(r)>0.7 else "Moderate" if abs(r)>0.4 else "Weak"
                direction = "Positive" if r>0 else "Negative"
                badge_cls = "badge-green" if abs(r)>0.7 else "badge-amber" if abs(r)>0.4 else "badge-red"
                bar_c = "#10b981" if r>0 else "#ef4444"
                st.markdown(
                    f'<div class="kpi-card" style="margin-bottom:8px">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center">'
                    f'<div><b style="color:#0d1f3c">{pair["col1"].replace("_"," ").title()}</b>'
                    f' <span style="color:#94a3b8">↔</span> '
                    f'<b style="color:#0d1f3c">{pair["col2"].replace("_"," ").title()}</b></div>'
                    f'<div><span class="badge {badge_cls}">{strength} {direction}</span>'
                    f' <b style="color:#0d1f3c;font-size:16px;margin-left:10px">r = {r}</b></div>'
                    f'</div><div style="margin-top:10px;background:#f1f5f9;border-radius:4px;height:5px">'
                    f'<div style="width:{int(abs(r)*100)}%;background:{bar_c};height:5px;border-radius:4px"></div>'
                    f'</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="sec-head">Scatter Plots — Visualising Metric Relationships</div>', unsafe_allow_html=True)
        nums = col_analysis["numeric"]
        pairs_to_plot = [(nums[i],nums[j]) for i in range(min(3,len(nums))) for j in range(i+1,min(4,len(nums)))]
        if pairs_to_plot:
            sc_cols = st.columns(min(len(pairs_to_plot), 3))
            for idx, (c1, c2) in enumerate(pairs_to_plot[:3]):
                with sc_cols[idx]:
                    st.markdown(f"**{c1.replace('_',' ').title()} vs {c2.replace('_',' ').title()}**")
                    try: st.plotly_chart(plot_scatter(df, c1, c2), use_container_width=True)
                    except Exception as e: st.error(str(e))

        st.markdown('<div class="sec-head">Full Correlation Matrix — Every Pair at a Glance</div>', unsafe_allow_html=True)
        corr_df = df[col_analysis["numeric"]].corr().round(3)
        def color_corr(val):
            try:
                v = float(val)
                if v > 0.7:  return "background-color:#bbf7d0;color:#14532d"
                if v > 0.4:  return "background-color:#d1fae5;color:#166534"
                if v < -0.7: return "background-color:#fecaca;color:#7f1d1d"
                if v < -0.4: return "background-color:#fee2e2;color:#991b1b"
                return "background-color:#f8fafc;color:#475569"
            except: return ""
        try:
            st.dataframe(corr_df.style.map(color_corr), use_container_width=True)
        except AttributeError:
            st.dataframe(corr_df.style.applymap(color_corr), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANOMALIES
# ══════════════════════════════════════════════════════════════════════════════
elif active_view == "anomalies":
    st.markdown('<div class="sec-head">Data Health Report — Quality, Gaps & Anomalies</div>', unsafe_allow_html=True)
    total_missing  = int(df.isnull().sum().sum())
    total_outliers = sum(s.get("outliers",0) for s in adv_stats.values())
    dupes = int(df.duplicated().sum())
    completeness = round((1 - df.isnull().sum().sum() / (df.shape[0]*df.shape[1])) * 100, 1)

    mc1,mc2,mc3,mc4 = st.columns(4)
    with mc1: st.markdown(f'<div class="kpi-card"><div class="kpi-bar" style="background:#2563eb"></div><div class="kpi-label">Data Completeness</div><div class="kpi-value">{completeness}%</div></div>', unsafe_allow_html=True)
    with mc2: st.markdown(f'<div class="kpi-card"><div class="kpi-bar" style="background:{"#ef4444" if total_missing>0 else "#10b981"}"></div><div class="kpi-label">Missing Values</div><div class="kpi-value">{total_missing:,}</div><div class="kpi-sub">Across all columns</div></div>', unsafe_allow_html=True)
    with mc3: st.markdown(f'<div class="kpi-card"><div class="kpi-bar" style="background:#f59e0b"></div><div class="kpi-label">Outliers (IQR)</div><div class="kpi-value">{total_outliers:,}</div></div>', unsafe_allow_html=True)
    with mc4: st.markdown(f'<div class="kpi-card"><div class="kpi-bar" style="background:{"#ef4444" if dupes>0 else "#10b981"}"></div><div class="kpi-label">Duplicate Rows</div><div class="kpi-value">{dupes:,}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-head">Column Summary — Types, Completeness & Uniqueness</div>', unsafe_allow_html=True)
    qual_data = []
    for col in df.columns:
        miss = int(df[col].isna().sum())
        miss_p = round(miss/len(df)*100,1)
        qual_data.append({
            "Column": col, "Type": str(df[col].dtype),
            "Missing": miss, "Missing %": f"{miss_p}%",
            "Unique": int(df[col].nunique()),
            "Outliers": adv_stats.get(col,{}).get("outliers",0),
            "Status": "⚠️ Check" if miss_p>10 or adv_stats.get(col,{}).get("outliers",0)>5 else "✅ OK"
        })
    st.dataframe(pd.DataFrame(qual_data), use_container_width=True, hide_index=True)

    st.markdown('<div class="sec-head">Outliers — Values Outside the Expected Range</div>', unsafe_allow_html=True)
    has_outliers = False
    for nc, s in adv_stats.items():
        if s["outliers"] > 0:
            has_outliers = True
            st.markdown(
                f'<div class="insight-card"><span class="insight-tag" style="background:rgba(239,68,68,0.2);color:#fca5a5">⚠️ Outliers Detected</span>'
                f'<div class="insight-title">{nc.replace("_"," ").title()}</div>'
                f'<div class="insight-text">{s["outliers"]} outlier(s) · IQR: [{fmt_plain(s["q1"])} — {fmt_plain(s["q3"])}] · Range: [{fmt_plain(s["min"])} — {fmt_plain(s["max"])}]</div>'
                f'</div>', unsafe_allow_html=True)
    if not has_outliers:
        st.success("✅ No significant outliers detected in numeric columns.")

    if "ai_anomaly" not in st.session_state:
        st.markdown('<div class="ai-box-title" style="font-size:11px;font-weight:700;color:#5b4bff;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">🤖 AI Data Quality Analysis</div>', unsafe_allow_html=True)
        st.session_state["ai_anomaly"] = st.write_stream(get_ai_analysis(df_info, dtype, "anomalies"))
    else:
        st.markdown(f'<div class="ai-box"><div class="ai-box-title">🤖 AI Data Quality Analysis</div>'
                    f'<div class="ai-box-text">{st.session_state["ai_anomaly"].replace(chr(10),"<br>")}</div></div>',
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AI ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif active_view == "ai":
    st.markdown(f'<div class="sec-head">🤖 AI Analysis — {cfg["label"]}</div>', unsafe_allow_html=True)
    st.markdown(f"<p style='color:#64748b;font-size:13px;margin-bottom:1.5rem'>Deep AI analysis tailored specifically for your {dtype} dataset. All insights use actual numbers from your data.</p>", unsafe_allow_html=True)

    ai_sections = [
        ("ai_exec",   "exec",       "🏢 Executive Summary",  "Key findings, risks and recommendations"),
        ("ai_trends", "trends",     "📈 Trend Analysis",      "Time patterns, growth rates, forecasting"),
        ("ai_cat",    "categories", "📦 Category Insights",   "Segment performance and quick wins"),
        ("ai_anomaly","anomalies",  "🔍 Data Quality Report", "Outliers, anomalies and data issues"),
    ]
    for key, page_key, title, desc in ai_sections:
        with st.expander(f"{title} — {desc}", expanded=(key=="ai_exec")):
            if key not in st.session_state:
                st.session_state[key] = st.write_stream(get_ai_analysis(df_info, dtype, page_key))
            else:
                st.markdown(
                    f'<div class="ai-box"><div class="ai-box-title">{title}</div>'
                    f'<div class="ai-box-text">{st.session_state[key].replace(chr(10),"<br>")}</div></div>',
                    unsafe_allow_html=True)
            if st.button(f"↻ Regenerate", key=f"regen_{key}"):
                del st.session_state[key]; st.rerun()

    st.markdown('<div class="sec-head">Data Chat — Ask a Question, Get an Answer</div>', unsafe_allow_html=True)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-ai">🤖 {msg["content"].replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
    with st.form("chat", clear_on_submit=True):
        qc1, qc2 = st.columns([5,1])
        with qc1:
            q = st.text_input("", placeholder=f"Ask anything about your {dtype} data...", label_visibility="collapsed")
        with qc2:
            sub = st.form_submit_button("Ask →", use_container_width=True)
    if sub and q:
        st.session_state.messages.append({"role":"user","content":q})
        answer = st.write_stream(ask_claude(q, df_info, dtype))
        st.session_state.messages.append({"role":"assistant","content":answer})
        st.rerun()
    if st.session_state.get("messages"):
        if st.button("Clear chat"):
            st.session_state.messages = []; st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA TABLE
# ══════════════════════════════════════════════════════════════════════════════
elif active_view == "table":
    st.markdown('<div class="sec-head">Data Table</div>', unsafe_allow_html=True)
    search = st.text_input("", placeholder="🔍 Search / filter rows...", label_visibility="collapsed")
    display_df = df
    if search:
        _safe = re.escape(search)
        mask = df.apply(lambda row: row.astype(str).str.contains(_safe, case=False, na=False, regex=True).any(), axis=1)
        display_df = df[mask]
        st.caption(f"{len(display_df):,} rows matching '{search}'")
    st.dataframe(display_df, use_container_width=True, height=520)
    dl1, dl2 = st.columns(2)
    with dl1:
        csv2 = display_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download filtered CSV", csv2, f"{file_title}_filtered.csv", "text/csv", use_container_width=True)
    with dl2:
        st.caption(f"{len(display_df):,} of {len(df):,} rows shown")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA AGENT
# ══════════════════════════════════════════════════════════════════════════════
elif active_view == "agent":
    st.markdown(f'''
    <div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);border-radius:16px;
    padding:1.5rem 2rem;margin-bottom:1.5rem;border:1px solid #1e293b">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">
        <span style="font-size:28px">🤖</span>
        <div>
          <div style="font-size:16px;font-weight:800;color:#f1f5f9">Data Agent</div>
          <div style="font-size:12px;color:#64748b">Ask questions in plain English — get back data, tables and charts</div>
        </div>
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:12px">
        <span style="background:rgba(255,255,255,0.08);border-radius:20px;padding:3px 12px;font-size:11px;color:#94a3b8">🔍 Filter data</span>
        <span style="background:rgba(255,255,255,0.08);border-radius:20px;padding:3px 12px;font-size:11px;color:#94a3b8">📊 Aggregate & group</span>
        <span style="background:rgba(255,255,255,0.08);border-radius:20px;padding:3px 12px;font-size:11px;color:#94a3b8">📈 Show charts</span>
        <span style="background:rgba(255,255,255,0.08);border-radius:20px;padding:3px 12px;font-size:11px;color:#94a3b8">🔢 Calculate stats</span>
        <span style="background:rgba(255,255,255,0.08);border-radius:20px;padding:3px 12px;font-size:11px;color:#94a3b8">📋 Export results</span>
      </div>
    </div>
    ''', unsafe_allow_html=True)

    # ── DYNAMIC quick queries based on ACTUAL columns in uploaded file ──────────
    def build_quick_queries(df, col_analysis, dtype):
        nums = col_analysis["numeric"]
        cats = col_analysis["categorical"]
        dates = col_analysis["date"]
        qs = []

        # Top N by first numeric
        if nums and cats:
            qs.append(f"Show top 10 rows by {nums[0]}")
            qs.append(f"Group by {cats[0]} and sum {nums[0]}")
            qs.append(f"Which {cats[0]} has the highest {nums[0]}?")
            qs.append(f"Which {cats[0]} has the lowest {nums[0]}?")

        # Filter by numeric threshold
        if nums:
            avg_val = df[nums[0]].mean()
            qs.append(f"Filter rows where {nums[0]} > {avg_val:.0f}")
            qs.append(f"Show top 5 rows sorted by {nums[0]} descending")
            qs.append(f"What is the average {nums[0]} by {cats[0]}?" if cats else f"Show summary stats for {nums[0]}")

        # Date-based queries
        if dates and nums:
            qs.append(f"Show {nums[0]} trend over time")
            qs.append(f"Which month has the highest {nums[0]}?")

        # Second numeric column
        if len(nums) >= 2:
            qs.append(f"Compare {nums[0]} vs {nums[1]} by {cats[0]}" if cats else f"Show {nums[0]} vs {nums[1]}")
            qs.append(f"Show correlation between {nums[0]} and {nums[1]}")

        # Second categorical
        if len(cats) >= 2 and nums:
            qs.append(f"Group by {cats[1]} and sum {nums[0]}")

        # Status/categorical filter
        for cat in cats:
            unique_vals = df[cat].dropna().unique()
            if len(unique_vals) <= 6:
                qs.append(f"Show rows where {cat} = {unique_vals[0]}")
                break

        # Always include these
        qs.append("Show summary statistics for all columns")
        qs.append("Find rows with missing values")
        qs.append(f"Show first 20 rows")

        return qs[:12]  # max 12 queries

    qs = build_quick_queries(df, col_analysis, dtype)

    # Quick question buttons - clicking sets pending and reruns
    st.markdown('<div class="sec-head">Quick Queries — Tailored to Your Data</div>', unsafe_allow_html=True)
    btn_cols = st.columns(3)
    for i, q_text in enumerate(qs):
        with btn_cols[i % 3]:
            if st.button(q_text, key=f"quick_{i}", use_container_width=True):
                if "agent_messages" not in st.session_state:
                    st.session_state.agent_messages = []
                st.session_state["agent_auto_q"] = q_text
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Chat area
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []

    # Display chat history
    for msg in st.session_state.agent_messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-ai">🤖 {msg["content"].replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
            if "dataframe" in msg:
                st.dataframe(msg["dataframe"], use_container_width=True)
                csv_out = msg["dataframe"].to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Download this result", csv_out,
                                   "query_result.csv", "text/csv",
                                   key=f"dl_{msg.get('id',0)}")
            if "chart" in msg:
                st.plotly_chart(msg["chart"], use_container_width=True)
            if "code" in msg:
                with st.expander("📝 View generated code"):
                    st.code(msg["code"], language="python")

    # Input form — handle quick query auto-submit
    auto_q = st.session_state.pop("agent_auto_q", None)

    with st.form("agent_form", clear_on_submit=True):
        fc1, fc2 = st.columns([5, 1])
        with fc1:
            ph = f"e.g. Show top 10 rows by {col_analysis['numeric'][0] if col_analysis['numeric'] else 'value'}..."
            agent_q = st.text_input("", placeholder=ph, label_visibility="collapsed")
        with fc2:
            agent_sub = st.form_submit_button("Query →", use_container_width=True)

    # If quick button was clicked, use that as the query
    if auto_q and not agent_sub:
        agent_q  = auto_q
        agent_sub = True

    if agent_sub and agent_q:
        st.session_state.agent_messages.append({"role": "user", "content": agent_q})

        # Ask Claude to write pandas code to answer the question
        with st.spinner("Thinking..."):
            try:
                client = anthropic.Anthropic()
                system = f"""You are a data analyst and Python/pandas expert.
The user has a dataframe called `df` with these properties:
- Shape: {df.shape[0]} rows x {df.shape[1]} columns
- Columns: {df.columns.tolist()}
- Dtypes: {df.dtypes.to_dict()}
- Sample data:
{df.head(3).to_string()}

The user will ask a question about this data. You must:
1. Write Python/pandas code to answer it
2. Store the RESULT in a variable called `result`
3. `result` must be either:
   - A DataFrame (for tables/filtered data)
   - A dict with keys: "answer" (string), "dataframe" (optional DataFrame), "chart_type" (optional: "bar","line","pie","scatter"), "chart_x", "chart_y", "chart_color" (optional column names)
4. Also set `answer_text` = a 1-2 sentence plain English summary of the result

Rules:
- Use ONLY pandas operations on `df`
- Handle errors gracefully
- For "top N" queries, sort and head()
- For "filter" queries, use boolean indexing
- For "group by" queries, use groupby().agg()
- For "trend" or "over time" queries, suggest chart_type="line"
- For "compare" queries, suggest chart_type="bar"
- For "distribution" or "share", suggest chart_type="pie"
- NEVER import anything - only use df, pd, and np which are already available
- Output ONLY the Python code, no explanation, no markdown, no backticks"""

                response = client.messages.create(
                    model=MODEL,
                    max_tokens=1000,
                    system=system,
                    messages=[{"role": "user", "content": agent_q}]
                )

                code = response.content[0].text.strip()
                # Remove any accidental markdown fences
                if "```" in code:
                    m = re.search(r"```(?:python)?\n?(.*?)```", code, re.DOTALL)
                    if m:
                        code = m.group(1).strip()
                    else:
                        code = code.replace("```python", "").replace("```", "").strip()

                # Execute the code
                exec_globals = {"df": df.copy(), "pd": pd, "np": np,
                                "result": None, "answer_text": ""}
                try:
                    exec(code, exec_globals)
                    result = exec_globals.get("result")
                    answer_text = exec_globals.get("answer_text", "")

                    msg = {"role": "assistant", "id": len(st.session_state.agent_messages),
                           "content": answer_text or "Here are the results:", "code": code}

                    if isinstance(result, pd.DataFrame):
                        msg["dataframe"] = result
                        msg["content"] = (answer_text or
                                          f"Found {len(result):,} rows matching your query.")

                    elif isinstance(result, dict):
                        if "answer" in result:
                            msg["content"] = result["answer"]
                        if "dataframe" in result and isinstance(result["dataframe"], pd.DataFrame):
                            msg["dataframe"] = result["dataframe"]
                        # Build chart if requested
                        if "chart_type" in result and "dataframe" in result:
                            try:
                                cdf = result["dataframe"]
                                if len(cdf.columns) == 0:
                                    raise ValueError("Chart DataFrame has no columns")
                                cx  = result.get("chart_x", cdf.columns[0])
                                cy  = result.get("chart_y", cdf.columns[1] if len(cdf.columns)>1 else cdf.columns[0])
                                cc  = result.get("chart_color")
                                ct  = result["chart_type"]
                                if ct == "bar":
                                    fig = px.bar(cdf, x=cx, y=cy, color=cc,
                                                 color_discrete_sequence=COLORS)
                                elif ct == "line":
                                    fig = px.line(cdf, x=cx, y=cy, color=cc,
                                                  color_discrete_sequence=COLORS,
                                                  markers=True)
                                elif ct == "pie":
                                    fig = px.pie(cdf, names=cx, values=cy,
                                                 color_discrete_sequence=COLORS, hole=0.4)
                                elif ct == "scatter":
                                    fig = px.scatter(cdf, x=cx, y=cy, color=cc,
                                                     color_discrete_sequence=COLORS)
                                else:
                                    fig = None
                                if fig:
                                    fig.update_layout(**mk_layout(height=340, showlegend=True))
                                    msg["chart"] = fig
                            except Exception:
                                pass
                    elif result is not None:
                        msg["content"] = str(result)

                    st.session_state.agent_messages.append(msg)

                except Exception as exec_err:
                    # Fallback: ask Claude to just answer in plain text
                    fallback_text = ""
                    with client.messages.stream(
                        model=MODEL, max_tokens=400,
                        system="You are a data analyst. Dataset:\n" + df_info + "\nAnswer concisely with numbers.",
                        messages=[{"role": "user", "content": agent_q}]
                    ) as stream:
                        for chunk in stream.text_stream:
                            fallback_text += chunk
                    st.session_state.agent_messages.append({
                        "role": "assistant",
                        "content": fallback_text or "I couldn't compute that. Please try rephrasing.",
                        "id": len(st.session_state.agent_messages)
                    })

            except anthropic.AuthenticationError:
                st.session_state.agent_messages.append({
                    "role": "assistant", "id": 0,
                    "content": "Auth error — check your API key."
                })
            except Exception as e:
                st.session_state.agent_messages.append({
                    "role": "assistant", "id": 0,
                    "content": f"Error: {e}"
                })
        st.rerun()

    if st.session_state.get("agent_messages"):
        if st.button("🗑️ Clear conversation", use_container_width=False):
            st.session_state.agent_messages = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════
elif active_view == "clean":
    st.markdown('<div class="sec-head">🧹 Data Cleaning Panel — Fix Missing Values, Duplicates & Columns</div>', unsafe_allow_html=True)

    _work_df = df.copy()
    total_missing = int(_work_df.isnull().sum().sum())
    dupes = int(_work_df.duplicated().sum())
    completeness = round((1 - _work_df.isnull().sum().sum() / (_work_df.shape[0]*_work_df.shape[1])) * 100, 1)

    cl1, cl2, cl3, cl4 = st.columns(4)
    with cl1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-bar" style="background:#2563eb"></div><div class="kpi-label">Completeness</div><div class="kpi-value">{completeness}%</div></div>', unsafe_allow_html=True)
    with cl2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-bar" style="background:{"#ef4444" if total_missing>0 else "#10b981"}"></div><div class="kpi-label">Missing Values</div><div class="kpi-value">{total_missing:,}</div><div class="kpi-sub">Across all columns</div></div>', unsafe_allow_html=True)
    with cl3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-bar" style="background:{"#ef4444" if dupes>0 else "#10b981"}"></div><div class="kpi-label">Duplicate Rows</div><div class="kpi-value">{dupes:,}</div></div>', unsafe_allow_html=True)
    with cl4:
        is_modified = st.session_state.get("df_modified_for") == _stored["name"]
        st.markdown(f'<div class="kpi-card"><div class="kpi-bar" style="background:#8b5cf6"></div><div class="kpi-label">Dataset Status</div><div class="kpi-value" style="font-size:16px">{"✏ Modified" if is_modified else "✅ Original"}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab_missing, tab_dupes, tab_cols = st.tabs(["📭 Missing Values", "🔄 Duplicates", "🗂️ Drop Columns"])

    with tab_missing:
        cols_with_missing = [c for c in _work_df.columns if _work_df[c].isna().sum() > 0]
        if not cols_with_missing:
            st.success("✅ No missing values found in your dataset!")
        else:
            st.markdown(f"<p style='color:#64748b;font-size:13px'>{len(cols_with_missing)} column(s) have missing values. Choose how to handle each.</p>", unsafe_allow_html=True)
            fill_actions = {}
            for col in cols_with_missing:
                miss_count = int(_work_df[col].isna().sum())
                miss_pct = round(miss_count / len(_work_df) * 100, 1)
                is_num = pd.api.types.is_numeric_dtype(_work_df[col])
                mc1, mc2, mc3 = st.columns([2, 1.5, 1.5])
                with mc1:
                    st.markdown(
                        f'<div style="padding:8px 0">'
                        f'<div style="font-size:13px;font-weight:600;color:#0d1f3c">{col}</div>'
                        f'<div style="font-size:11px;color:#94a3b8">{miss_count:,} missing ({miss_pct}%)</div>'
                        f'</div>', unsafe_allow_html=True)
                with mc2:
                    if is_num:
                        opts = ["Leave as-is", "Fill with Mean", "Fill with Median", "Fill with Zero", "Drop rows"]
                    else:
                        opts = ["Leave as-is", "Fill with Mode", "Fill with 'Unknown'", "Drop rows"]
                    action = st.selectbox("", opts, key=f"fill_{col}", label_visibility="collapsed")
                    fill_actions[col] = action
                with mc3:
                    if is_num and "Mean" in action:
                        st.caption(f"→ will fill with {fmt_plain(_work_df[col].mean())}")
                    elif is_num and "Median" in action:
                        st.caption(f"→ will fill with {fmt_plain(_work_df[col].median())}")
                    elif is_num and "Zero" in action:
                        st.caption("→ will fill with 0")
                    elif "Mode" in action:
                        mode_v = _work_df[col].mode()
                        st.caption(f"→ will fill with '{mode_v[0] if len(mode_v) else '?'}'")
                    elif "Unknown" in action:
                        st.caption("→ will fill with 'Unknown'")
                    elif "Drop" in action:
                        st.caption(f"→ will drop {miss_count} rows")

            if st.button("✅ Apply Missing Value Fixes", type="primary", key="apply_missing"):
                new_df = _work_df.copy()
                for col, action in fill_actions.items():
                    if action == "Fill with Mean":
                        new_df[col] = new_df[col].fillna(new_df[col].mean())
                    elif action == "Fill with Median":
                        new_df[col] = new_df[col].fillna(new_df[col].median())
                    elif action == "Fill with Zero":
                        new_df[col] = new_df[col].fillna(0)
                    elif action == "Fill with Mode":
                        mv = new_df[col].mode()
                        if len(mv): new_df[col] = new_df[col].fillna(mv[0])
                    elif action == "Fill with 'Unknown'":
                        new_df[col] = new_df[col].fillna("Unknown")
                    elif action == "Drop rows":
                        new_df = new_df.dropna(subset=[col])
                st.session_state["df_modified"] = new_df.reset_index(drop=True)
                st.session_state["df_modified_for"] = _stored["name"]
                for k in list(st.session_state.keys()):
                    if k.startswith("ai_"): del st.session_state[k]
                st.success(f"✅ Applied! Dataset now has {len(new_df):,} rows and {new_df.isnull().sum().sum()} missing values.")
                st.rerun()

    with tab_dupes:
        if dupes == 0:
            st.success("✅ No duplicate rows found in your dataset!")
        else:
            st.warning(f"⚠️ Found **{dupes:,}** duplicate row(s) — {round(dupes/len(_work_df)*100,1)}% of your data.")
            with st.expander("Preview duplicate rows (first 20)"):
                st.dataframe(_work_df[_work_df.duplicated(keep=False)].head(20), use_container_width=True)
            if st.button(f"🗑️ Remove All {dupes:,} Duplicate Rows", type="primary", key="remove_dupes"):
                new_df = _work_df.drop_duplicates().reset_index(drop=True)
                st.session_state["df_modified"] = new_df
                st.session_state["df_modified_for"] = _stored["name"]
                for k in list(st.session_state.keys()):
                    if k.startswith("ai_"): del st.session_state[k]
                st.success(f"✅ Removed {dupes:,} duplicates. {len(new_df):,} unique rows remain.")
                st.rerun()

    with tab_cols:
        st.markdown("<p style='color:#64748b;font-size:13px'>Select columns to permanently remove from the working dataset.</p>", unsafe_allow_html=True)
        cols_to_drop = st.multiselect("Columns to drop", options=_work_df.columns.tolist(), key="cols_to_drop")
        if cols_to_drop:
            st.warning(f"Will remove: **{', '.join(cols_to_drop)}**  ({len(_work_df.columns) - len(cols_to_drop)} columns will remain)")
            if st.button(f"🗑️ Drop {len(cols_to_drop)} Column(s)", type="primary", key="drop_cols"):
                new_df = _work_df.drop(columns=cols_to_drop)
                st.session_state["df_modified"] = new_df
                st.session_state["df_modified_for"] = _stored["name"]
                for k in list(st.session_state.keys()):
                    if k.startswith("ai_"): del st.session_state[k]
                st.success(f"✅ Dropped {len(cols_to_drop)} column(s). {len(new_df.columns)} columns remain.")
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.session_state.get("df_modified_for") == _stored["name"]:
        st.markdown('<div class="sec-head">Reset to Original</div>', unsafe_allow_html=True)
        st.markdown("<p style='color:#64748b;font-size:13px'>Undo all cleaning changes and restore the original uploaded data.</p>", unsafe_allow_html=True)
        if st.button("↺ Reset to Original Data", key="reset_clean"):
            st.session_state["df_modified"] = None
            st.session_state["df_modified_for"] = None
            for k in list(st.session_state.keys()):
                if k.startswith("ai_"): del st.session_state[k]
            st.success("✅ Restored to original data.")
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
elif active_view == "forecast":
    st.markdown('<div class="sec-head">🔮 Forecasting — Predict Future Trends</div>', unsafe_allow_html=True)
    if not col_analysis["date"] or not col_analysis["numeric"]:
        st.info("Forecasting requires at least one date column and one numeric column.")
    else:
        fc1, fc2, fc3 = st.columns([2, 1, 1])
        with fc1:
            fc_num = st.selectbox("Metric to forecast", col_analysis["numeric"], key="fc_num")
        with fc2:
            fc_periods = st.selectbox("Forecast periods", [7, 14, 30, 60, 90], index=2, key="fc_periods")
        with fc3:
            fc_unit = st.selectbox("Period unit", ["days", "months"], key="fc_unit")

        try:
            date_col = col_analysis["date"][0]
            df_fc = df[[date_col, fc_num]].copy().dropna()
            df_fc[date_col] = pd.to_datetime(df_fc[date_col], errors="coerce")
            df_fc = df_fc.dropna(subset=[date_col])

            if fc_unit == "months":
                df_fc["_period"] = df_fc[date_col].dt.to_period("M").dt.to_timestamp()
            else:
                df_fc["_period"] = df_fc[date_col].dt.normalize()

            agg = df_fc.groupby("_period")[fc_num].sum().reset_index().sort_values("_period")

            if len(agg) < 3:
                st.warning("Need at least 3 data points to forecast. Try switching to 'months' unit or upload more data.")
            else:
                x = np.arange(len(agg), dtype=float)
                y = agg[fc_num].values.astype(float)
                coeffs = np.polyfit(x, y, 1)
                trend_line = np.polyval(coeffs, x)
                std_resid = np.std(y - trend_line)

                last_date = agg["_period"].iloc[-1]
                if fc_unit == "months":
                    future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(fc_periods)]
                else:
                    future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(fc_periods)]

                x_future = np.arange(len(agg), len(agg) + fc_periods, dtype=float)
                y_forecast = np.polyval(coeffs, x_future)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=agg["_period"], y=y, mode="lines+markers",
                    name="Historical", line=dict(color="#5b4bff", width=2.5),
                    marker=dict(size=6, color="#5b4bff")))
                fig.add_trace(go.Scatter(
                    x=agg["_period"], y=trend_line, mode="lines",
                    name="Trend line", line=dict(color="#94a3b8", width=1.5, dash="dot")))
                fig.add_trace(go.Scatter(
                    x=future_dates, y=y_forecast, mode="lines+markers",
                    name="Forecast", line=dict(color="#10b981", width=2.5, dash="dash"),
                    marker=dict(size=7, color="#10b981", symbol="diamond")))
                fig.add_trace(go.Scatter(
                    x=list(future_dates) + list(reversed(future_dates)),
                    y=list(y_forecast + 1.96*std_resid) + list(reversed(y_forecast - 1.96*std_resid)),
                    fill="toself", fillcolor="rgba(16,185,129,0.10)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="95% confidence", hoverinfo="skip"))

                fig.add_vline(x=str(last_date), line_width=1.5, line_dash="dash", line_color="#e2e8f0")
                fig.update_layout(**mk_layout(height=420, showlegend=True))
                fig.update_layout(
                    legend=dict(orientation="h", y=1.02, x=0),
                    xaxis_title="Period",
                    yaxis_title=fc_num.replace("_", " ").title()
                )
                st.plotly_chart(fig, use_container_width=True)

                last_val = float(y[-1])
                next_val = float(y_forecast[0])
                end_val  = float(y_forecast[-1])
                pct_chg  = (next_val - last_val) / abs(last_val) * 100 if last_val != 0 else 0
                growth_rate = float(coeffs[0])

                fk1, fk2, fk3, fk4 = st.columns(4)
                with fk1:
                    dir_color = "#10b981" if growth_rate >= 0 else "#ef4444"
                    dir_txt   = "▲ Upward" if growth_rate >= 0 else "▼ Downward"
                    st.markdown(f'<div class="kpi-card"><div class="kpi-bar" style="background:{dir_color}"></div><div class="kpi-label">Trend Direction</div><div class="kpi-value" style="font-size:18px;color:{dir_color}">{dir_txt}</div><div class="kpi-sub">{fmt_plain(abs(growth_rate))} per period</div></div>', unsafe_allow_html=True)
                with fk2:
                    st.markdown(f'<div class="kpi-card"><div class="kpi-bar" style="background:#5b4bff"></div><div class="kpi-label">Next Period</div><div class="kpi-value">{fmt_plain(next_val)}</div><div class="kpi-sub">Current: {fmt_plain(last_val)}</div></div>', unsafe_allow_html=True)
                with fk3:
                    chg_cls = "kpi-up" if pct_chg >= 0 else "kpi-dn"
                    st.markdown(f'<div class="kpi-card"><div class="kpi-bar" style="background:#f59e0b"></div><div class="kpi-label">Expected Change</div><div class="kpi-value"><span class="{chg_cls}">{pct_chg:+.1f}%</span></div><div class="kpi-sub">vs current period</div></div>', unsafe_allow_html=True)
                with fk4:
                    st.markdown(f'<div class="kpi-card"><div class="kpi-bar" style="background:#8b5cf6"></div><div class="kpi-label">{fc_periods} {fc_unit.title()} Out</div><div class="kpi-value">{fmt_plain(end_val)}</div><div class="kpi-sub">End of forecast range</div></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                fc_cache_key = f"ai_forecast_{fc_num}_{fc_periods}_{fc_unit}"
                forecast_ctx = (
                    f"Metric: {fc_num}. Historical periods: {len(agg)}. "
                    f"Last value: {fmt_plain(last_val)}. "
                    f"Trend: {'growing' if growth_rate>=0 else 'declining'} at {fmt_plain(abs(growth_rate))} per period. "
                    f"Next period forecast: {fmt_plain(next_val)} ({pct_chg:+.1f}% change). "
                    f"{fc_periods}-{fc_unit} forecast end: {fmt_plain(end_val)}. Dataset type: {dtype}."
                )
                if fc_cache_key not in st.session_state:
                    st.markdown('<div class="ai-box-title" style="font-size:11px;font-weight:700;color:#5b4bff;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">🤖 Forecast Interpretation</div>', unsafe_allow_html=True)
                    st.session_state[fc_cache_key] = st.write_stream(
                        ask_claude(
                            f"Interpret this forecast and give 3-4 specific actionable recommendations: {forecast_ctx}",
                            df_info, dtype
                        )
                    )
                else:
                    st.markdown(
                        f'<div class="ai-box"><div class="ai-box-title">🤖 Forecast Interpretation</div>'
                        f'<div class="ai-box-text">{st.session_state[fc_cache_key].replace(chr(10),"<br>")}</div></div>',
                        unsafe_allow_html=True)
                if st.button("↻ Regenerate interpretation", key="regen_forecast"):
                    if fc_cache_key in st.session_state:
                        del st.session_state[fc_cache_key]
                    st.rerun()

        except Exception as _fe:
            st.error(f"Forecasting error: {_fe}")

else:
    st.info("Select a page from the navigation bar above.")
