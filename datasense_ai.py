import os, json
import streamlit as st
import pandas as pd
import numpy as np
import anthropic
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

try:
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass

st.set_page_config(page_title="DataSense AI", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
*, html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background: #f1f5f9; }

/* SIDEBAR */
section[data-testid="stSidebar"] { background: #0f172a !important; }
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
section[data-testid="stSidebar"] .stSuccess p { color: #10b981 !important; }
section[data-testid="stSidebar"] hr { border-color: #1e293b !important; }

/* BUTTONS */
.stButton > button {
    background: #1e293b; color: #e2e8f0; border: 1px solid #334155;
    border-radius: 8px; font-size: 12px; font-weight: 500;
    transition: all 0.15s; width: 100%;
}
.stButton > button:hover { background: #2563eb; border-color: #2563eb; color: #fff; }

/* FILE UPLOADER */
div[data-testid="stFileUploader"] { background: #1e293b; border: 1.5px dashed #334155; border-radius: 10px; padding: 0.75rem; }
div[data-testid="stFileUploader"] * { color: #94a3b8 !important; }
div[data-testid="stFileUploader"] button { background: #2563eb !important; color: #fff !important; border: none !important; border-radius: 6px !important; }
div[data-testid="stFileUploaderDropzoneInstructions"] span { color: #60a5fa !important; font-weight: 600; }

/* TABS */
.stTabs [data-baseweb="tab-list"] { background: #fff; border-radius: 10px; padding: 3px; border: 1px solid #e2e8f0; gap: 2px; }
.stTabs [data-baseweb="tab"] { color: #64748b; border-radius: 8px; font-size: 13px; font-weight: 500; padding: 6px 14px; }
.stTabs [aria-selected="true"] { background: #0f172a !important; color: #fff !important; }

/* KPI CARDS */
.kpi-card {
    background: #fff; border-radius: 14px; padding: 1.25rem 1.5rem;
    border: 1px solid #e2e8f0; position: relative; overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.kpi-bar { position: absolute; top: 0; left: 0; width: 4px; height: 100%; border-radius: 14px 0 0 14px; }
.kpi-label { font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 10px; }
.kpi-value { font-size: 28px; font-weight: 800; color: #0f172a; line-height: 1; margin-bottom: 6px; }
.kpi-sub { font-size: 12px; color: #64748b; }
.kpi-up { color: #10b981; font-weight: 700; }
.kpi-dn { color: #ef4444; font-weight: 700; }

/* INSIGHT CARDS */
.insight-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-radius: 14px; padding: 1.25rem 1.5rem; margin-bottom: 0.75rem;
    border: 1px solid #334155;
}
.insight-tag {
    display: inline-block; border-radius: 20px; padding: 2px 10px;
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.07em; margin-bottom: 10px;
}
.insight-title { font-size: 14px; font-weight: 700; color: #f1f5f9; margin-bottom: 6px; }
.insight-text { font-size: 13px; line-height: 1.7; color: #94a3b8; }
.insight-number { font-size: 24px; font-weight: 800; margin-top: 8px; }

/* AI BOX */
.ai-box {
    background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%);
    border: 1px solid #bfdbfe; border-radius: 14px;
    padding: 1.25rem 1.5rem; margin-bottom: 0.75rem;
}
.ai-box-title { font-size: 11px; font-weight: 700; color: #1d4ed8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
.ai-box-text { font-size: 13px; color: #1e293b; line-height: 1.75; }

/* SECTION HEADER */
.sec-head {
    font-size: 12px; font-weight: 700; color: #475569;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin: 1.5rem 0 0.75rem; padding-left: 12px;
    border-left: 3px solid #2563eb;
}

/* BADGES */
.badge { display: inline-block; border-radius: 6px; padding: 2px 8px; font-size: 11px; font-weight: 700; }
.badge-green { background: #dcfce7; color: #15803d; }
.badge-amber { background: #fef9c3; color: #a16207; }
.badge-red   { background: #fee2e2; color: #b91c1c; }
.badge-blue  { background: #dbeafe; color: #1d4ed8; }

/* STAT ROW */
.stat-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 0.5rem; }
.stat-pill {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 4px 12px; font-size: 12px; color: #475569;
}
.stat-pill b { color: #0f172a; }

h1,h2,h3,h4 { color: #0f172a !important; }
.stTextInput > div > div > input { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; color: #0f172a; }
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
        if abs(v) >= 1e9:  return f"£{v/1e9:.1f}B"
        if abs(v) >= 1e6:  return f"£{v/1e6:.1f}M"
        if abs(v) >= 1000: return f"£{v:,.0f}"
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
def compute_smart_insights(df, col_analysis, dtype):
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
                "text": f"For every £1 spent, you earn £{avg_roas:.2f} back. Range: {df[roas_col].min():.2f}x — {df[roas_col].max():.2f}x.",
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
            avg_ctr = df[ctr_col].mean() * 100
            insights.append({
                "tag": "👆 CTR", "color": "#8b5cf6",
                "title": f"Average CTR: {avg_ctr:.2f}%",
                "text": f"Click-through rate ranges from {df[ctr_col].min()*100:.2f}% to {df[ctr_col].max()*100:.2f}%. Industry avg is 2-5%.",
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

def compute_time_insights(df, col_analysis):
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

def compute_advanced_stats(df, col_analysis):
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

def compute_correlations(df, col_analysis):
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
            font=dict(family='Plus Jakarta Sans, sans-serif', size=12, color='#475569'),
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
def get_ai_analysis(df_info, dtype):
    try:
        client = anthropic.Anthropic()
        prompt = DATASET_CONFIG[dtype]["insights_prompt"]
        response = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=800,
            system=prompt,
            messages=[{"role":"user","content":f"Dataset:\n{df_info}"}]
        )
        return response.content[0].text
    except anthropic.AuthenticationError:
        return "Auth error — check your API key."
    except Exception as e:
        return f"Error: {e}"

def ask_claude(question, df_info, dtype):
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=800,
            system=(f"You are an expert {dtype} data analyst.\nDataset:\n{df_info}\n"
                    f"Give precise, specific answers using actual numbers from the data. Use bullet points."),
            messages=[{"role":"user","content":question}]
        )
        return response.content[0].text
    except anthropic.AuthenticationError:
        return "Auth error — check your API key."
    except Exception as e:
        return f"Error: {e}"

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<div style='padding:1rem 0 0.5rem'><span style='font-size:22px'>🧠</span>"
                "<span style='color:#f1f5f9;font-size:17px;font-weight:800;margin-left:8px'>DataSense AI</span></div>",
                unsafe_allow_html=True)
    st.markdown("<p style='color:#475569;font-size:11px;margin-top:-4px;margin-bottom:1rem'>Power BI Style Analytics</p>",
                unsafe_allow_html=True)
    st.divider()
    st.markdown("<p style='color:#475569;font-size:10px;text-transform:uppercase;letter-spacing:0.1em;font-weight:700;margin-bottom:8px'>Data Source</p>",
                unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["csv","xlsx","xls"], label_visibility="collapsed")
    if uploaded:
        st.success(f"✓ {uploaded.name}")

# ══════════════════════════════════════════════════════════════════════════════
# WELCOME
# ══════════════════════════════════════════════════════════════════════════════
if not uploaded:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='max-width:680px'>
      <h1 style='font-size:34px;font-weight:800;color:#0f172a;line-height:1.2;margin-bottom:12px'>
        AI-Powered Analytics<br><span style='color:#2563eb'>for Any Dataset</span>
      </h1>
      <p style='font-size:16px;color:#64748b;margin-bottom:2rem'>
        Upload any CSV or Excel file. DataSense AI automatically detects your data type,
        generates custom KPIs, charts, and AI-written insights specific to your data.
      </p>
    </div>
    """, unsafe_allow_html=True)

    cards = [
        ("🛒","Sales","Revenue, products, regions, orders","#dbeafe","#1d4ed8"),
        ("💰","Finance","Budget, expenses, profit, cashflow","#dcfce7","#15803d"),
        ("👥","HR","Employees, salary, performance","#ede9fe","#6d28d9"),
        ("📣","Marketing","Campaigns, ROAS, CTR, conversions","#fce7f3","#9d174d"),
    ]
    c1,c2,c3,c4 = st.columns(4)
    for col_obj, (icon, title, desc, bg, tc) in zip([c1,c2,c3,c4], cards):
        with col_obj:
            st.markdown(f"""
            <div style='background:{bg};border-radius:16px;padding:1.25rem 1.5rem;border:1px solid transparent'>
              <div style='font-size:28px;margin-bottom:8px'>{icon}</div>
              <div style='font-weight:700;color:{tc};font-size:15px;margin-bottom:4px'>{title}</div>
              <div style='font-size:12px;color:#475569'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#fff;border-radius:16px;padding:1.5rem 2rem;border:1px solid #e2e8f0;max-width:680px'>
      <p style='font-weight:700;color:#0f172a;font-size:14px;margin-bottom:12px'>How it works</p>
      <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;font-size:13px;color:#475569'>
        <div>📤 <b style='color:#0f172a'>Upload</b> any CSV or Excel</div>
        <div>🔍 <b style='color:#0f172a'>Auto-detect</b> dataset type</div>
        <div>📊 <b style='color:#0f172a'>Generate</b> custom KPIs & charts</div>
        <div>🤖 <b style='color:#0f172a'>AI insights</b> specific to your data</div>
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
try:
    df = load_dataframe(uploaded)
except Exception as e:
    st.error(f"Could not load file: {e}"); st.stop()

if st.session_state.get("last_file") != uploaded.name:
    st.session_state["active_view"] = "overview"
    st.session_state["last_file"]   = uploaded.name
    st.session_state["messages"]    = []
    for k in list(st.session_state.keys()):
        if k.startswith("ai_"): del st.session_state[k]

col_analysis  = analyze_columns(df)
dtype         = detect_type(df)
cfg           = DATASET_CONFIG[dtype]
adv_stats     = compute_advanced_stats(df, col_analysis)
correlations  = compute_correlations(df, col_analysis)
time_insights = compute_time_insights(df, col_analysis)
smart_insights= compute_smart_insights(df, col_analysis, dtype)

# ── SIDEBAR NAV ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.markdown(f"<p style='color:{cfg['color']};font-size:11px;font-weight:700;margin-bottom:8px'>{cfg['icon']} {cfg['label'].upper()}</p>",
                unsafe_allow_html=True)
    st.markdown("<p style='color:#475569;font-size:10px;text-transform:uppercase;letter-spacing:0.1em;font-weight:700;margin-bottom:6px'>Pages</p>",
                unsafe_allow_html=True)

    pages = [("🏠  Overview","overview"),("💡  Smart Insights","smart"),
             ("📊  KPI Report","kpis"),("📈  Trends","trends"),
             ("📦  Categories","categories"),("🔗  Correlations","correlations"),
             ("🔍  Anomalies","anomalies"),("🤖  AI Analysis","ai"),
             ("📋  Data Table","table")]

    if not col_analysis["date"] or not col_analysis["numeric"]:
        pages = [p for p in pages if p[1] != "trends"]
    if not col_analysis["categorical"]:
        pages = [p for p in pages if p[1] != "categories"]
    if len(col_analysis["numeric"]) < 2:
        pages = [p for p in pages if p[1] != "correlations"]

    for label, key in pages:
        if st.button(label, key=f"pg_{key}", use_container_width=True):
            st.session_state["active_view"] = key

    st.divider()
    st.markdown(f"<p style='color:#334155;font-size:11px'>"
                f"<b style='color:#64748b'>{df.shape[0]:,}</b> rows · "
                f"<b style='color:#64748b'>{df.shape[1]}</b> cols<br>"
                f"<b style='color:#64748b'>{len(col_analysis['numeric'])}</b> numeric · "
                f"<b style='color:#64748b'>{len(col_analysis['categorical'])}</b> categorical · "
                f"<b style='color:#64748b'>{len(col_analysis['date'])}</b> date</p>",
                unsafe_allow_html=True)

active_view = st.session_state.get("active_view", "overview")

# ── PAGE HEADER ───────────────────────────────────────────────────────────────
file_title = (uploaded.name.replace(".csv","").replace(".xlsx","").replace(".xls","")
              .replace("_"," ").replace("-"," ").title())

hc1, hc2, hc3 = st.columns([3, 1, 1])
with hc1:
    st.markdown(f"<h3 style='margin-bottom:2px;font-weight:800'>{cfg['icon']} {file_title}</h3>",
                unsafe_allow_html=True)
    st.markdown(f"<div class='stat-row'>"
                f"<span class='stat-pill'>{dtype.title()} Dataset</span>"
                f"<span class='stat-pill'><b>{df.shape[0]:,}</b> rows</span>"
                f"<span class='stat-pill'><b>{df.shape[1]}</b> columns</span>"
                f"<span class='stat-pill'>{datetime.now().strftime('%d %b %Y')}</span>"
                f"</div>", unsafe_allow_html=True)
with hc2:
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇ Export CSV", csv, f"{file_title}.csv", "text/csv", use_container_width=True)
with hc3:
    if st.button("🔄 Refresh AI", use_container_width=True):
        for k in list(st.session_state.keys()):
            if k.startswith("ai_"): del st.session_state[k]
        st.rerun()

# ── DATA AGENT button — full width below title ────────────────────────────────
st.markdown(
    f'''<div style="margin-bottom:1rem">
    <button onclick="window.location.href=\'#\';" style="
        width:100%; padding:10px 20px; background:linear-gradient(135deg,#0f172a,#1e3a5f);
        color:#e2e8f0; border:1px solid #334155; border-radius:10px;
        font-size:13px; font-weight:600; cursor:pointer; text-align:left;
        display:flex; align-items:center; gap:8px;">
        🔎 Data Agent — Ask questions about your {file_title} data in plain English
    </button></div>''', unsafe_allow_html=True)

if st.button("🔎 Open Data Agent — Ask questions, filter data, get charts", use_container_width=True):
    st.session_state["active_view"] = "agent"
    st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

df_info = (f"Dataset: {uploaded.name} | Type: {dtype} | {df.shape[0]} rows x {df.shape[1]} cols\n"
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
        st.markdown('<div class="sec-head">Primary Chart</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="sec-head">Distribution</div>', unsafe_allow_html=True)
        try:
            if col_analysis["categorical"] and col_analysis["numeric"]:
                cat = col_analysis["categorical"][0]; num = col_analysis["numeric"][0]
                st.plotly_chart(plot_pie(df, cat, num), use_container_width=True)
        except Exception as e:
            st.error(f"Chart error: {e}")

    # AI Summary
    st.markdown('<div class="sec-head">AI Executive Summary</div>', unsafe_allow_html=True)
    if "ai_exec" not in st.session_state:
        with st.spinner("Generating AI insights..."):
            st.session_state["ai_exec"] = get_ai_analysis(df_info, dtype)
    st.markdown(f'<div class="ai-box"><div class="ai-box-title">🤖 {cfg["label"]} — Executive Summary</div>'
                f'<div class="ai-box-text">{st.session_state["ai_exec"].replace(chr(10),"<br>")}</div></div>',
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SMART INSIGHTS (dataset-specific)
# ══════════════════════════════════════════════════════════════════════════════
elif active_view == "smart":
    st.markdown(f'<div class="sec-head">{cfg["icon"]} Smart Insights — {cfg["label"]}</div>', unsafe_allow_html=True)
    st.markdown(f"<p style='color:#64748b;font-size:13px;margin-bottom:1.5rem'>Auto-generated insights specific to your {dtype} dataset based on actual column values.</p>", unsafe_allow_html=True)

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
        st.markdown('<div class="sec-head">Category Deep Dive</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="sec-head">KPI Report — All Numeric Columns</div>', unsafe_allow_html=True)
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

        st.markdown('<div class="sec-head">Distributions</div>', unsafe_allow_html=True)
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
            st.markdown('<div class="sec-head">Heatmap by Period & Category</div>', unsafe_allow_html=True)
            try: st.plotly_chart(plot_heatmap(df, col_analysis), use_container_width=True)
            except Exception as e: st.error(f"Heatmap error: {e}")

        if "ai_trends" not in st.session_state:
            with st.spinner("AI trend analysis..."):
                st.session_state["ai_trends"] = get_ai_analysis(df_info, dtype)
        st.markdown(f'<div class="ai-box"><div class="ai-box-title">🤖 Trend Analysis</div>'
                    f'<div class="ai-box-text">{st.session_state["ai_trends"].replace(chr(10),"<br>")}</div></div>',
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CATEGORIES
# ══════════════════════════════════════════════════════════════════════════════
elif active_view == "categories":
    st.markdown('<div class="sec-head">Category Analysis</div>', unsafe_allow_html=True)
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
            with st.spinner("AI category analysis..."):
                st.session_state["ai_cat"] = get_ai_analysis(df_info, dtype)
        st.markdown(f'<div class="ai-box"><div class="ai-box-title">🤖 Category Analysis</div>'
                    f'<div class="ai-box-text">{st.session_state["ai_cat"].replace(chr(10),"<br>")}</div></div>',
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CORRELATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif active_view == "correlations":
    st.markdown('<div class="sec-head">Correlation Analysis</div>', unsafe_allow_html=True)
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
                    f'<div><b style="color:#0f172a">{pair["col1"].replace("_"," ").title()}</b>'
                    f' <span style="color:#94a3b8">↔</span> '
                    f'<b style="color:#0f172a">{pair["col2"].replace("_"," ").title()}</b></div>'
                    f'<div><span class="badge {badge_cls}">{strength} {direction}</span>'
                    f' <b style="color:#0f172a;font-size:16px;margin-left:10px">r = {r}</b></div>'
                    f'</div><div style="margin-top:10px;background:#f1f5f9;border-radius:4px;height:5px">'
                    f'<div style="width:{int(abs(r)*100)}%;background:{bar_c};height:5px;border-radius:4px"></div>'
                    f'</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="sec-head">Scatter Plots</div>', unsafe_allow_html=True)
        nums = col_analysis["numeric"]
        pairs_to_plot = [(nums[i],nums[j]) for i in range(min(3,len(nums))) for j in range(i+1,min(4,len(nums)))]
        if pairs_to_plot:
            sc_cols = st.columns(min(len(pairs_to_plot), 3))
            for idx, (c1, c2) in enumerate(pairs_to_plot[:3]):
                with sc_cols[idx]:
                    st.markdown(f"**{c1.replace('_',' ').title()} vs {c2.replace('_',' ').title()}**")
                    try: st.plotly_chart(plot_scatter(df, c1, c2), use_container_width=True)
                    except Exception as e: st.error(str(e))

        st.markdown('<div class="sec-head">Correlation Matrix</div>', unsafe_allow_html=True)
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
        st.dataframe(corr_df.style.applymap(color_corr), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANOMALIES
# ══════════════════════════════════════════════════════════════════════════════
elif active_view == "anomalies":
    st.markdown('<div class="sec-head">Data Quality & Anomaly Detection</div>', unsafe_allow_html=True)
    total_missing  = int(df.isnull().sum().sum())
    total_outliers = sum(s.get("outliers",0) for s in adv_stats.values())
    dupes = int(df.duplicated().sum())
    completeness = round((1 - df.isnull().sum().sum() / (df.shape[0]*df.shape[1])) * 100, 1)

    mc1,mc2,mc3,mc4 = st.columns(4)
    with mc1: st.markdown(f'<div class="kpi-card"><div class="kpi-bar" style="background:#2563eb"></div><div class="kpi-label">Data Completeness</div><div class="kpi-value">{completeness}%</div></div>', unsafe_allow_html=True)
    with mc2: st.markdown(f'<div class="kpi-card"><div class="kpi-bar" style="background:{"#ef4444" if total_missing>0 else "#10b981"}"></div><div class="kpi-label">Missing Values</div><div class="kpi-value">{total_missing:,}</div><div class="kpi-sub">Across all columns</div></div>', unsafe_allow_html=True)
    with mc3: st.markdown(f'<div class="kpi-card"><div class="kpi-bar" style="background:#f59e0b"></div><div class="kpi-label">Outliers (IQR)</div><div class="kpi-value">{total_outliers:,}</div></div>', unsafe_allow_html=True)
    with mc4: st.markdown(f'<div class="kpi-card"><div class="kpi-bar" style="background:{"#ef4444" if dupes>0 else "#10b981"}"></div><div class="kpi-label">Duplicate Rows</div><div class="kpi-value">{dupes:,}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-head">Column Quality Report</div>', unsafe_allow_html=True)
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

    st.markdown('<div class="sec-head">Outlier Details</div>', unsafe_allow_html=True)
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
        with st.spinner("AI anomaly analysis..."):
            st.session_state["ai_anomaly"] = get_ai_analysis(df_info, dtype)
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
        ("ai_exec",   "🏢 Executive Summary",  "Key findings, risks and recommendations"),
        ("ai_trends", "📈 Trend Analysis",      "Time patterns, growth rates, forecasting"),
        ("ai_cat",    "📦 Category Insights",   "Segment performance and quick wins"),
        ("ai_anomaly","🔍 Data Quality Report", "Outliers, anomalies and data issues"),
    ]
    for key, title, desc in ai_sections:
        with st.expander(f"{title} — {desc}", expanded=(key=="ai_exec")):
            if key not in st.session_state:
                with st.spinner(f"Generating {title}..."):
                    st.session_state[key] = get_ai_analysis(df_info, dtype)
            st.markdown(
                f'<div class="ai-box"><div class="ai-box-title">{title}</div>'
                f'<div class="ai-box-text">{st.session_state[key].replace(chr(10),"<br>")}</div></div>',
                unsafe_allow_html=True)
            if st.button(f"↻ Regenerate", key=f"regen_{key}"):
                del st.session_state[key]; st.rerun()

    st.markdown('<div class="sec-head">Ask a Question</div>', unsafe_allow_html=True)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div style="background:#0f172a;border-radius:12px 12px 4px 12px;padding:0.75rem 1rem;margin:0.5rem 0;color:#e2e8f0;font-size:13px">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ai-box" style="margin:0.5rem 0"><div class="ai-box-text">🤖 {msg["content"].replace(chr(10),"<br>")}</div></div>', unsafe_allow_html=True)
    with st.form("chat", clear_on_submit=True):
        qc1, qc2 = st.columns([5,1])
        with qc1:
            q = st.text_input("", placeholder=f"Ask anything about your {dtype} data...", label_visibility="collapsed")
        with qc2:
            sub = st.form_submit_button("Ask →", use_container_width=True)
    if sub and q:
        st.session_state.messages.append({"role":"user","content":q})
        with st.spinner("Thinking..."):
            a = ask_claude(q, df_info, dtype)
        st.session_state.messages.append({"role":"assistant","content":a})
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
        mask = df.apply(lambda row: row.astype(str).str.contains(search, case=False, na=False).any(), axis=1)
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
    st.markdown('<div class="sec-head">Quick Queries — based on your data columns</div>', unsafe_allow_html=True)
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
            st.markdown(
                f'''<div style="background:#0f172a;border-radius:12px 12px 4px 12px;
                padding:0.75rem 1rem;margin:0.5rem 0;color:#e2e8f0;font-size:13px;
                border:1px solid #1e293b">🧑 {msg["content"]}</div>''',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'''<div style="background:#fff;border:1px solid #e2e8f0;
                border-radius:12px 12px 12px 4px;padding:0.75rem 1rem;
                margin:0.5rem 0;font-size:13px;color:#1e293b">
                🤖 {msg["content"]}</div>''',
                unsafe_allow_html=True)
            if "dataframe" in msg:
                st.dataframe(msg["dataframe"], use_container_width=True)
                csv_out = msg["dataframe"].to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Download this result", csv_out,
                                   "query_result.csv", "text/csv",
                                   key=f"dl_{msg.get('id',0)}")
            if "chart" in msg:
                st.plotly_chart(msg["chart"], use_container_width=True)

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
                    model="claude-sonnet-4-20250514",
                    max_tokens=1000,
                    system=system,
                    messages=[{"role": "user", "content": agent_q}]
                )

                code = response.content[0].text.strip()
                # Remove any accidental markdown
                if "```" in code:
                    code = code.split("```python")[-1].split("```")[0].strip()
                    if not code:
                        code = code.split("```")[-2].strip()

                # Execute the code
                exec_globals = {"df": df.copy(), "pd": pd, "np": np,
                                "result": None, "answer_text": ""}
                try:
                    exec(code, exec_globals)
                    result = exec_globals.get("result")
                    answer_text = exec_globals.get("answer_text", "")

                    msg = {"role": "assistant", "id": len(st.session_state.agent_messages),
                           "content": answer_text or "Here are the results:"}

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
                    fallback = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=400,
                        system="You are a data analyst. Dataset:\n" + df_info + "\nAnswer concisely with numbers.",
                        messages=[{"role": "user", "content": agent_q}]
                    )
                    st.session_state.agent_messages.append({
                        "role": "assistant",
                        "content": fallback.content[0].text,
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

else:
    st.info("Select a page from the sidebar.")
