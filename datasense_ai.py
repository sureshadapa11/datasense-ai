import os, json
import streamlit as st
import pandas as pd
import numpy as np
import anthropic
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# ── API KEY ───────────────────────────────────────────────────────────────────
try:
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass

st.set_page_config(
    page_title="DataSense AI — Power BI Style Analytics",
    page_icon="🧠", layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.stApp{background:#f7f8fc;}
section[data-testid="stSidebar"]{background:#0f172a;border-right:1px solid #1e293b;}
section[data-testid="stSidebar"] *{color:#94a3b8!important;}
section[data-testid="stSidebar"] h2{color:#f1f5f9!important;}
section[data-testid="stSidebar"] .stSuccess p{color:#10b981!important;}
.kpi{background:#fff;border-radius:12px;padding:1.25rem 1.5rem;border:1px solid #e2e8f0;position:relative;overflow:hidden;margin-bottom:0.5rem;}
.kpi-accent{position:absolute;top:0;left:0;width:4px;height:100%;}
.kpi-label{font-size:11px;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:8px;}
.kpi-value{font-size:26px;font-weight:700;color:#0f172a;line-height:1;}
.kpi-sub{font-size:12px;color:#64748b;margin-top:6px;}
.kpi-trend-up{color:#10b981;font-weight:600;}
.kpi-trend-dn{color:#ef4444;font-weight:600;}
.section-head{font-size:13px;font-weight:700;color:#0f172a;margin:1.25rem 0 0.5rem;text-transform:uppercase;letter-spacing:0.06em;border-left:3px solid #2563eb;padding-left:10px;}
.ai-box{background:linear-gradient(135deg,#eff6ff,#f0fdf4);border:1px solid #bfdbfe;border-radius:12px;padding:1.25rem 1.5rem;margin-bottom:0.75rem;}
.ai-box-title{font-size:12px;font-weight:700;color:#1d4ed8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;}
.ai-box-text{font-size:13px;color:#1e293b;line-height:1.7;}
.badge{display:inline-block;border-radius:6px;padding:2px 8px;font-size:11px;font-weight:600;}
.badge-green{background:#dcfce7;color:#15803d;}
.badge-amber{background:#fef9c3;color:#a16207;}
.badge-red{background:#fee2e2;color:#b91c1c;}
.insight-card{background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);border-radius:12px;padding:1.25rem 1.5rem;color:#e2e8f0;margin-bottom:0.75rem;}
.insight-tag{display:inline-block;background:rgba(255,255,255,0.12);border-radius:20px;padding:2px 10px;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;}
.insight-text{font-size:13px;line-height:1.6;color:#cbd5e1;}
.stButton>button{background:#0f172a;color:#f1f5f9;border:1px solid #1e293b;border-radius:8px;font-size:12px;font-weight:500;}
.stButton>button:hover{background:#1e293b;border-color:#3b82f6;color:#fff;}
div[data-testid="stFileUploader"]{background:#1e293b;border:1px dashed #334155;border-radius:10px;padding:0.75rem;}
div[data-testid="stFileUploader"] *{color:#94a3b8!important;}
div[data-testid="stFileUploader"] button{background:#2563eb!important;color:#fff!important;border:none!important;border-radius:6px!important;}
div[data-testid="stFileUploaderDropzoneInstructions"] span{color:#60a5fa!important;font-weight:600;}
.stTabs [data-baseweb="tab-list"]{background:#fff;border-radius:10px;padding:3px;border:1px solid #e2e8f0;}
.stTabs [data-baseweb="tab"]{color:#64748b;border-radius:8px;font-size:13px;font-weight:500;}
.stTabs [aria-selected="true"]{background:#0f172a;color:#fff;}
h1,h2,h3,h4{color:#0f172a!important;}
</style>
""", unsafe_allow_html=True)

COLORS = ['#2563eb','#10b981','#f59e0b','#ef4444','#8b5cf6',
          '#06b6d4','#f97316','#84cc16','#ec4899','#14b8a6']

PLOTLY_LAYOUT = dict(
    paper_bgcolor='white', plot_bgcolor='white',
    font=dict(family='DM Sans, sans-serif', size=12, color='#475569'),
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis=dict(showgrid=False, linecolor='#e2e8f0'),
    yaxis=dict(gridcolor='#f1f5f9', linecolor='#e2e8f0'),
)

DATASET_TYPES = {
    "sales":    {"icon":"🛒","label":"Sales Analytics","keywords":["revenue","sales","product","region","order","customer","discount","profit","units","sold"]},
    "finance":  {"icon":"💰","label":"Finance Analytics","keywords":["expense","budget","cashflow","profit","loss","asset","liability","income","cost","invoice"]},
    "hr":       {"icon":"👥","label":"HR Analytics","keywords":["employee","salary","department","hire","leave","performance","headcount","staff","payroll","role"]},
    "marketing":{"icon":"📣","label":"Marketing Analytics","keywords":["campaign","click","impression","conversion","lead","channel","ctr","cpc","roas","audience"]},
    "inventory":{"icon":"📦","label":"Inventory Analytics","keywords":["stock","inventory","sku","warehouse","supplier","reorder","quantity","item","shelf"]},
    "generic":  {"icon":"📊","label":"Data Analytics","keywords":[]},
}

# ── CORE FUNCTIONS ────────────────────────────────────────────────────────────
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
        for hr in [0,1,2]:
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
    df.columns = [str(c).strip().replace('\n',' ') for c in df.columns]
    df = df.dropna(how='all').dropna(axis=1, how='all').reset_index(drop=True)
    st.session_state[key] = df
    return df

def analyze_columns(df):
    result = {"numeric":[], "categorical":[], "date":[], "text":[]}
    id_names = {'id','index','no','num','number','#','sr','sr.','row','seq'}
    for col in df.columns:
        s = df[col].dropna()
        if len(s) == 0: continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            result["date"].append(col); continue
        if pd.api.types.is_numeric_dtype(df[col]):
            is_id = (str(col).lower().strip() in id_names or
                     (df[col].nunique()==len(df) and str(col).lower().strip().endswith('id')))
            if not is_id: result["numeric"].append(col)
            continue
        try:
            conv = pd.to_datetime(s.astype(str), errors='coerce')
            if conv.notna().sum()/len(s) > 0.7:
                result["date"].append(col); continue
        except: pass
        if s.nunique() <= 30 or s.nunique()/len(s) <= 0.3:
            result["categorical"].append(col)
        else:
            result["text"].append(col)
    return result

def detect_type(df):
    cols = ' '.join(df.columns.tolist()).lower()
    scores = {k: sum(1 for w in v["keywords"] if w in cols)
              for k,v in DATASET_TYPES.items() if k != "generic"}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "generic"

def fmt(v):
    try:
        if pd.isna(v): return "—"
        v = float(v)
        if abs(v)>=1e9:  return f"{v/1e9:.1f}B"
        if abs(v)>=1e6:  return f"{v/1e6:.1f}M"
        if abs(v)>=1000: return f"{v:,.0f}"
        if abs(v)>=1:    return f"{v:,.1f}"
        return f"{v:.3f}"
    except: return "—"

def pct_change(a, b):
    try:
        return round((a-b)/abs(b)*100,1) if b!=0 else 0
    except: return 0

def compute_advanced_stats(df, col_analysis):
    stats = {}
    for nc in col_analysis["numeric"]:
        s = df[nc].dropna()
        if len(s) == 0: continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outliers = int(((s < q1-1.5*iqr) | (s > q3+1.5*iqr)).sum())
        stats[nc] = {"sum":s.sum(),"mean":s.mean(),"median":s.median(),
                     "min":s.min(),"max":s.max(),"std":s.std(),
                     "count":len(s),"missing":int(df[nc].isna().sum()),
                     "outliers":outliers,"q1":q1,"q3":q3}
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
                pairs.append({"col1":nums[i],"col2":nums[j],"r":round(float(r),3)})
    pairs.sort(key=lambda x: abs(x["r"]), reverse=True)
    return pairs[:8]

def compute_category_insights(df, col_analysis):
    insights = {}
    if not col_analysis["categorical"] or not col_analysis["numeric"]:
        return insights
    num_col = col_analysis["numeric"][0]
    for cat in col_analysis["categorical"][:4]:
        grp = df.groupby(cat)[num_col].sum().sort_values(ascending=False)
        total = grp.sum()
        insights[cat] = {
            "top": str(grp.index[0]) if len(grp)>0 else "—",
            "top_val": float(grp.iloc[0]) if len(grp)>0 else 0,
            "bottom": str(grp.index[-1]) if len(grp)>0 else "—",
            "bottom_val": float(grp.iloc[-1]) if len(grp)>0 else 0,
            "top3_share": round(float(grp.head(3).sum()/total*100),1) if total>0 else 0,
            "unique": int(df[cat].nunique()), "num_col": num_col,
        }
    return insights

def compute_time_insights(df, col_analysis):
    if not col_analysis["date"] or not col_analysis["numeric"]:
        return {}
    date_col = col_analysis["date"][0]
    num_col  = col_analysis["numeric"][0]
    df2 = df.copy()
    df2['_dt'] = pd.to_datetime(df2[date_col].astype(str), errors='coerce')
    df2 = df2.dropna(subset=['_dt'])
    df2['_m'] = df2['_dt'].dt.to_period('M')
    monthly = df2.groupby('_m')[num_col].sum().sort_index()
    if len(monthly) < 2: return {}
    growth = pct_change(float(monthly.iloc[-1]), float(monthly.iloc[0]))
    mom = monthly.pct_change().dropna()
    return {
        "periods": len(monthly), "growth": growth,
        "best_period": str(monthly.idxmax()),
        "worst_period": str(monthly.idxmin()),
        "best_val": float(monthly.max()),
        "worst_val": float(monthly.min()),
        "best_growth_pct": round(float(mom.max()*100),1) if len(mom)>0 else 0,
        "num_col": num_col, "date_col": date_col,
    }

# ── PLOTLY CHART FUNCTIONS ────────────────────────────────────────────────────
def plot_trend(df, col_analysis):
    date_col = col_analysis["date"][0]
    num_cols = col_analysis["numeric"][:4]
    df2 = df.copy()
    df2['_dt'] = pd.to_datetime(df2[date_col].astype(str), errors='coerce')
    df2 = df2.dropna(subset=['_dt'])
    df2['_period'] = df2['_dt'].dt.to_period('M').dt.to_timestamp()
    fig = go.Figure()
    for i, nc in enumerate(num_cols):
        agg = df2.groupby('_period')[nc].sum().reset_index()
        fig.add_trace(go.Scatter(
            x=agg['_period'], y=agg[nc],
            name=nc.replace('_',' ').title(),
            line=dict(color=COLORS[i], width=2.5),
            fill='tozeroy' if i==0 else 'none',
            fillcolor='rgba(37,99,235,0.08)',
            mode='lines+markers', marker=dict(size=5)
        ))
    fig.update_layout(**PLOTLY_LAYOUT, height=380,
                      legend=dict(orientation='h', yanchor='bottom', y=1.02))
    return fig

def plot_bar(df, cat_col, num_col, top_n=12, horizontal=False):
    grp = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(top_n).reset_index()
    if horizontal:
        grp = grp.sort_values(num_col, ascending=True)
        fig = px.bar(grp, x=num_col, y=cat_col, orientation='h',
                     color=cat_col, color_discrete_sequence=COLORS)
    else:
        fig = px.bar(grp, x=cat_col, y=num_col,
                     color=cat_col, color_discrete_sequence=COLORS)
    layout = dict(**PLOTLY_LAYOUT)
    layout['height'] = 350
    layout['showlegend'] = False
    layout['xaxis'] = dict(showgrid=False, linecolor='#e2e8f0')
    layout['yaxis'] = dict(gridcolor='#f1f5f9', linecolor='#e2e8f0')
    fig.update_layout(**layout)
    fig.update_traces(marker_line_width=0)
    return fig

def plot_pie(df, cat_col, num_col):
    grp = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(8).reset_index()
    fig = px.pie(grp, names=cat_col, values=num_col,
                 color_discrete_sequence=COLORS, hole=0.4)
    layout = dict(**PLOTLY_LAYOUT)
    layout['height'] = 320
    layout['showlegend'] = True
    layout['legend'] = dict(orientation='v', x=1.0)
    fig.update_layout(**layout)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def plot_histogram(df, num_col, bins=20):
    fig = px.histogram(df, x=num_col, nbins=bins,
                       color_discrete_sequence=['#2563eb'])
    layout = dict(**PLOTLY_LAYOUT)
    layout['height'] = 260
    layout['showlegend'] = False
    layout['xaxis'] = dict(showgrid=False, linecolor='#e2e8f0')
    layout['yaxis'] = dict(gridcolor='#f1f5f9', linecolor='#e2e8f0')
    fig.update_layout(**layout)
    fig.update_traces(marker_line_width=0.5, marker_line_color='white')
    return fig

def plot_scatter(df, col1, col2):
    sample = df[[col1, col2]].dropna().head(500)
    fig = px.scatter(sample, x=col1, y=col2,
                     color_discrete_sequence=['#2563eb'])
    fig.update_layout(**PLOTLY_LAYOUT, height=300)
    fig.update_traces(marker=dict(size=7, opacity=0.6))
    return fig

def plot_heatmap(df, col_analysis):
    date_col = col_analysis["date"][0]
    cat_col  = col_analysis["categorical"][0]
    num_col  = col_analysis["numeric"][0]
    df2 = df.copy()
    df2['_dt'] = pd.to_datetime(df2[date_col].astype(str), errors='coerce')
    df2['_month'] = df2['_dt'].dt.strftime('%b')
    df2['_mnum']  = df2['_dt'].dt.month
    cats  = df2[cat_col].dropna().unique().tolist()[:7]
    months_order = df2.drop_duplicates('_month').sort_values('_mnum')['_month'].tolist()
    pivot = pd.pivot_table(df2, values=num_col, index=cat_col,
                           columns='_month', aggfunc='sum', fill_value=0)
    pivot = pivot.reindex(columns=[m for m in months_order if m in pivot.columns])
    fig = px.imshow(pivot, color_continuous_scale='Blues',
                    aspect='auto', text_auto='.0f')
    layout = dict(**PLOTLY_LAYOUT)
    layout['height'] = max(250, len(cats)*50+80)
    layout['showlegend'] = False
    fig.update_layout(**layout)
    fig.update_coloraxes(showscale=False)
    return fig

# ── AI FUNCTIONS ──────────────────────────────────────────────────────────────
def get_ai_analysis(df_info, analysis_type="executive"):
    try:
        client = anthropic.Anthropic()
        prompts = {
            "executive": ("You are a senior data analyst writing an executive summary. "
                "Provide:\n1. A 2-sentence executive summary\n2. Top 3 key findings (with numbers)\n"
                "3. Top 3 risks\n4. Top 3 recommended actions\nBe specific with real numbers. Concise."),
            "trends": ("Analyze the time trends:\n1. Overall trend direction\n"
                "2. Key inflection points\n3. Seasonality patterns\n4. Forecast direction\nUse numbers."),
            "categories": ("Analyze categorical breakdowns:\n1. Top performing segments\n"
                "2. Underperforming segments\n3. Concentration risk\n4. Quick wins\nUse numbers."),
            "anomalies": ("Find anomalies:\n1. Statistical outliers\n2. Unusual patterns\n"
                "3. Data quality issues\n4. Risk flags\nBe specific and actionable."),
        }
        response = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=600,
            system=prompts.get(analysis_type, prompts["executive"]),
            messages=[{"role":"user","content":f"Dataset:\n{df_info}"}]
        )
        return response.content[0].text
    except anthropic.AuthenticationError:
        return "Auth error — check your API key in Streamlit secrets."
    except Exception as e:
        return f"Error: {e}"

def ask_claude(question, df_info):
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=800,
            system=(f"You are an expert data analyst.\nDataset:\n{df_info}\n"
                    f"Give precise answers with specific numbers. Use bullet points."),
            messages=[{"role":"user","content":question}]
        )
        return response.content[0].text
    except anthropic.AuthenticationError:
        return "Auth error — check your API key."
    except Exception as e:
        return f"Error: {e}"

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='color:#f1f5f9!important;font-size:18px;margin-bottom:0'>🧠 DataSense AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#475569!important;font-size:11px;margin-top:2px'>Power BI Style Analytics</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='color:#475569!important;font-size:10px;text-transform:uppercase;letter-spacing:0.1em;font-weight:600'>DATA SOURCE</p>", unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["csv","xlsx","xls"], label_visibility="collapsed")
    if uploaded:
        st.success(f"✓ {uploaded.name}")

# ── WELCOME SCREEN ────────────────────────────────────────────────────────────
if not uploaded:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h1 style='font-size:30px;font-weight:700;color:#0f172a'>Power BI Style Analytics 🧠</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:15px;color:#64748b;margin-bottom:2rem'>Upload any CSV or Excel file for instant AI-powered analysis — KPIs, charts, trends, correlations, anomalies and executive insights.</p>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    for co, icon, title, desc, color in [
        (c1,"🛒","Sales","Revenue, products, regions","#dbeafe"),
        (c2,"💰","Finance","Budget, expenses, profit","#dcfce7"),
        (c3,"👥","HR","Employees, salary, KPIs","#ede9fe"),
        (c4,"📊","Any Data","Any CSV or Excel file","#fef9c3"),
    ]:
        with co:
            st.markdown(f'<div class="kpi" style="background:{color};border-color:transparent"><div style="font-size:24px;margin-bottom:6px">{icon}</div><div style="font-weight:700;color:#0f172a;font-size:14px;margin-bottom:4px">{title}</div><div style="font-size:12px;color:#475569">{desc}</div></div>', unsafe_allow_html=True)
    st.stop()

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
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
info          = DATASET_TYPES[dtype]
adv_stats     = compute_advanced_stats(df, col_analysis)
correlations  = compute_correlations(df, col_analysis)
cat_insights  = compute_category_insights(df, col_analysis)
time_insights = compute_time_insights(df, col_analysis)

# ── SIDEBAR NAV ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown(f"<p style='color:#60a5fa!important;font-size:11px;font-weight:700'>{info['icon']} {info['label'].upper()}</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#475569!important;font-size:10px;text-transform:uppercase;letter-spacing:0.1em;font-weight:600'>PAGES</p>", unsafe_allow_html=True)

    pages = [("🏠 Overview","overview"),("📊 KPI Report","kpis"),
             ("📈 Trends","trends"),("📦 Categories","categories"),
             ("🔗 Correlations","correlations"),("🔍 Anomalies","anomalies"),
             ("🤖 AI Insights","ai_insights"),("📋 Data Table","table")]

    if not col_analysis["date"] or not col_analysis["numeric"]:
        pages = [p for p in pages if p[1]!="trends"]
    if not col_analysis["categorical"]:
        pages = [p for p in pages if p[1]!="categories"]
    if len(col_analysis["numeric"]) < 2:
        pages = [p for p in pages if p[1]!="correlations"]

    for label, key in pages:
        if st.button(label, key=f"pg_{key}", use_container_width=True):
            st.session_state["active_view"] = key

    st.markdown("---")
    st.markdown(f"<p style='color:#475569!important;font-size:11px'>"
                f"<b style='color:#94a3b8!important'>{df.shape[0]:,}</b> rows · "
                f"<b style='color:#94a3b8!important'>{df.shape[1]}</b> cols<br>"
                f"<b style='color:#94a3b8!important'>{len(col_analysis['numeric'])}</b> numeric · "
                f"<b style='color:#94a3b8!important'>{len(col_analysis['categorical'])}</b> categorical · "
                f"<b style='color:#94a3b8!important'>{len(col_analysis['date'])}</b> date</p>", unsafe_allow_html=True)

active_view = st.session_state.get("active_view", "overview")

# ── PAGE HEADER ───────────────────────────────────────────────────────────────
file_title = (uploaded.name.replace(".csv","").replace(".xlsx","").replace(".xls","")
              .replace("_"," ").replace("-"," ").title())
hc1, hc2 = st.columns([3,1])
with hc1:
    st.markdown(f"<h3 style='margin-bottom:2px'>{info['icon']} {file_title}</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#64748b;font-size:12px;margin-bottom:1rem'>{dtype.title()} Dataset · {df.shape[0]:,} rows · {df.shape[1]} columns · {datetime.now().strftime('%d %b %Y')}</p>", unsafe_allow_html=True)
with hc2:
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇ Export CSV", csv, f"{file_title}.csv", "text/csv", use_container_width=True)

df_info = (f"File: {uploaded.name} | Type: {dtype} | Shape: {df.shape[0]}x{df.shape[1]}\n"
           f"Columns: {', '.join(df.columns.tolist())}\n"
           f"Numeric: {', '.join(col_analysis['numeric'])}\n"
           f"Categorical: {', '.join(col_analysis['categorical'])}\n"
           f"Date: {', '.join(col_analysis['date'])}\n"
           f"Stats:\n{df.describe().to_string()}\nSample:\n{df.head(3).to_string()}")

# ── TOP KPI ROW ───────────────────────────────────────────────────────────────
if col_analysis["numeric"]:
    accent = ["#2563eb","#10b981","#f59e0b","#8b5cf6"]
    kcols  = st.columns(min(len(col_analysis["numeric"]), 4))
    for i, nc in enumerate(col_analysis["numeric"][:4]):
        s = df[nc].dropna()
        with kcols[i]:
            st.markdown(
                f'<div class="kpi"><div class="kpi-accent" style="background:{accent[i]}"></div>'
                f'<div class="kpi-label">{nc.replace("_"," ").title()}</div>'
                f'<div class="kpi-value">{fmt(s.sum())}</div>'
                f'<div class="kpi-sub">Avg: {fmt(s.mean())} · Max: {fmt(s.max())}</div>'
                f'</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if active_view == "overview":
    oc1, oc2 = st.columns(2)
    with oc1:
        st.markdown('<div class="section-head">Trend / Primary Chart</div>', unsafe_allow_html=True)
        try:
            if col_analysis["date"] and col_analysis["numeric"]:
                st.plotly_chart(plot_trend(df, col_analysis), use_container_width=True)
            elif col_analysis["categorical"] and col_analysis["numeric"]:
                cat = col_analysis["categorical"][0]
                num = col_analysis["numeric"][0]
                n = df[cat].nunique()
                st.plotly_chart(plot_bar(df, cat, num, horizontal=n>6), use_container_width=True)
            else:
                st.info("Upload data with numeric columns to see charts.")
        except Exception as e:
            st.error(f"Chart error: {e}")

    with oc2:
        st.markdown('<div class="section-head">Distribution</div>', unsafe_allow_html=True)
        try:
            if col_analysis["categorical"] and col_analysis["numeric"]:
                cat = col_analysis["categorical"][0]
                num = col_analysis["numeric"][0]
                st.plotly_chart(plot_pie(df, cat, num), use_container_width=True)
        except Exception as e:
            st.error(f"Chart error: {e}")

    st.markdown('<div class="section-head">AI Executive Summary</div>', unsafe_allow_html=True)
    if "ai_exec" not in st.session_state:
        with st.spinner("Generating AI insights..."):
            st.session_state["ai_exec"] = get_ai_analysis(df_info, "executive")
    st.markdown(f'<div class="ai-box"><div class="ai-box-title">🤖 Executive Summary</div>'
                f'<div class="ai-box-text">{st.session_state["ai_exec"].replace(chr(10),"<br>")}</div></div>',
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: KPI REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif active_view == "kpis":
    st.markdown('<div class="section-head">KPI Report — All Numeric Columns</div>', unsafe_allow_html=True)
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
                        f'<div class="kpi"><div class="kpi-accent" style="background:{COLORS[j%len(COLORS)]}"></div>'
                        f'<div class="kpi-label">{nc.replace("_"," ").title()}</div>'
                        f'<div class="kpi-value" style="font-size:22px">{fmt(s.sum())}</div>'
                        f'<div class="kpi-sub">Sum total</div>'
                        f'<div style="margin-top:10px;font-size:12px;color:#64748b">'
                        f'Avg: <b>{fmt(s.mean())}</b> · Median: <b>{fmt(s.median())}</b><br>'
                        f'Min: <b>{fmt(s.min())}</b> · Max: <b>{fmt(s.max())}</b><br>'
                        f'Std Dev: <b>{fmt(s.std())}</b> · Missing: <b>{int(df[nc].isna().sum())}</b>'
                        f'</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-head">Distributions</div>', unsafe_allow_html=True)
        for batch_start in range(0, len(col_analysis["numeric"]), 3):
            batch = col_analysis["numeric"][batch_start:batch_start+3]
            cols  = st.columns(len(batch))
            for j, nc in enumerate(batch):
                with cols[j]:
                    st.markdown(f"**{nc.replace('_',' ').title()}**")
                    try:
                        st.plotly_chart(plot_histogram(df, nc), use_container_width=True)
                    except Exception as e:
                        st.error(str(e))

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TRENDS
# ══════════════════════════════════════════════════════════════════════════════
elif active_view == "trends":
    st.markdown('<div class="section-head">Time Series Analysis</div>', unsafe_allow_html=True)
    if not col_analysis["date"] or not col_analysis["numeric"]:
        st.info("Trend analysis requires at least one date and one numeric column.")
    else:
        if time_insights:
            ti = time_insights
            t1,t2,t3,t4 = st.columns(4)
            nc_label = ti["num_col"].replace("_"," ").title()
            trend_icon = "▲" if ti["growth"]>0 else "▼"
            trend_cls  = "kpi-trend-up" if ti["growth"]>0 else "kpi-trend-dn"
            with t1:
                st.markdown(f'<div class="kpi"><div class="kpi-accent" style="background:#2563eb"></div><div class="kpi-label">Overall Growth</div><div class="kpi-value"><span class="{trend_cls}">{trend_icon} {ti["growth"]:+.1f}%</span></div><div class="kpi-sub">First to last period</div></div>', unsafe_allow_html=True)
            with t2:
                st.markdown(f'<div class="kpi"><div class="kpi-accent" style="background:#10b981"></div><div class="kpi-label">Best Period</div><div class="kpi-value" style="font-size:18px">{ti["best_period"]}</div><div class="kpi-sub">{fmt(ti["best_val"])} {nc_label}</div></div>', unsafe_allow_html=True)
            with t3:
                st.markdown(f'<div class="kpi"><div class="kpi-accent" style="background:#ef4444"></div><div class="kpi-label">Worst Period</div><div class="kpi-value" style="font-size:18px">{ti["worst_period"]}</div><div class="kpi-sub">{fmt(ti["worst_val"])} {nc_label}</div></div>', unsafe_allow_html=True)
            with t4:
                st.markdown(f'<div class="kpi"><div class="kpi-accent" style="background:#f59e0b"></div><div class="kpi-label">Periods Analyzed</div><div class="kpi-value">{ti["periods"]}</div><div class="kpi-sub">Monthly periods</div></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="section-head">Full Time Series</div>', unsafe_allow_html=True)
        try:
            st.plotly_chart(plot_trend(df, col_analysis), use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

        if col_analysis["categorical"]:
            st.markdown('<div class="section-head">Heatmap by Period & Category</div>', unsafe_allow_html=True)
            try:
                st.plotly_chart(plot_heatmap(df, col_analysis), use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

        if "ai_trends" not in st.session_state:
            with st.spinner("AI trend analysis..."):
                st.session_state["ai_trends"] = get_ai_analysis(df_info, "trends")
        st.markdown(f'<div class="ai-box"><div class="ai-box-title">🤖 AI Trend Analysis</div>'
                    f'<div class="ai-box-text">{st.session_state["ai_trends"].replace(chr(10),"<br>")}</div></div>',
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CATEGORIES
# ══════════════════════════════════════════════════════════════════════════════
elif active_view == "categories":
    st.markdown('<div class="section-head">Category Analysis</div>', unsafe_allow_html=True)
    if not col_analysis["categorical"] or not col_analysis["numeric"]:
        st.info("Category analysis requires categorical and numeric columns.")
    else:
        for cat, ci in cat_insights.items():
            st.markdown(f'<div class="section-head">{cat.replace("_"," ").title()} — {ci["unique"]} unique values</div>', unsafe_allow_html=True)
            cc1,cc2,cc3,cc4 = st.columns(4)
            risk = "High" if ci["top3_share"]>70 else "Medium" if ci["top3_share"]>50 else "Low"
            badge = "badge-red" if risk=="High" else "badge-amber" if risk=="Medium" else "badge-green"
            with cc1:
                st.markdown(f'<div class="kpi"><div class="kpi-accent" style="background:#10b981"></div><div class="kpi-label">Top Performer</div><div class="kpi-value" style="font-size:16px">{ci["top"]}</div><div class="kpi-sub">{fmt(ci["top_val"])} {ci["num_col"]}</div></div>', unsafe_allow_html=True)
            with cc2:
                st.markdown(f'<div class="kpi"><div class="kpi-accent" style="background:#ef4444"></div><div class="kpi-label">Lowest Performer</div><div class="kpi-value" style="font-size:16px">{ci["bottom"]}</div><div class="kpi-sub">{fmt(ci["bottom_val"])} {ci["num_col"]}</div></div>', unsafe_allow_html=True)
            with cc3:
                st.markdown(f'<div class="kpi"><div class="kpi-accent" style="background:#f59e0b"></div><div class="kpi-label">Top 3 Share</div><div class="kpi-value">{ci["top3_share"]}%</div><div class="kpi-sub">Concentration</div></div>', unsafe_allow_html=True)
            with cc4:
                st.markdown(f'<div class="kpi"><div class="kpi-accent" style="background:#8b5cf6"></div><div class="kpi-label">Concentration Risk</div><div class="kpi-value" style="font-size:20px"><span class="badge {badge}">{risk}</span></div><div class="kpi-sub">Based on top-3 share</div></div>', unsafe_allow_html=True)

            bc1, bc2 = st.columns(2)
            with bc1:
                try:
                    n = df[cat].nunique()
                    st.plotly_chart(plot_bar(df, cat, ci["num_col"], horizontal=n>6), use_container_width=True)
                except Exception as e: st.error(str(e))
            with bc2:
                try:
                    st.plotly_chart(plot_pie(df, cat, ci["num_col"]), use_container_width=True)
                except Exception as e: st.error(str(e))

        if "ai_categories" not in st.session_state:
            with st.spinner("AI category analysis..."):
                st.session_state["ai_categories"] = get_ai_analysis(df_info, "categories")
        st.markdown(f'<div class="ai-box"><div class="ai-box-title">🤖 AI Category Analysis</div>'
                    f'<div class="ai-box-text">{st.session_state["ai_categories"].replace(chr(10),"<br>")}</div></div>',
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CORRELATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif active_view == "correlations":
    st.markdown('<div class="section-head">Correlation Analysis</div>', unsafe_allow_html=True)
    if len(col_analysis["numeric"]) < 2:
        st.info("Correlation analysis requires at least 2 numeric columns.")
    else:
        if correlations:
            st.markdown("#### Top Correlated Pairs")
            for pair in correlations[:6]:
                r = pair["r"]
                strength  = "Strong" if abs(r)>0.7 else "Moderate" if abs(r)>0.4 else "Weak"
                direction = "Positive" if r>0 else "Negative"
                badge_cls = "badge-green" if abs(r)>0.7 else "badge-amber" if abs(r)>0.4 else "badge-red"
                bar_w = int(abs(r)*100)
                bar_c = "#10b981" if r>0 else "#ef4444"
                st.markdown(
                    f'<div class="kpi" style="margin-bottom:8px">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center">'
                    f'<div><b style="color:#0f172a">{pair["col1"].replace("_"," ").title()}</b>'
                    f' ↔ <b style="color:#0f172a">{pair["col2"].replace("_"," ").title()}</b></div>'
                    f'<div><span class="badge {badge_cls}">{strength} {direction}</span>'
                    f' <b style="color:#0f172a;font-size:16px;margin-left:8px">r = {r}</b></div>'
                    f'</div><div style="margin-top:10px;background:#f1f5f9;border-radius:4px;height:6px">'
                    f'<div style="width:{bar_w}%;background:{bar_c};height:6px;border-radius:4px"></div>'
                    f'</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-head">Scatter Plots</div>', unsafe_allow_html=True)
        nums = col_analysis["numeric"]
        pairs_to_plot = [(nums[i],nums[j]) for i in range(min(3,len(nums))) for j in range(i+1,min(4,len(nums)))]
        if pairs_to_plot:
            sc_cols = st.columns(min(len(pairs_to_plot), 3))
            for idx, (c1, c2) in enumerate(pairs_to_plot[:3]):
                with sc_cols[idx]:
                    st.markdown(f"**{c1.replace('_',' ').title()} vs {c2.replace('_',' ').title()}**")
                    try:
                        st.plotly_chart(plot_scatter(df, c1, c2), use_container_width=True)
                    except Exception as e: st.error(str(e))

        st.markdown('<div class="section-head">Correlation Matrix</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-head">Anomaly Detection & Data Quality</div>', unsafe_allow_html=True)
    missing = df.isnull().sum()
    total_missing = int(missing.sum())
    total_outliers = sum(s.get("outliers",0) for s in adv_stats.values())
    dupes = int(df.duplicated().sum())

    mc1,mc2,mc3,mc4 = st.columns(4)
    with mc1:
        st.markdown(f'<div class="kpi"><div class="kpi-accent" style="background:#2563eb"></div><div class="kpi-label">Total Rows</div><div class="kpi-value">{df.shape[0]:,}</div></div>', unsafe_allow_html=True)
    with mc2:
        st.markdown(f'<div class="kpi"><div class="kpi-accent" style="background:{"#ef4444" if total_missing>0 else "#10b981"}"></div><div class="kpi-label">Missing Values</div><div class="kpi-value">{total_missing:,}</div><div class="kpi-sub">Across all columns</div></div>', unsafe_allow_html=True)
    with mc3:
        st.markdown(f'<div class="kpi"><div class="kpi-accent" style="background:#f59e0b"></div><div class="kpi-label">Outliers (IQR)</div><div class="kpi-value">{total_outliers:,}</div><div class="kpi-sub">Statistical outliers</div></div>', unsafe_allow_html=True)
    with mc4:
        st.markdown(f'<div class="kpi"><div class="kpi-accent" style="background:{"#ef4444" if dupes>0 else "#10b981"}"></div><div class="kpi-label">Duplicate Rows</div><div class="kpi-value">{dupes:,}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-head">Column Quality Report</div>', unsafe_allow_html=True)
    qual_data = []
    for col in df.columns:
        miss  = int(df[col].isna().sum())
        miss_p= round(miss/len(df)*100,1)
        qual_data.append({
            "Column": col, "Type": str(df[col].dtype),
            "Missing": miss, "Missing %": f"{miss_p}%",
            "Unique": int(df[col].nunique()),
            "Outliers": adv_stats.get(col,{}).get("outliers",0),
            "Status": "⚠️ Check" if miss_p>10 or adv_stats.get(col,{}).get("outliers",0)>5 else "✅ OK"
        })
    st.dataframe(pd.DataFrame(qual_data), use_container_width=True, hide_index=True)

    if adv_stats:
        st.markdown('<div class="section-head">Outlier Details</div>', unsafe_allow_html=True)
        has_outliers = False
        for nc, s in adv_stats.items():
            if s["outliers"] > 0:
                has_outliers = True
                st.markdown(
                    f'<div class="insight-card"><span class="insight-tag">⚠️ Outliers</span>'
                    f'<div style="font-weight:600;color:#fff;margin-bottom:6px">{nc.replace("_"," ").title()}</div>'
                    f'<div class="insight-text">{s["outliers"]} outlier(s) detected · '
                    f'IQR: [{fmt(s["q1"])} — {fmt(s["q3"])}] · '
                    f'Range: [{fmt(s["min"])} — {fmt(s["max"])}]</div></div>',
                    unsafe_allow_html=True)
        if not has_outliers:
            st.success("No significant outliers detected in numeric columns.")

    if "ai_anomalies" not in st.session_state:
        with st.spinner("AI anomaly analysis..."):
            st.session_state["ai_anomalies"] = get_ai_analysis(df_info, "anomalies")
    st.markdown(f'<div class="ai-box"><div class="ai-box-title">🤖 AI Anomaly Analysis</div>'
                f'<div class="ai-box-text">{st.session_state["ai_anomalies"].replace(chr(10),"<br>")}</div></div>',
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AI INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif active_view == "ai_insights":
    st.markdown('<div class="section-head">AI-Powered Analysis Suite</div>', unsafe_allow_html=True)
    ai_sections = [
        ("executive","🏢 Executive Summary","Full business summary with key findings"),
        ("trends","📈 Trend Analysis","Time patterns, growth rates, forecasting"),
        ("categories","📦 Category Analysis","Segment performance, concentration risk"),
        ("anomalies","🔍 Anomaly Report","Data quality issues, outliers, risk flags"),
    ]
    for key, title, desc in ai_sections:
        with st.expander(f"{title} — {desc}", expanded=(key=="executive")):
            ai_key = f"ai_{key}"
            if ai_key not in st.session_state:
                with st.spinner(f"Generating {title}..."):
                    st.session_state[ai_key] = get_ai_analysis(df_info, key)
            st.markdown(
                f'<div class="ai-box"><div class="ai-box-title">{title}</div>'
                f'<div class="ai-box-text">{st.session_state[ai_key].replace(chr(10),"<br>")}</div></div>',
                unsafe_allow_html=True)
            if st.button(f"↻ Regenerate", key=f"regen_{key}"):
                del st.session_state[ai_key]; st.rerun()

    st.markdown('<div class="section-head">Ask a Custom Question</div>', unsafe_allow_html=True)
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
            q = st.text_input("", placeholder="e.g. What are the top 3 opportunities?", label_visibility="collapsed")
        with qc2:
            sub = st.form_submit_button("Ask →", use_container_width=True)
    if sub and q:
        st.session_state.messages.append({"role":"user","content":q})
        with st.spinner("Thinking..."):
            a = ask_claude(q, df_info)
        st.session_state.messages.append({"role":"assistant","content":a})
        st.rerun()
    if st.session_state.get("messages"):
        if st.button("Clear conversation"):
            st.session_state.messages = []; st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA TABLE
# ══════════════════════════════════════════════════════════════════════════════
elif active_view == "table":
    st.markdown('<div class="section-head">Data Table</div>', unsafe_allow_html=True)
    search = st.text_input("", placeholder="🔍 Search / filter rows...", label_visibility="collapsed")
    display_df = df
    if search:
        mask = df.apply(lambda row: row.astype(str).str.contains(search, case=False, na=False).any(), axis=1)
        display_df = df[mask]
        st.caption(f"{len(display_df):,} rows matching '{search}'")
    st.dataframe(display_df, use_container_width=True, height=500)
    dl1, dl2 = st.columns(2)
    with dl1:
        csv2 = display_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download filtered CSV", csv2, f"{file_title}_filtered.csv", "text/csv", use_container_width=True)
    with dl2:
        st.caption(f"{len(display_df):,} of {len(df):,} rows shown")

else:
    st.info("Select a page from the sidebar.")
