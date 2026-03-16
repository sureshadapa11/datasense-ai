import os, json, traceback
import streamlit as st
import pandas as pd
import numpy as np
import anthropic
from datetime import datetime

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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.stApp{background:#f7f8fc;}
section[data-testid="stSidebar"]{background:#0f172a;border-right:1px solid #1e293b;}
section[data-testid="stSidebar"] *{color:#94a3b8!important;}
section[data-testid="stSidebar"] h2{color:#f1f5f9!important;}
section[data-testid="stSidebar"] .stSuccess{background:rgba(16,185,129,0.15)!important;color:#10b981!important;border-radius:8px;}
.kpi{background:#fff;border-radius:12px;padding:1.25rem 1.5rem;border:1px solid #e2e8f0;position:relative;overflow:hidden;}
.kpi-accent{position:absolute;top:0;left:0;width:4px;height:100%;border-radius:12px 0 0 12px;}
.kpi-label{font-size:11px;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:8px;}
.kpi-value{font-size:28px;font-weight:700;color:#0f172a;line-height:1;}
.kpi-sub{font-size:12px;color:#64748b;margin-top:6px;}
.kpi-trend-up{color:#10b981;font-weight:600;}
.kpi-trend-dn{color:#ef4444;font-weight:600;}
.chart-wrap{background:#fff;border-radius:12px;padding:1.25rem 1.5rem;border:1px solid #e2e8f0;margin-bottom:1rem;}
.chart-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;}
.chart-title{font-size:13px;font-weight:600;color:#0f172a;letter-spacing:0.02em;}
.chart-sub{font-size:11px;color:#94a3b8;}
.insight-card{background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);border-radius:12px;padding:1.25rem 1.5rem;color:#e2e8f0;margin-bottom:0.75rem;}
.insight-tag{display:inline-block;background:rgba(255,255,255,0.12);border-radius:20px;padding:2px 10px;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;}
.insight-text{font-size:13px;line-height:1.6;color:#cbd5e1;}
.insight-metric{font-size:20px;font-weight:700;color:#fff;margin-top:8px;}
.badge{display:inline-block;border-radius:6px;padding:2px 8px;font-size:11px;font-weight:600;}
.badge-blue{background:#dbeafe;color:#1d4ed8;}
.badge-green{background:#dcfce7;color:#15803d;}
.badge-amber{background:#fef9c3;color:#a16207;}
.badge-red{background:#fee2e2;color:#b91c1c;}
.badge-purple{background:#ede9fe;color:#6d28d9;}
.section-head{font-size:14px;font-weight:700;color:#0f172a;margin:1.5rem 0 0.75rem;text-transform:uppercase;letter-spacing:0.06em;border-left:3px solid #2563eb;padding-left:10px;}
.ai-box{background:linear-gradient(135deg,#eff6ff,#f0fdf4);border:1px solid #bfdbfe;border-radius:12px;padding:1.25rem 1.5rem;margin-bottom:0.75rem;}
.ai-box-title{font-size:12px;font-weight:700;color:#1d4ed8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;}
.ai-box-text{font-size:13px;color:#1e293b;line-height:1.7;}
.stButton>button{background:#0f172a;color:#f1f5f9;border:1px solid #1e293b;border-radius:8px;font-size:12px;font-weight:500;transition:all 0.2s;}
.stButton>button:hover{background:#1e293b;border-color:#3b82f6;color:#fff;}
div[data-testid="stFileUploader"]{background:#1e293b;border:1px dashed #334155;border-radius:10px;padding:0.75rem;}
div[data-testid="stFileUploader"] *{color:#94a3b8!important;}
div[data-testid="stFileUploader"] button{background:#2563eb!important;color:#fff!important;border:none!important;border-radius:6px!important;}
div[data-testid="stFileUploaderDropzoneInstructions"] span{color:#60a5fa!important;font-weight:600;}
.stTabs [data-baseweb="tab-list"]{background:#fff;border-radius:10px;padding:3px;border:1px solid #e2e8f0;}
.stTabs [data-baseweb="tab"]{color:#64748b;border-radius:8px;font-size:13px;font-weight:500;}
.stTabs [aria-selected="true"]{background:#0f172a;color:#fff;}
.stTextInput>div>div>input{background:#fff;color:#0f172a;border:1px solid #e2e8f0;border-radius:8px;font-size:14px;}
h1,h2,h3,h4{color:#0f172a!important;}
.dataframe{font-size:12px!important;}
</style>
""", unsafe_allow_html=True)

CHARTJS = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"
COLORS  = ['#2563eb','#10b981','#f59e0b','#ef4444','#8b5cf6',
           '#06b6d4','#f97316','#84cc16','#ec4899','#14b8a6']

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
        if pd.api.types.is_numeric_dtype(df[col]):
            is_id = (str(col).lower().strip() in id_names or
                     (df[col].nunique()==len(df) and str(col).lower().strip().endswith('id')))
            if not is_id: result["numeric"].append(col)
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            result["date"].append(col); continue
        if pd.api.types.is_object_dtype(df[col]):
            try:
                conv = pd.to_datetime(s, infer_datetime_format=True, errors='coerce')
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
        if b == 0: return 0
        return round((a - b) / abs(b) * 100, 1)
    except: return 0

def compute_advanced_stats(df, col_analysis):
    """Compute advanced statistics for all numeric columns."""
    stats = {}
    for nc in col_analysis["numeric"]:
        s = df[nc].dropna()
        if len(s) == 0: continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outliers = ((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()
        stats[nc] = {
            "sum":     s.sum(),
            "mean":    s.mean(),
            "median":  s.median(),
            "min":     s.min(),
            "max":     s.max(),
            "std":     s.std(),
            "count":   len(s),
            "missing": df[nc].isna().sum(),
            "outliers":int(outliers),
            "q1": q1, "q3": q3,
        }
    return stats

def compute_correlations(df, col_analysis):
    """Compute correlations between numeric columns."""
    nums = col_analysis["numeric"]
    if len(nums) < 2: return {}
    corr = df[nums].corr()
    pairs = []
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            r = corr.iloc[i,j]
            if not pd.isna(r):
                pairs.append({"col1": nums[i], "col2": nums[j],
                               "r": round(float(r),3)})
    pairs.sort(key=lambda x: abs(x["r"]), reverse=True)
    return pairs[:8]

def compute_category_insights(df, col_analysis):
    """Top/bottom performers and concentration for each categorical column."""
    insights = {}
    if not col_analysis["categorical"] or not col_analysis["numeric"]:
        return insights
    num_col = col_analysis["numeric"][0]
    for cat in col_analysis["categorical"][:4]:
        grp = df.groupby(cat)[num_col].sum().sort_values(ascending=False)
        total = grp.sum()
        top1  = grp.index[0] if len(grp) > 0 else "—"
        top1v = grp.iloc[0] if len(grp) > 0 else 0
        bot1  = grp.index[-1] if len(grp) > 0 else "—"
        bot1v = grp.iloc[-1] if len(grp) > 0 else 0
        top3_share = grp.head(3).sum() / total * 100 if total > 0 else 0
        insights[cat] = {
            "top": str(top1), "top_val": float(top1v),
            "bottom": str(bot1), "bottom_val": float(bot1v),
            "top3_share": round(float(top3_share), 1),
            "unique": int(df[cat].nunique()),
            "num_col": num_col,
        }
    return insights

def compute_time_insights(df, col_analysis):
    """Growth rates, best/worst periods, seasonality."""
    if not col_analysis["date"] or not col_analysis["numeric"]:
        return {}
    date_col = col_analysis["date"][0]
    num_col  = col_analysis["numeric"][0]
    df2 = df.copy()
    df2['_dt'] = pd.to_datetime(df2[date_col], errors='coerce')
    df2 = df2.dropna(subset=['_dt'])
    df2['_m'] = df2['_dt'].dt.to_period('M')
    monthly = df2.groupby('_m')[num_col].sum().sort_index()
    if len(monthly) < 2:
        return {}
    first_val = float(monthly.iloc[0])
    last_val  = float(monthly.iloc[-1])
    growth    = pct_change(last_val, first_val)
    best_m    = str(monthly.idxmax())
    worst_m   = str(monthly.idxmin())
    mom = monthly.pct_change().dropna()
    best_growth  = float(mom.max() * 100) if len(mom) > 0 else 0
    worst_growth = float(mom.min() * 100) if len(mom) > 0 else 0
    return {
        "periods": len(monthly),
        "growth": growth,
        "best_period": best_m,
        "worst_period": worst_m,
        "best_val": float(monthly.max()),
        "worst_val": float(monthly.min()),
        "best_growth_pct": round(best_growth, 1),
        "worst_growth_pct": round(worst_growth, 1),
        "num_col": num_col,
        "date_col": date_col,
    }

# ── CHART HTML BUILDERS ───────────────────────────────────────────────────────
LINE_JS = ("new Chart(document.getElementById('c'),{type:'line',data:d,"
           "options:{responsive:true,maintainAspectRatio:false,"
           "plugins:{legend:{display:true,labels:{color:'#475569',font:{size:11},boxWidth:12}}},"
           "scales:{x:{ticks:{color:'#94a3b8',font:{size:10}},grid:{display:false}},"
           "y:{ticks:{color:'#94a3b8',font:{size:10}},grid:{color:'#f1f5f9'}}}}}}});")

BAR_JS  = ("new Chart(document.getElementById('c'),{type:'bar',data:d,"
           "options:{responsive:true,maintainAspectRatio:false,"
           "plugins:{legend:{display:false}},"
           "scales:{x:{ticks:{color:'#94a3b8',font:{size:10}},grid:{display:false}},"
           "y:{ticks:{color:'#94a3b8',font:{size:10},"
           "callback:function(v){return v>=1e6?Math.round(v/1e6)+'M':v>=1000?Math.round(v/1000)+'k':v;}}"
           ",grid:{color:'#f1f5f9'}}}}}}});")

HBAR_JS = ("new Chart(document.getElementById('c'),{type:'bar',data:d,"
           "options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,"
           "plugins:{legend:{display:false}},"
           "scales:{y:{ticks:{color:'#94a3b8',font:{size:10}},grid:{display:false}},"
           "x:{ticks:{color:'#94a3b8',font:{size:10},"
           "callback:function(v){return v>=1e6?Math.round(v/1e6)+'M':v>=1000?Math.round(v/1000)+'k':v;}}"
           ",grid:{color:'#f1f5f9'}}}}}}});")

PIE_JS  = ("new Chart(document.getElementById('c'),{type:'doughnut',data:d,"
           "options:{responsive:true,maintainAspectRatio:false,"
           "plugins:{legend:{position:'right',labels:{color:'#475569',font:{size:11},padding:12}}}}}}});")

SCATTER_JS = ("new Chart(document.getElementById('c'),{type:'scatter',data:d,"
              "options:{responsive:true,maintainAspectRatio:false,"
              "plugins:{legend:{display:true,labels:{color:'#475569',font:{size:11}}}},"
              "scales:{x:{ticks:{color:'#94a3b8',font:{size:10}},grid:{color:'#f1f5f9'}},"
              "y:{ticks:{color:'#94a3b8',font:{size:10}},grid:{color:'#f1f5f9'}}}}}}});")

def html_wrap(elem, h, script):
    return (f'<!DOCTYPE html><html><head><meta charset="utf-8">'
            f'<script src="{CHARTJS}"></script>'
            f'<style>body{{margin:0;padding:4px;background:white}}'
            f'canvas{{width:100%!important}}</style>'
            f'</head><body><div style="height:{h};position:relative">{elem}</div>'
            f'<script>{script}</script></body></html>')

def chart_trend(df, col_analysis):
    dc = col_analysis["date"][0]
    ncs = col_analysis["numeric"][:5]
    df2 = df.copy()
    df2['_dt'] = pd.to_datetime(df2[dc], errors='coerce')
    df2 = df2.dropna(subset=['_dt'])
    df2['_p'] = df2['_dt'].dt.strftime('%b %Y')
    order = df2.drop_duplicates('_p').sort_values('_dt')['_p'].tolist()
    datasets = []
    for i, nc in enumerate(ncs):
        agg = df2.groupby('_p')[nc].sum().reindex(order).fillna(0)
        datasets.append({"label": nc.replace('_',' ').title(),
                          "data": [round(float(v),2) for v in agg.values],
                          "borderColor": COLORS[i], "backgroundColor": COLORS[i]+"18",
                          "fill": i==0, "tension": 0.4, "pointRadius": 3, "borderWidth": 2})
    j = json.dumps({"labels": order, "datasets": datasets})
    return html_wrap('<canvas id="c"></canvas>', "340px", "var d="+j+";"+LINE_JS)

def chart_bar(df, cat, num, top=12):
    grp = df.groupby(cat)[num].sum().sort_values(ascending=False).head(top)
    labels = [str(x).replace('"','') for x in grp.index]
    values = [round(float(v),2) for v in grp.values]
    j = json.dumps({"labels": labels, "datasets": [{"data": values,
        "backgroundColor": COLORS[:len(labels)], "borderRadius": 5}]})
    js = HBAR_JS if len(labels)>6 else BAR_JS
    h  = str(max(300, len(labels)*44+80))+"px" if len(labels)>6 else "300px"
    return html_wrap('<canvas id="c"></canvas>', h, "var d="+j+";"+js)

def chart_pie(df, cat, num):
    grp = df.groupby(cat)[num].sum().sort_values(ascending=False).head(8)
    labels = [str(x).replace('"','') for x in grp.index]
    values = [round(float(v),2) for v in grp.values]
    j = json.dumps({"labels": labels, "datasets": [{"data": values,
        "backgroundColor": COLORS[:len(labels)], "borderWidth": 2,
        "borderColor": "#fff"}]})
    return html_wrap('<canvas id="c"></canvas>', "300px", "var d="+j+";"+PIE_JS)

def chart_scatter(df, col1, col2):
    sample = df[[col1, col2]].dropna().head(500)
    points = [{"x": round(float(r[col1]),2), "y": round(float(r[col2]),2)}
              for _, r in sample.iterrows()]
    j = json.dumps({"datasets": [{"label": f"{col1} vs {col2}", "data": points,
        "backgroundColor": "#2563eb44", "borderColor": "#2563eb", "pointRadius": 5}]})
    return html_wrap('<canvas id="c"></canvas>', "300px", "var d="+j+";"+SCATTER_JS)

def chart_heatmap(df, col_analysis):
    dc  = col_analysis["date"][0]
    cat = col_analysis["categorical"][0]
    num = col_analysis["numeric"][0]
    df2 = df.copy()
    df2['_dt'] = pd.to_datetime(df2[dc], errors='coerce')
    df2['_p']  = df2['_dt'].dt.strftime('%b')
    order = df2.drop_duplicates('_p').sort_values('_dt')['_p'].tolist()
    cats  = df2[cat].dropna().unique().tolist()[:7]
    hdata = {str(c).replace('"',''): [round(float(df2[df2[cat]==c].groupby('_p')[num].sum().get(p,0)),2) for p in order] for c in cats}
    j = json.dumps({"periods": order, "cats": [str(c).replace('"','') for c in cats], "data": hdata})
    script = ("var d="+j+";"
        "var all=[];d.cats.forEach(c=>d.data[c].forEach(v=>all.push(v)));"
        "var mn=Math.min(...all.filter(v=>v>0)||[0]),mx=Math.max(...all)||1;"
        "function bg(v){var t=v===0?0:(v-mn)/(mx-mn);return 'rgba(37,99,235,'+((t*0.85)+0.08)+')';}  "
        "function fg(v){return (v-mn)/(mx-mn)>0.45?'#fff':'#1e293b';}"
        "var t=document.getElementById('c');"
        "var h='<tr><th style=\"padding:8px 12px;background:#f8faff;color:#94a3b8;border:1px solid #e2e8f0;font-size:12px\">Period</th>'"
        "+d.cats.map(c=>'<th style=\"padding:8px 12px;background:#f8faff;color:#334155;border:1px solid #e2e8f0;font-size:12px;font-weight:600\">'+c+'</th>').join('')+'<th style=\"padding:8px 12px;background:#f8faff;color:#94a3b8;border:1px solid #e2e8f0;font-size:12px\">Total</th></tr>';"
        "d.periods.forEach((p,i)=>{"
        "var tot=d.cats.reduce((s,c)=>s+(d.data[c][i]||0),0);"
        "h+='<tr><td style=\"padding:8px 12px;color:#64748b;border:1px solid #e2e8f0;font-size:12px;font-weight:500\">'+p+'</td>';"
        "d.cats.forEach(c=>{var v=d.data[c][i]||0;"
        "h+='<td style=\"padding:8px 12px;text-align:center;border:1px solid #e2e8f0;font-size:12px;font-weight:500;background:'+bg(v)+';color:'+fg(v)+'\">'+Math.round(v).toLocaleString()+'</td>';});"
        "h+='<td style=\"padding:8px 12px;text-align:center;color:#64748b;border:1px solid #e2e8f0;font-size:12px\">'+Math.round(tot).toLocaleString()+'</td></tr>';});"
        "t.innerHTML=h;")
    return html_wrap('<table id="c" style="width:100%;border-collapse:collapse"></table>', "auto", script)

def chart_histogram(df, num_col, bins=20):
    s = df[num_col].dropna()
    counts, edges = np.histogram(s, bins=bins)
    labels = [f"{edges[i]:.0f}-{edges[i+1]:.0f}" for i in range(len(counts))]
    j = json.dumps({"labels": labels, "datasets": [{"data": counts.tolist(),
        "backgroundColor": "#2563eb66", "borderColor": "#2563eb",
        "borderWidth": 1, "borderRadius": 3}]})
    return html_wrap('<canvas id="c"></canvas>', "260px", "var d="+j+";"+BAR_JS)

# ── AI ANALYSIS ───────────────────────────────────────────────────────────────
def get_ai_analysis(df_info, analysis_type="full"):
    try:
        client = anthropic.Anthropic()
        prompts = {
            "executive": (
                "You are a senior data analyst writing an executive summary. "
                "Based on the dataset, provide:\n"
                "1. A 2-sentence executive summary\n"
                "2. Top 3 key findings (with specific numbers)\n"
                "3. Top 3 risks or areas of concern\n"
                "4. Top 3 recommended actions\n"
                "Be specific and use real numbers from the data. Keep it concise."
            ),
            "trends": (
                "You are a data analyst. Analyze the time trends in this dataset and provide:\n"
                "1. Overall trend direction (growing/declining/stable)\n"
                "2. Key inflection points or anomalies\n"
                "3. Seasonality patterns if visible\n"
                "4. Forecast direction for next period\n"
                "Use specific numbers. Keep responses short and factual."
            ),
            "categories": (
                "You are a data analyst. Analyze the categorical breakdowns and provide:\n"
                "1. Top performing segments and why they stand out\n"
                "2. Underperforming segments that need attention\n"
                "3. Concentration risk (is revenue too dependent on one segment?)\n"
                "4. Quick wins — where to focus for maximum impact\n"
                "Use specific numbers. Be concise."
            ),
            "anomalies": (
                "You are a data analyst. Find anomalies and outliers in this dataset:\n"
                "1. Statistical outliers in numeric columns\n"
                "2. Unusual patterns or unexpected values\n"
                "3. Data quality issues (missing values, inconsistencies)\n"
                "4. Risk flags that need investigation\n"
                "Be specific and actionable."
            ),
        }
        system = prompts.get(analysis_type, prompts["executive"])
        response = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=600,
            system=system,
            messages=[{"role":"user","content":f"Dataset info:\n{df_info}"}]
        )
        return response.content[0].text
    except anthropic.AuthenticationError:
        return "Auth error — check your API key in Streamlit secrets."
    except Exception as e:
        return f"Error generating analysis: {e}"

def ask_claude(question, df_info):
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=800,
            system=(f"You are an expert data analyst with Power BI expertise.\n"
                    f"Dataset:\n{df_info}\n"
                    f"Give precise, actionable answers with specific numbers. "
                    f"Format clearly with bullet points where appropriate."),
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
    st.markdown("""
    <div style="max-width:700px">
    <h1 style="font-size:32px;font-weight:700;color:#0f172a;margin-bottom:8px">Power BI Style Analytics 🧠</h1>
    <p style="font-size:16px;color:#64748b;margin-bottom:2rem">Upload any CSV or Excel file for instant AI-powered analysis — KPIs, charts, trends, correlations, anomalies and executive insights.</p>
    </div>
    """, unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    for col_obj, icon, title, desc, color in [
        (c1,"🛒","Sales","Revenue, products, regions","#dbeafe"),
        (c2,"💰","Finance","Budget, expenses, profit","#dcfce7"),
        (c3,"👥","HR","Employees, salary, KPIs","#ede9fe"),
        (c4,"📊","Any Data","Any CSV or Excel file","#fef9c3"),
    ]:
        with col_obj:
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

col_analysis = analyze_columns(df)
dtype        = detect_type(df)
info         = DATASET_TYPES[dtype]
adv_stats    = compute_advanced_stats(df, col_analysis)
correlations = compute_correlations(df, col_analysis)
cat_insights = compute_category_insights(df, col_analysis)
time_insights= compute_time_insights(df, col_analysis)

# ── SIDEBAR NAV ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown(f"<p style='color:#60a5fa!important;font-size:11px;font-weight:700'>{info['icon']} {info['label'].upper()}</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#475569!important;font-size:10px;text-transform:uppercase;letter-spacing:0.1em;font-weight:600'>PAGES</p>", unsafe_allow_html=True)

    pages = [("🏠 Overview", "overview"),("📊 KPI Report", "kpis"),
             ("📈 Trends", "trends"),("📦 Categories", "categories"),
             ("🔗 Correlations", "correlations"),("🔍 Anomalies", "anomalies"),
             ("🤖 AI Insights", "ai_insights"),("📋 Data Table", "table")]

    if not col_analysis["date"] or not col_analysis["numeric"]:
        pages = [p for p in pages if p[1] != "trends"]
    if not col_analysis["categorical"]:
        pages = [p for p in pages if p[1] != "categories"]
    if len(col_analysis["numeric"]) < 2:
        pages = [p for p in pages if p[1] != "correlations"]

    for label, key in pages:
        if st.button(label, key=f"pg_{key}", use_container_width=True):
            st.session_state["active_view"] = key

    st.markdown("---")
    st.markdown(f"<p style='color:#475569!important;font-size:11px'><b style='color:#94a3b8!important'>{df.shape[0]:,}</b> rows · <b style='color:#94a3b8!important'>{df.shape[1]}</b> cols<br><b style='color:#94a3b8!important'>{len(col_analysis['numeric'])}</b> numeric · <b style='color:#94a3b8!important'>{len(col_analysis['categorical'])}</b> categorical · <b style='color:#94a3b8!important'>{len(col_analysis['date'])}</b> date</p>", unsafe_allow_html=True)

active_view = st.session_state.get("active_view", "overview")

# ── PAGE HEADER ───────────────────────────────────────────────────────────────
file_title = (uploaded.name.replace(".csv","").replace(".xlsx","").replace(".xls","")
              .replace("_"," ").replace("-"," ").title())
col_h1, col_h2 = st.columns([3,1])
with col_h1:
    st.markdown(f"<h3 style='margin-bottom:2px'>{info['icon']} {file_title}</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#64748b;font-size:12px;margin-bottom:1rem'>{dtype.title()} Dataset · {df.shape[0]:,} rows · {df.shape[1]} columns · Last updated: {datetime.now().strftime('%d %b %Y')}</p>", unsafe_allow_html=True)
with col_h2:
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇ Export CSV", csv, f"{file_title}.csv", "text/csv", use_container_width=True)

df_info = (f"File: {uploaded.name} | Type: {dtype} | Shape: {df.shape[0]} rows x {df.shape[1]} cols\n"
           f"Columns: {', '.join(df.columns.tolist())}\n"
           f"Numeric cols: {', '.join(col_analysis['numeric'])}\n"
           f"Categorical cols: {', '.join(col_analysis['categorical'])}\n"
           f"Date cols: {', '.join(col_analysis['date'])}\n"
           f"Stats:\n{df.describe().to_string()}\nSample:\n{df.head(3).to_string()}")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if active_view == "overview":
    # KPI row
    if col_analysis["numeric"]:
        num_show = min(len(col_analysis["numeric"]), 4)
        kpi_cols = st.columns(num_show)
        accent_colors = ["#2563eb","#10b981","#f59e0b","#8b5cf6"]
        for i, nc in enumerate(col_analysis["numeric"][:4]):
            s = df[nc].dropna()
            with kpi_cols[i]:
                st.markdown(
                    f'<div class="kpi">'
                    f'<div class="kpi-accent" style="background:{accent_colors[i]}"></div>'
                    f'<div class="kpi-label">{nc.replace("_"," ").title()}</div>'
                    f'<div class="kpi-value">{fmt(s.sum())}</div>'
                    f'<div class="kpi-sub">Avg: {fmt(s.mean())} · Max: {fmt(s.max())}</div>'
                    f'</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # main charts
    oc1, oc2 = st.columns(2)
    with oc1:
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        if col_analysis["date"] and col_analysis["numeric"]:
            st.markdown('<div class="chart-head"><span class="chart-title">Trend Over Time</span><span class="chart-sub">Monthly aggregation</span></div>', unsafe_allow_html=True)
            try: st.components.v1.html(chart_trend(df, col_analysis), height=360)
            except Exception as e: st.error(f"Trend error: {e}")
        elif col_analysis["categorical"] and col_analysis["numeric"]:
            cat, num = col_analysis["categorical"][0], col_analysis["numeric"][0]
            st.markdown(f'<div class="chart-head"><span class="chart-title">{cat.title()} Breakdown</span></div>', unsafe_allow_html=True)
            try:
                n = df[cat].nunique()
                st.components.v1.html(chart_bar(df, cat, num), height=max(300, min(n*40+80,400)))
            except Exception as e: st.error(f"Bar error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    with oc2:
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        if col_analysis["categorical"] and col_analysis["numeric"]:
            cat, num = col_analysis["categorical"][0], col_analysis["numeric"][0]
            st.markdown(f'<div class="chart-head"><span class="chart-title">{cat.title()} Distribution</span></div>', unsafe_allow_html=True)
            try: st.components.v1.html(chart_pie(df, cat, num), height=320)
            except Exception as e: st.error(f"Pie error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    # quick AI summary
    st.markdown('<div class="section-head">AI Executive Summary</div>', unsafe_allow_html=True)
    ai_key = "ai_exec"
    if ai_key not in st.session_state:
        with st.spinner("Generating AI insights..."):
            st.session_state[ai_key] = get_ai_analysis(df_info, "executive")
    st.markdown(f'<div class="ai-box"><div class="ai-box-title">🤖 Executive Summary</div><div class="ai-box-text">{st.session_state[ai_key].replace(chr(10),"<br>")}</div></div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: KPI REPORT
# ═════════════════════════════════════════════════════════════════════════════
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
                        f'<div class="kpi">'
                        f'<div class="kpi-accent" style="background:{COLORS[j%len(COLORS)]}"></div>'
                        f'<div class="kpi-label">{nc.replace("_"," ").title()}</div>'
                        f'<div class="kpi-value" style="font-size:22px">{fmt(s.sum())}</div>'
                        f'<div class="kpi-sub">Sum (total)</div>'
                        f'<div style="margin-top:10px;font-size:12px;color:#64748b">'
                        f'Avg: <b>{fmt(s.mean())}</b><br>'
                        f'Median: <b>{fmt(s.median())}</b><br>'
                        f'Min: <b>{fmt(s.min())}</b> · Max: <b>{fmt(s.max())}</b><br>'
                        f'Std Dev: <b>{fmt(s.std())}</b><br>'
                        f'Missing: <b>{df[nc].isna().sum()}</b>'
                        f'</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # histograms
        st.markdown('<div class="section-head">Distributions</div>', unsafe_allow_html=True)
        for batch_start in range(0, len(col_analysis["numeric"]), 3):
            batch = col_analysis["numeric"][batch_start:batch_start+3]
            cols  = st.columns(len(batch))
            for j, nc in enumerate(batch):
                with cols[j]:
                    st.markdown(f'<div class="chart-wrap"><div class="chart-title">{nc.replace("_"," ").title()}</div>', unsafe_allow_html=True)
                    try: st.components.v1.html(chart_histogram(df, nc), height=280)
                    except Exception as e: st.error(str(e))
                    st.markdown('</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: TRENDS
# ═════════════════════════════════════════════════════════════════════════════
elif active_view == "trends":
    st.markdown('<div class="section-head">Time Series Analysis</div>', unsafe_allow_html=True)
    if not col_analysis["date"] or not col_analysis["numeric"]:
        st.info("Trend analysis requires at least one date and one numeric column.")
    else:
        # time insights cards
        if time_insights:
            ti = time_insights
            nc_label = ti["num_col"].replace("_"," ").title()
            t1,t2,t3,t4 = st.columns(4)
            with t1:
                trend_icon = "▲" if ti["growth"] > 0 else "▼"
                trend_cls  = "kpi-trend-up" if ti["growth"] > 0 else "kpi-trend-dn"
                st.markdown(f'<div class="kpi"><div class="kpi-accent" style="background:#2563eb"></div><div class="kpi-label">Overall Growth</div><div class="kpi-value"><span class="{trend_cls}">{trend_icon} {ti["growth"]:+.1f}%</span></div><div class="kpi-sub">First to last period</div></div>', unsafe_allow_html=True)
            with t2:
                st.markdown(f'<div class="kpi"><div class="kpi-accent" style="background:#10b981"></div><div class="kpi-label">Best Period</div><div class="kpi-value" style="font-size:18px">{ti["best_period"]}</div><div class="kpi-sub">{fmt(ti["best_val"])} {nc_label}</div></div>', unsafe_allow_html=True)
            with t3:
                st.markdown(f'<div class="kpi"><div class="kpi-accent" style="background:#ef4444"></div><div class="kpi-label">Worst Period</div><div class="kpi-value" style="font-size:18px">{ti["worst_period"]}</div><div class="kpi-sub">{fmt(ti["worst_val"])} {nc_label}</div></div>', unsafe_allow_html=True)
            with t4:
                st.markdown(f'<div class="kpi"><div class="kpi-accent" style="background:#f59e0b"></div><div class="kpi-label">Periods Analyzed</div><div class="kpi-value">{ti["periods"]}</div><div class="kpi-sub">Monthly periods</div></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        # trend chart
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.markdown('<div class="chart-head"><span class="chart-title">Full Time Series</span><span class="chart-sub">All numeric metrics over time</span></div>', unsafe_allow_html=True)
        try: st.components.v1.html(chart_trend(df, col_analysis), height=400)
        except Exception as e: st.error(f"Error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

        # heatmap if categorical available
        if col_analysis["categorical"]:
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.markdown('<div class="chart-head"><span class="chart-title">Heatmap by Period & Category</span></div>', unsafe_allow_html=True)
            try: st.components.v1.html(chart_heatmap(df, col_analysis), height=480)
            except Exception as e: st.error(f"Error: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        # AI trend analysis
        ai_key = "ai_trends"
        if ai_key not in st.session_state:
            with st.spinner("Analyzing trends with AI..."):
                st.session_state[ai_key] = get_ai_analysis(df_info, "trends")
        st.markdown('<div class="ai-box"><div class="ai-box-title">🤖 AI Trend Analysis</div>'
                    f'<div class="ai-box-text">{st.session_state["ai_trends"].replace(chr(10),"<br>")}</div></div>',
                    unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: CATEGORIES
# ═════════════════════════════════════════════════════════════════════════════
elif active_view == "categories":
    st.markdown('<div class="section-head">Category Analysis</div>', unsafe_allow_html=True)
    if not col_analysis["categorical"] or not col_analysis["numeric"]:
        st.info("Category analysis requires categorical and numeric columns.")
    else:
        # category insight cards
        for cat, ci in cat_insights.items():
            st.markdown(f'<div class="section-head">{cat.replace("_"," ").title()} — {ci["unique"]} unique values</div>', unsafe_allow_html=True)
            cc1,cc2,cc3,cc4 = st.columns(4)
            with cc1:
                st.markdown(f'<div class="kpi"><div class="kpi-accent" style="background:#10b981"></div><div class="kpi-label">Top Performer</div><div class="kpi-value" style="font-size:16px">{ci["top"]}</div><div class="kpi-sub">{fmt(ci["top_val"])} {ci["num_col"]}</div></div>', unsafe_allow_html=True)
            with cc2:
                st.markdown(f'<div class="kpi"><div class="kpi-accent" style="background:#ef4444"></div><div class="kpi-label">Lowest Performer</div><div class="kpi-value" style="font-size:16px">{ci["bottom"]}</div><div class="kpi-sub">{fmt(ci["bottom_val"])} {ci["num_col"]}</div></div>', unsafe_allow_html=True)
            with cc3:
                st.markdown(f'<div class="kpi"><div class="kpi-accent" style="background:#f59e0b"></div><div class="kpi-label">Top 3 Share</div><div class="kpi-value">{ci["top3_share"]}%</div><div class="kpi-sub">Concentration risk</div></div>', unsafe_allow_html=True)
            with cc4:
                risk = "High" if ci["top3_share"] > 70 else "Medium" if ci["top3_share"] > 50 else "Low"
                badge = "badge-red" if risk=="High" else "badge-amber" if risk=="Medium" else "badge-green"
                st.markdown(f'<div class="kpi"><div class="kpi-accent" style="background:#8b5cf6"></div><div class="kpi-label">Concentration Risk</div><div class="kpi-value" style="font-size:20px"><span class="badge {badge}">{risk}</span></div><div class="kpi-sub">Based on top-3 share</div></div>', unsafe_allow_html=True)

            bc1, bc2 = st.columns(2)
            with bc1:
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                st.markdown(f'<div class="chart-head"><span class="chart-title">Bar Chart</span></div>', unsafe_allow_html=True)
                try:
                    n = df[cat].nunique()
                    st.components.v1.html(chart_bar(df, cat, ci["num_col"]), height=max(280, min(n*42+80, 550)))
                except Exception as e: st.error(str(e))
                st.markdown('</div>', unsafe_allow_html=True)
            with bc2:
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                st.markdown(f'<div class="chart-head"><span class="chart-title">Pie Distribution</span></div>', unsafe_allow_html=True)
                try: st.components.v1.html(chart_pie(df, cat, ci["num_col"]), height=300)
                except Exception as e: st.error(str(e))
                st.markdown('</div>', unsafe_allow_html=True)

        ai_key = "ai_categories"
        if ai_key not in st.session_state:
            with st.spinner("Analyzing categories with AI..."):
                st.session_state[ai_key] = get_ai_analysis(df_info, "categories")
        st.markdown('<div class="ai-box"><div class="ai-box-title">🤖 AI Category Analysis</div>'
                    f'<div class="ai-box-text">{st.session_state["ai_categories"].replace(chr(10),"<br>")}</div></div>',
                    unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: CORRELATIONS
# ═════════════════════════════════════════════════════════════════════════════
elif active_view == "correlations":
    st.markdown('<div class="section-head">Correlation Analysis</div>', unsafe_allow_html=True)
    if len(col_analysis["numeric"]) < 2:
        st.info("Correlation analysis requires at least 2 numeric columns.")
    else:
        if correlations:
            st.markdown("#### Top Correlated Column Pairs")
            for pair in correlations[:6]:
                r = pair["r"]
                strength = "Strong" if abs(r)>0.7 else "Moderate" if abs(r)>0.4 else "Weak"
                direction = "Positive" if r>0 else "Negative"
                badge_cls = "badge-green" if abs(r)>0.7 else "badge-amber" if abs(r)>0.4 else "badge-red"
                bar_w = int(abs(r)*100)
                bar_c = "#10b981" if r>0 else "#ef4444"
                st.markdown(
                    f'<div class="kpi" style="margin-bottom:8px">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center">'
                    f'<div><b style="color:#0f172a">{pair["col1"].replace("_"," ").title()}</b>'
                    f' <span style="color:#94a3b8;font-size:12px">↔</span> '
                    f'<b style="color:#0f172a">{pair["col2"].replace("_"," ").title()}</b></div>'
                    f'<div><span class="badge {badge_cls}">{strength} {direction}</span>'
                    f' <b style="color:#0f172a;font-size:16px;margin-left:8px">r = {r}</b></div>'
                    f'</div>'
                    f'<div style="margin-top:10px;background:#f1f5f9;border-radius:4px;height:6px">'
                    f'<div style="width:{bar_w}%;background:{bar_c};height:6px;border-radius:4px"></div></div>'
                    f'</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-head">Scatter Plots</div>', unsafe_allow_html=True)
        nums = col_analysis["numeric"]
        pairs_to_plot = [(nums[i], nums[j]) for i in range(min(3,len(nums))) for j in range(i+1, min(4,len(nums)))]
        if pairs_to_plot:
            sc_cols = st.columns(min(len(pairs_to_plot), 3))
            for idx, (c1, c2) in enumerate(pairs_to_plot[:3]):
                with sc_cols[idx]:
                    st.markdown(f'<div class="chart-wrap"><div class="chart-title">{c1.replace("_"," ").title()} vs {c2.replace("_"," ").title()}</div>', unsafe_allow_html=True)
                    try: st.components.v1.html(chart_scatter(df, c1, c2), height=280)
                    except Exception as e: st.error(str(e))
                    st.markdown('</div>', unsafe_allow_html=True)

        # full correlation matrix as table
        st.markdown('<div class="section-head">Correlation Matrix</div>', unsafe_allow_html=True)
        corr_df = df[col_analysis["numeric"]].corr().round(3)
        st.dataframe(corr_df.style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1), use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ANOMALIES
# ═════════════════════════════════════════════════════════════════════════════
elif active_view == "anomalies":
    st.markdown('<div class="section-head">Anomaly Detection & Data Quality</div>', unsafe_allow_html=True)

    # missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    has_missing = missing[missing > 0]

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(f'<div class="kpi"><div class="kpi-accent" style="background:#2563eb"></div><div class="kpi-label">Total Rows</div><div class="kpi-value">{df.shape[0]:,}</div></div>', unsafe_allow_html=True)
    with mc2:
        total_missing = int(missing.sum())
        st.markdown(f'<div class="kpi"><div class="kpi-accent" style="background:{"#ef4444" if total_missing>0 else "#10b981"}"></div><div class="kpi-label">Missing Values</div><div class="kpi-value">{total_missing:,}</div><div class="kpi-sub">Across all columns</div></div>', unsafe_allow_html=True)
    with mc3:
        total_outliers = sum(s.get("outliers",0) for s in adv_stats.values())
        st.markdown(f'<div class="kpi"><div class="kpi-accent" style="background:#f59e0b"></div><div class="kpi-label">Outliers Detected</div><div class="kpi-value">{total_outliers:,}</div><div class="kpi-sub">IQR method</div></div>', unsafe_allow_html=True)
    with mc4:
        dupes = df.duplicated().sum()
        st.markdown(f'<div class="kpi"><div class="kpi-accent" style="background:{"#ef4444" if dupes>0 else "#10b981"}"></div><div class="kpi-label">Duplicate Rows</div><div class="kpi-value">{int(dupes):,}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # per-column stats
    st.markdown('<div class="section-head">Column Quality Report</div>', unsafe_allow_html=True)
    qual_data = []
    for col in df.columns:
        miss  = int(df[col].isna().sum())
        miss_p = round(miss/len(df)*100,1)
        uniq  = int(df[col].nunique())
        dtype = str(df[col].dtype)
        out   = adv_stats.get(col, {}).get("outliers", 0)
        qual_data.append({
            "Column": col,
            "Type": dtype,
            "Missing": miss,
            "Missing %": f"{miss_p}%",
            "Unique": uniq,
            "Outliers": out,
            "Status": "⚠️ Check" if miss_p>10 or out>5 else "✅ OK"
        })
    qual_df = pd.DataFrame(qual_data)
    st.dataframe(qual_df, use_container_width=True, hide_index=True)

    # outlier details for numeric cols
    if adv_stats:
        st.markdown('<div class="section-head">Numeric Column Outlier Details</div>', unsafe_allow_html=True)
        for nc, s in adv_stats.items():
            if s["outliers"] > 0:
                st.markdown(
                    f'<div class="insight-card">'
                    f'<span class="insight-tag">⚠️ Outliers</span>'
                    f'<div style="font-weight:600;color:#fff;margin-bottom:6px">{nc.replace("_"," ").title()}</div>'
                    f'<div class="insight-text">'
                    f'{s["outliers"]} outlier{"s" if s["outliers"]>1 else ""} detected · '
                    f'IQR: [{fmt(s["q1"])} — {fmt(s["q3"])}] · '
                    f'Range: [{fmt(s["min"])} — {fmt(s["max"])}]'
                    f'</div></div>', unsafe_allow_html=True)

    ai_key = "ai_anomalies"
    if ai_key not in st.session_state:
        with st.spinner("AI anomaly analysis..."):
            st.session_state[ai_key] = get_ai_analysis(df_info, "anomalies")
    st.markdown('<div class="ai-box"><div class="ai-box-title">🤖 AI Anomaly Analysis</div>'
                f'<div class="ai-box-text">{st.session_state["ai_anomalies"].replace(chr(10),"<br>")}</div></div>',
                unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: AI INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
elif active_view == "ai_insights":
    st.markdown('<div class="section-head">AI-Powered Analysis Suite</div>', unsafe_allow_html=True)
    st.markdown("Full AI analysis of your dataset across 4 dimensions — executive summary, trends, categories, and anomalies.")
    st.markdown("<br>", unsafe_allow_html=True)

    ai_sections = [
        ("executive","🏢 Executive Summary","Full business summary with key findings and recommendations"),
        ("trends","📈 Trend Analysis","Time-based patterns, growth rates and forecasting direction"),
        ("categories","📦 Category Analysis","Segment performance, concentration risk and quick wins"),
        ("anomalies","🔍 Anomaly Report","Data quality issues, outliers and risk flags"),
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
                del st.session_state[ai_key]
                st.rerun()

    # ask anything
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
            q = st.text_input("", placeholder="e.g. What are the top 3 opportunities in this data?", label_visibility="collapsed")
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
            st.session_state.messages = []
            st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: DATA TABLE
# ═════════════════════════════════════════════════════════════════════════════
elif active_view == "table":
    st.markdown('<div class="section-head">Data Table</div>', unsafe_allow_html=True)
    search = st.text_input("", placeholder="🔍 Search / filter rows...", label_visibility="collapsed")
    display_df = df
    if search:
        mask = df.apply(lambda row: row.astype(str).str.contains(search, case=False, na=False).any(), axis=1)
        display_df = df[mask]
        st.markdown(f"<p style='color:#64748b;font-size:12px'>{len(display_df):,} rows matching '{search}'</p>", unsafe_allow_html=True)
    st.dataframe(display_df, use_container_width=True, height=500)
    c1, c2 = st.columns(2)
    with c1:
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download filtered CSV", csv, f"{file_title}_filtered.csv", "text/csv", use_container_width=True)
    with c2:
        st.markdown(f"<p style='color:#64748b;font-size:12px;margin-top:0.5rem'>{len(display_df):,} of {len(df):,} rows shown</p>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# DEFAULT
# ═════════════════════════════════════════════════════════════════════════════
else:
    st.info("Select a page from the sidebar navigation.")
