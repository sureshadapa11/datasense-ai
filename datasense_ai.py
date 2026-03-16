import os, json, re
import streamlit as st
import pandas as pd
import numpy as np
import anthropic

os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"] if hasattr(st, 'secrets') and "ANTHROPIC_API_KEY" in st.secrets else os.environ.get("ANTHROPIC_API_KEY","your-api-key-here")

st.set_page_config(page_title="DataSense AI", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#f0f4ff;}
section[data-testid="stSidebar"]{background:linear-gradient(160deg,#1e3a5f 0%,#2563eb 100%);border-right:none;}
section[data-testid="stSidebar"] *{color:#fff!important;}
section[data-testid="stSidebar"] .stMarkdown p{color:rgba(255,255,255,0.75)!important;}
.kpi-card{background:#fff;border-radius:16px;padding:1.25rem 1.5rem;box-shadow:0 2px 12px rgba(37,99,235,0.08);margin-bottom:0.5rem;}
.kpi-label{font-size:11px;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;}
.kpi-value{font-size:26px;font-weight:700;}
.kpi-sub{font-size:11px;color:#94a3b8;margin-top:3px;}
.chart-card{background:#fff;border-radius:16px;padding:1.25rem;box-shadow:0 2px 12px rgba(37,99,235,0.06);margin-bottom:1rem;}
.chart-title{font-size:13px;font-weight:600;color:#475569;margin-bottom:0.75rem;text-transform:uppercase;letter-spacing:0.05em;}
.dataset-badge{display:inline-block;background:rgba(255,255,255,0.2);border:1px solid rgba(255,255,255,0.4);border-radius:20px;padding:3px 12px;font-size:11px;font-weight:600;color:#fff;margin-bottom:1rem;}
.stButton>button{background:rgba(255,255,255,0.15);color:#fff;border:1px solid rgba(255,255,255,0.3);border-radius:10px;font-size:12px;transition:all 0.2s;}
.stButton>button:hover{background:rgba(255,255,255,0.3);border-color:rgba(255,255,255,0.7);}
.stButton>button[data-active="true"]{background:rgba(255,255,255,0.35);border-color:#fff;}
div[data-testid="stFileUploader"]{background:#fff;border:2px dashed #2563eb;border-radius:12px;padding:0.5rem;}
div[data-testid="stFileUploader"] *{color:#1e293b!important;}
div[data-testid="stFileUploader"] button{background:#2563eb!important;color:#fff!important;border:none!important;border-radius:8px!important;}
div[data-testid="stFileUploaderDropzoneInstructions"] span{color:#2563eb!important;font-weight:600;}
.stTabs [data-baseweb="tab-list"]{background:#fff;border-radius:12px;padding:4px;box-shadow:0 1px 6px rgba(0,0,0,0.06);}
.stTabs [data-baseweb="tab"]{color:#64748b;border-radius:8px;font-weight:500;}
.stTabs [aria-selected="true"]{background:#2563eb;color:#fff;}
.stTextInput>div>div>input{background:#fff;color:#1e293b;border:1px solid #e2e8f0;border-radius:10px;}
.stTextInput>div>div>input:focus{border-color:#2563eb;box-shadow:0 0 0 3px rgba(37,99,235,0.15);}
h1,h2,h3{color:#1e293b!important;}
</style>
""", unsafe_allow_html=True)

CHARTJS = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"
P = "String.fromCharCode(163)"

DATASET_TYPES = {
    "sales":    {"icon":"🛒","color":"#2563eb","keywords":["revenue","sales","product","region","order","customer","discount","profit","units","sold"]},
    "finance":  {"icon":"💰","color":"#059669","keywords":["expense","budget","cashflow","profit","loss","asset","liability","income","cost","invoice"]},
    "hr":       {"icon":"👥","color":"#7c3aed","keywords":["employee","salary","department","hire","leave","performance","headcount","staff","payroll","role"]},
    "marketing":{"icon":"📣","color":"#db2777","keywords":["campaign","click","impression","conversion","lead","channel","ctr","cpc","roas","audience"]},
    "inventory":{"icon":"📦","color":"#d97706","keywords":["stock","inventory","sku","warehouse","supplier","reorder","quantity","item","shelf"]},
    "generic":  {"icon":"📊","color":"#475569","keywords":[]},
}

def detect_dataset_type(df):
    cols = ' '.join(df.columns.tolist()).lower()
    scores = {k: sum(1 for w in v["keywords"] if w in cols) for k,v in DATASET_TYPES.items() if k != "generic"}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "generic"

def analyze_columns(df):
    analysis = {"numeric":[], "categorical":[], "date":[], "text":[]}
    for col in df.columns:
        try:
            converted = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
            if converted.notna().sum() > len(df) * 0.6:
                analysis["date"].append(col); continue
        except: pass
        if pd.api.types.is_numeric_dtype(df[col]):
            analysis["numeric"].append(col)
        elif df[col].nunique() <= max(20, len(df)*0.3):
            analysis["categorical"].append(col)
        else:
            analysis["text"].append(col)
    return analysis

def build_sidebar_menu(df, dtype, col_analysis):
    info = DATASET_TYPES[dtype]
    menu = []
    if col_analysis["date"] and col_analysis["numeric"]:
        menu.append(("📈 Trend Over Time", "trend"))
    for cat in col_analysis["categorical"][:4]:
        num = col_analysis["numeric"][0] if col_analysis["numeric"] else None
        if num:
            label = f"{cat.replace('_',' ').title()} Analysis"
            icons = {"product":"📦","region":"🌍","department":"🏢","category":"🍩","channel":"📣",
                     "status":"🔖","type":"🏷️","country":"🌐","team":"👥","brand":"🎯"}
            icon = next((v for k,v in icons.items() if k in cat.lower()), "📊")
            menu.append((f"{icon} {label}", f"bar_{cat}"))
    if col_analysis["categorical"] and col_analysis["numeric"]:
        menu.append(("🍩 Distribution", "pie"))
    if col_analysis["date"] and len(col_analysis["categorical"]) >= 2 and col_analysis["numeric"]:
        menu.append(("🔥 Heatmap", "heatmap"))
    menu.append(("📋 Raw Data", "table"))
    return menu

def make_chart(ctype, data_json, options=""):
    j = json.dumps(data_json) if not isinstance(data_json, str) else data_json
    script = f"var d={j};"
    elem = '<canvas id="c"></canvas>'
    height = "320px"

    LINE_OPTS   = "new Chart(document.getElementById('c'),{type:'line',data:d,options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:true,labels:{color:'#475569',font:{size:11}}}},scales:{x:{ticks:{color:'#94a3b8',font:{size:11}},grid:{display:false}},y:{ticks:{color:'#94a3b8',font:{size:11}},grid:{color:'#f1f5f9'}}}}}}});"
    BAR_OPTS    = "new Chart(document.getElementById('c'),{type:'bar',data:d,options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{ticks:{color:'#94a3b8',font:{size:11}},grid:{display:false}},y:{ticks:{color:'#94a3b8',font:{size:11},callback:function(v){return v>=1000?Math.round(v/1000)+'k':v;}},grid:{color:'#f1f5f9'}}}}}}});"
    HBAR_OPTS   = "new Chart(document.getElementById('c'),{type:'bar',data:d,options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{y:{ticks:{color:'#94a3b8',font:{size:11}},grid:{display:false}},x:{ticks:{color:'#94a3b8',font:{size:11},callback:function(v){return v>=1000?Math.round(v/1000)+'k':v;}},grid:{color:'#f1f5f9'}}}}}}});"
    DOUGHNUT_OPTS = "new Chart(document.getElementById('c'),{type:'doughnut',data:d,options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'right',labels:{color:'#475569',font:{size:12},padding:16}}}}}}});"

    if ctype == "line":
        script += LINE_OPTS
    elif ctype == "bar":
        script += BAR_OPTS
    elif ctype == "hbar":
        script += HBAR_OPTS
        height = str(max(280, len(data_json.get('labels',[])) * 45 + 60)) + "px"
    elif ctype == "doughnut":
        script += DOUGHNUT_OPTS
    elif ctype == "heatmap":
        elem = '<table id="c" style="width:100%;border-collapse:collapse;font-size:12px"></table>'
        height = "auto"
        script += """
var all=[];d.cats.forEach(function(c){d.data[c].forEach(function(v){all.push(v);});});
var mn=Math.min.apply(null,all.filter(function(v){return v>0;})||[0]),mx=Math.max.apply(null,all)||1;
function bg(v){var t=v===0?0:(v-mn)/(mx-mn);return 'rgba(37,99,235,'+((t*0.8)+0.05)+')';}
function fg(v){return (v-mn)/(mx-mn)>0.5?'#fff':'#1e293b';}
var tbl=document.getElementById('c');
var h='<tr><th style="padding:8px 10px;background:#f8faff;color:#94a3b8;border:1px solid #e2e8f0;font-weight:500">Period</th>'+d.cats.map(function(c){return '<th style="padding:8px 10px;background:#f8faff;color:#475569;border:1px solid #e2e8f0;font-weight:600">'+c+'</th>';}).join('')+'</tr>';
d.periods.forEach(function(p,i){h+='<tr><td style="padding:8px 10px;color:#64748b;border:1px solid #e2e8f0;font-weight:500">'+p+'</td>';
d.cats.forEach(function(c){var v=d.data[c][i]||0;h+='<td style="padding:8px 10px;text-align:center;border:1px solid #e2e8f0;font-weight:500;background:'+bg(v)+';color:'+fg(v)+'">'+Math.round(v).toLocaleString()+'</td>';});h+='</tr>';});
tbl.innerHTML=h;"""

    html = (f'<!DOCTYPE html><html><head><meta charset="utf-8">'
            f'<script src="{CHARTJS}"></script>'
            f'<style>body{{margin:0;padding:4px;background:white}}canvas{{width:100%!important}}</style>'
            f'</head><body><div style="height:{height};position:relative">{elem}</div>'
            f'<script>{script}</script></body></html>')
    return html

COLORS = ['#2563eb','#10b981','#f59e0b','#ef4444','#8b5cf6','#06b6d4','#f97316','#84cc16','#ec4899','#14b8a6']

def render_trend(df, col_analysis):
    date_col = col_analysis["date"][0]
    num_cols  = col_analysis["numeric"][:4]
    df2 = df.copy()
    df2['_dt'] = pd.to_datetime(df2[date_col], errors='coerce')
    df2 = df2.dropna(subset=['_dt'])
    df2['_period'] = df2['_dt'].dt.strftime('%b %Y')
    df2['_sort']   = df2['_dt'].dt.to_period('M').astype(str)
    order = df2.drop_duplicates('_period').sort_values('_dt')['_period'].tolist()
    datasets = []
    for i, nc in enumerate(num_cols):
        agg = df2.groupby('_period')[nc].sum().reindex(order).fillna(0)
        datasets.append({"label": nc.replace('_',' ').title(), "data": [round(float(v),2) for v in agg.values],
                         "borderColor": COLORS[i], "backgroundColor": COLORS[i]+"22",
                         "fill": i==0, "tension": 0.4, "pointRadius": 4, "borderWidth": 2})
    data = {"labels": order, "datasets": datasets}
    return make_chart("line", data)

def render_bar(df, cat_col, num_col, top_n=10):
    grp = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(top_n)
    labels = [str(x).replace('"','') for x in grp.index.tolist()]
    values = [round(float(v),2) for v in grp.values]
    data = {"labels": labels, "datasets": [{"data": values,
             "backgroundColor": COLORS[:len(labels)], "borderRadius": 6, "borderSkipped": False}]}
    ctype = "hbar" if len(labels) > 6 else "bar"
    return make_chart(ctype, data)

def render_pie(df, cat_col, num_col):
    grp = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(8)
    labels = [str(x).replace('"','') for x in grp.index.tolist()]
    values = [round(float(v),2) for v in grp.values]
    data = {"labels": labels, "datasets": [{"data": values, "backgroundColor": COLORS[:len(labels)], "borderWidth": 0}]}
    return make_chart("doughnut", data)

def render_heatmap(df, col_analysis):
    date_col = col_analysis["date"][0]
    cat_col  = col_analysis["categorical"][0]
    num_col  = col_analysis["numeric"][0]
    df2 = df.copy()
    df2['_dt']     = pd.to_datetime(df2[date_col], errors='coerce')
    df2['_period'] = df2['_dt'].dt.strftime('%b')
    df2['_mnum']   = df2['_dt'].dt.month
    order   = df2.drop_duplicates('_period').sort_values('_dt')['_period'].tolist()
    cats    = df2[cat_col].dropna().unique().tolist()[:6]
    hdata   = {}
    for cat in cats:
        sub = df2[df2[cat_col]==cat].groupby('_period')[num_col].sum()
        hdata[str(cat).replace('"','')] = [round(float(sub.get(p,0)),2) for p in order]
    data = {"periods": order, "cats": [str(c).replace('"','') for c in cats], "data": hdata}
    return make_chart("heatmap", data)

def get_kpis(df, col_analysis, dtype):
    kpis = []
    colors = ["#2563eb","#10b981","#f59e0b","#8b5cf6","#ef4444","#06b6d4"]
    for i, nc in enumerate(col_analysis["numeric"][:6]):
        total = df[nc].sum()
        avg   = df[nc].mean()
        fmt   = lambda v: f"{v:,.0f}" if v >= 1 else f"{v:.2f}"
        kpis.append({"label": nc.replace('_',' ').title(), "value": fmt(total),
                     "sub": f"Avg: {fmt(avg)}", "color": colors[i % len(colors)]})
    return kpis

def ask_claude(question, df_info):
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=800,
            system=f"You are an expert data analyst.\nDataset info:\n{df_info}\nBe concise, use real numbers, format with bullet points.",
            messages=[{"role":"user","content":question}]
        )
        return response.content[0].text
    except anthropic.AuthenticationError:
        return "Auth error — check your API key."
    except Exception as e:
        return f"Error: {e}"

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='color:white;font-size:20px;margin-bottom:0'>🧠 DataSense AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:rgba(255,255,255,0.6);font-size:12px;margin-top:4px'>Powered by Claude</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='color:rgba(255,255,255,0.6);font-size:11px;text-transform:uppercase;letter-spacing:0.08em;font-weight:600'>UPLOAD DATA</p>", unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["csv","xlsx","xls"], label_visibility="collapsed")
    if uploaded:
        st.success(f"✓ {uploaded.name}")

    if uploaded:
        st.markdown("---")
        try:
            ext = os.path.splitext(uploaded.name)[1].lower()
            df_side = pd.read_csv(uploaded) if ext==".csv" else pd.read_excel(uploaded)
            uploaded.seek(0)
            col_analysis = analyze_columns(df_side)
            dtype = detect_dataset_type(df_side)
            info  = DATASET_TYPES[dtype]
            st.markdown(f'<div class="dataset-badge">{info["icon"]} {dtype.upper()} DATASET</div>', unsafe_allow_html=True)
            menu  = build_sidebar_menu(df_side, dtype, col_analysis)
            st.markdown("<p style='color:rgba(255,255,255,0.6);font-size:11px;text-transform:uppercase;letter-spacing:0.08em;font-weight:600'>VIEWS</p>", unsafe_allow_html=True)
            chart_clicked = None
            cols2 = st.columns(2)
            for i,(label,key) in enumerate(menu):
                with cols2[i%2]:
                    if st.button(label, key=f"nav_{key}", use_container_width=True):
                        chart_clicked = key
                        st.session_state["active_view"] = key
        except Exception as e:
            st.error(f"Preview error: {e}")
            chart_clicked = None
    else:
        chart_clicked = None

    st.markdown("---")
    st.markdown("<p style='color:rgba(255,255,255,0.6);font-size:11px;text-transform:uppercase;letter-spacing:0.08em;font-weight:600'>SAMPLE QUESTIONS</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:12px;color:rgba(255,255,255,0.75)'>• What are the key trends?<br>• Which category performs best?<br>• Any outliers or anomalies?<br>• Give me a summary<br>• What should I focus on?</p>", unsafe_allow_html=True)

# ── MAIN ─────────────────────────────────────────────────────────────────────
if not uploaded:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:#1e293b'>Welcome to DataSense AI 🧠</h2>", unsafe_allow_html=True)
    st.markdown("Upload **any** CSV or Excel file — DataSense AI automatically detects your dataset type and builds a unique dashboard just for your data.")
    c1,c2,c3,c4 = st.columns(4)
    for col_obj, icon, title, desc in [
        (c1,"🛒","Sales Data","Revenue, products, regions, orders"),
        (c2,"💰","Finance Data","Expenses, budget, cashflow, profit"),
        (c3,"👥","HR Data","Employees, salary, departments"),
        (c4,"📣","Any Dataset","Generic analytics for any data"),
    ]:
        with col_obj:
            st.markdown(f'<div class="kpi-card"><div style="font-size:24px">{icon}</div><div style="font-weight:600;color:#1e293b;margin:6px 0 4px">{title}</div><div style="font-size:12px;color:#64748b">{desc}</div></div>', unsafe_allow_html=True)
    st.stop()

# Load data
try:
    ext = os.path.splitext(uploaded.name)[1].lower()
    df = pd.read_csv(uploaded) if ext==".csv" else pd.read_excel(uploaded)
    uploaded.seek(0)
except Exception as e:
    st.error(f"Could not load file: {e}"); st.stop()

col_analysis = analyze_columns(df)
dtype        = detect_dataset_type(df)
info         = DATASET_TYPES[dtype]
menu         = build_sidebar_menu(df, dtype, col_analysis)
active_view  = st.session_state.get("active_view", menu[0][1] if menu else "table")
if chart_clicked:
    active_view = chart_clicked
    st.session_state["active_view"] = active_view

df_info = f"""File: {uploaded.name} | Type: {dtype} | Shape: {df.shape[0]} rows x {df.shape[1]} cols
Columns: {', '.join(df.columns.tolist())}
Numeric: {', '.join(col_analysis['numeric'])}
Categorical: {', '.join(col_analysis['categorical'])}
Date: {', '.join(col_analysis['date'])}
Stats:\n{df.describe().to_string()}
Sample:\n{df.head(3).to_string()}"""

# ── HEADER ────────────────────────────────────────────────────────────────────
file_title = uploaded.name.replace(".csv","").replace(".xlsx","").replace(".xls","").replace("_"," ").replace("-"," ").title()
st.markdown(f"<h3 style='margin-bottom:0.25rem'>{info['icon']} {file_title}</h3>", unsafe_allow_html=True)
st.markdown(f"<p style='color:#64748b;font-size:13px;margin-bottom:1rem'>{df.shape[0]:,} rows · {df.shape[1]} columns · {dtype.title()} dataset</p>", unsafe_allow_html=True)

# ── KPI CARDS ────────────────────────────────────────────────────────────────
kpis = get_kpis(df, col_analysis, dtype)
if kpis:
    kcols = st.columns(min(len(kpis), 4))
    for i, kpi in enumerate(kpis[:4]):
        with kcols[i]:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">{kpi["label"]}</div><div class="kpi-value" style="color:{kpi["color"]}">{kpi["value"]}</div><div class="kpi-sub">{kpi["sub"]}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "💬 Ask Claude", "📋 Raw Data"])

with tab1:
    view_label = next((l for l,k in menu if k==active_view), "Overview")
    st.markdown(f'<div class="chart-card"><div class="chart-title">{view_label}</div>', unsafe_allow_html=True)

    try:
        if active_view == "trend" and col_analysis["date"] and col_analysis["numeric"]:
            html = render_trend(df, col_analysis)
            st.components.v1.html(html, height=360)

        elif active_view.startswith("bar_"):
            cat_col = active_view[4:]
            num_col = col_analysis["numeric"][0]
            html = render_bar(df, cat_col, num_col)
            n = df[cat_col].nunique()
            st.components.v1.html(html, height=max(320, min(n*45+80, 600)))

        elif active_view == "pie" and col_analysis["categorical"] and col_analysis["numeric"]:
            cat_col = col_analysis["categorical"][0]
            num_col = col_analysis["numeric"][0]
            html = render_pie(df, cat_col, num_col)
            st.components.v1.html(html, height=360)

        elif active_view == "heatmap":
            if col_analysis["date"] and col_analysis["categorical"] and col_analysis["numeric"]:
                html = render_heatmap(df, col_analysis)
                st.components.v1.html(html, height=500)
            else:
                st.info("Heatmap needs date, category and numeric columns.")

        elif active_view == "table":
            st.dataframe(df.head(100), use_container_width=True)

        else:
            st.info("Select a view from the sidebar.")

    except Exception as e:
        st.error(f"Chart error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── OVERVIEW GRID ────────────────────────────────────────────────────────
    if col_analysis["numeric"] and col_analysis["categorical"]:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Overview")
        oc1, oc2 = st.columns(2)
        with oc1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            if col_analysis["date"] and col_analysis["numeric"]:
                st.markdown('<div class="chart-title">Trend</div>', unsafe_allow_html=True)
                st.components.v1.html(render_trend(df, col_analysis), height=220)
            elif col_analysis["categorical"] and col_analysis["numeric"]:
                cat = col_analysis["categorical"][0]
                num = col_analysis["numeric"][0]
                st.markdown(f'<div class="chart-title">{cat.title()}</div>', unsafe_allow_html=True)
                st.components.v1.html(render_bar(df, cat, num), height=220)
            st.markdown('</div>', unsafe_allow_html=True)
        with oc2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            if col_analysis["categorical"] and col_analysis["numeric"]:
                cat = col_analysis["categorical"][0]
                num = col_analysis["numeric"][0]
                st.markdown(f'<div class="chart-title">{cat.title()} Distribution</div>', unsafe_allow_html=True)
                st.components.v1.html(render_pie(df, cat, num), height=220)
            st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("#### Ask anything about your data")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div style="background:#2563eb;border-radius:16px 16px 4px 16px;padding:0.75rem 1rem;margin:0.5rem 0;color:#fff;font-size:14px">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:16px 16px 16px 4px;padding:0.75rem 1rem;margin:0.5rem 0;color:#1e293b;font-size:14px;box-shadow:0 1px 4px rgba(0,0,0,0.06)">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        fc1,fc2 = st.columns([5,1])
        with fc1:
            question = st.text_input("", placeholder="e.g. What are the key insights from this data?", label_visibility="collapsed")
        with fc2:
            submitted = st.form_submit_button("Ask →", use_container_width=True)
    if submitted and question:
        st.session_state.messages.append({"role":"user","content":question})
        with st.spinner("Thinking..."):
            answer = ask_claude(question, df_info)
        st.session_state.messages.append({"role":"assistant","content":answer})
        st.rerun()
    if st.session_state.get("messages"):
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

with tab3:
    st.markdown(f"**{df.shape[0]:,} rows × {df.shape[1]} columns**")
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, f"{file_title}.csv", "text/csv")
