import os
import json
import streamlit as st
import pandas as pd
import anthropic

# ── PUT YOUR API KEY HERE ──────────────────────────────────────
os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DataSense AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #f0f4ff; color: #1e293b; }
section[data-testid="stSidebar"] { background: linear-gradient(160deg, #1e3a5f 0%, #2563eb 100%); border-right: none; }
section[data-testid="stSidebar"] * { color: #ffffff !important; }
section[data-testid="stSidebar"] .stMarkdown p { color: rgba(255,255,255,0.75) !important; }
.metric-card { background: #ffffff; border: none; border-radius: 16px; padding: 1.25rem 1.5rem; margin-bottom: 0.5rem; box-shadow: 0 2px 12px rgba(37,99,235,0.08); }
.metric-label { font-size: 12px; color: #64748b; margin-bottom: 6px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-value { font-size: 28px; font-weight: 700; color: #2563eb; }
.metric-sub { font-size: 12px; color: #94a3b8; margin-top: 3px; }
.chat-user { background: #2563eb; border-radius: 16px 16px 4px 16px; padding: 0.75rem 1rem; margin: 0.5rem 0; color: #ffffff; font-size: 14px; }
.chat-claude { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px 16px 16px 4px; padding: 0.75rem 1rem; margin: 0.5rem 0; color: #1e293b; font-size: 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
h1, h2, h3 { color: #1e293b !important; }
.stButton > button { background: rgba(255,255,255,0.15); color: #ffffff; border: 1px solid rgba(255,255,255,0.3); border-radius: 10px; font-size: 13px; padding: 0.4rem 1rem; transition: all 0.2s; backdrop-filter: blur(4px); }
.stButton > button:hover { background: rgba(255,255,255,0.25); border-color: rgba(255,255,255,0.6); color: #ffffff; }
.stTextInput > div > div > input { background: #ffffff; color: #1e293b; border: 1px solid #e2e8f0; border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.05); }
.stTextInput > div > div > input:focus { border-color: #2563eb; box-shadow: 0 0 0 3px rgba(37,99,235,0.15); }
div[data-testid="stFileUploader"] { background: #ffffff; border: 2px dashed #2563eb; border-radius: 12px; padding: 0.5rem; }
div[data-testid="stFileUploader"] * { color: #1e293b !important; }
div[data-testid="stFileUploader"] span { color: #2563eb !important; font-weight: 600; }
div[data-testid="stFileUploader"] small { color: #64748b !important; font-size: 11px; }
div[data-testid="stFileUploader"] p { color: #1e293b !important; font-size: 13px; }
div[data-testid="stFileUploader"] button { background: #2563eb !important; color: #ffffff !important; border: none !important; border-radius: 8px !important; font-weight: 500 !important; padding: 0.35rem 1rem !important; }
div[data-testid="stFileUploader"] button:hover { background: #1d4ed8 !important; }
div[data-testid="stFileUploaderDropzoneInstructions"] div { color: #1e293b !important; }
div[data-testid="stFileUploaderDropzoneInstructions"] span { color: #2563eb !important; }
[data-testid="stFileUploaderDropzone"] { background: #f8faff !important; }
.stTabs [data-baseweb="tab-list"] { background: #ffffff; border-radius: 12px; padding: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.06); }
.stTabs [data-baseweb="tab"] { color: #64748b; border-radius: 8px; font-weight: 500; }
.stTabs [aria-selected="true"] { background: #2563eb; color: #ffffff; }
[data-testid="stMetric"] { background: white; border-radius: 12px; padding: 1rem; }
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

CHARTJS = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"

def get_client():
    return anthropic.Anthropic()

def extract_data(df):
    d = {}
    try:
        df = df.copy()
        df['_dt'] = pd.to_datetime(df['date'])
        df['_mn'] = df['_dt'].dt.month
        df['_mb'] = df['_dt'].dt.strftime('%b')
        m = df.groupby(['_mn','_mb'])['revenue'].sum().reset_index().sort_values('_mn')
        d['ml'] = m['_mb'].tolist()
        d['mv'] = [int(x) for x in m['revenue']]
        d['tr'] = int(df['revenue'].sum())
        d['tu'] = int(df['units_sold'].sum())
        d['to'] = int(len(df))
        d['av'] = round(float(df['revenue'].mean()), 2)
        d['bm'] = m.loc[m['revenue'].idxmax(), '_mb']
        d['bm_v'] = int(m['revenue'].max())
        bp = df.groupby('product')['revenue'].sum().sort_values(ascending=False)
        d['pl'] = [s.replace('"','') for s in bp.index.tolist()]
        d['pv'] = [int(x) for x in bp.values]
        br = df.groupby('region')['revenue'].sum().sort_values(ascending=False)
        d['rl'] = br.index.tolist(); d['rv'] = [int(x) for x in br.values]
        bs = df.groupby('salesperson')['revenue'].sum().sort_values(ascending=False)
        d['sl'] = bs.index.tolist(); d['sv'] = [int(x) for x in bs.values]
        bc = df.groupby('category')['revenue'].sum().sort_values(ascending=False)
        d['cl'] = bc.index.tolist(); d['cv'] = [int(x) for x in bc.values]
        cats = df['category'].unique().tolist()
        hc = {}
        for cat in cats:
            sub = df[df['category']==cat].groupby(['_mn','_mb'])['revenue'].sum().reset_index().sort_values('_mn')
            hc[cat] = [int(x) for x in sub['revenue']]
        d['hm'] = m['_mb'].tolist(); d['hc'] = cats; d['hd'] = hc
        prods = [s.replace('"','') for s in df['product'].unique().tolist()]
        hp = {}
        for prod in df['product'].unique().tolist():
            sub = df[df['product']==prod].groupby(['_mn'])['revenue'].sum()
            hp[prod.replace('"','')] = [int(sub.get(i, 0)) for i in range(1,13)]
        d['hpl'] = prods; d['hpd'] = hp
    except Exception as e:
        st.error(f"Data error: {e}")
    return d

def chart_html(chart_type, d):
    j = json.dumps(d)
    P = "String.fromCharCode(163)"

    scripts = {
        "trend": "var d=" + j + ";new Chart(document.getElementById('c'),{type:'line',data:{labels:d.ml,datasets:[{data:d.mv,borderColor:'#58a6ff',backgroundColor:'rgba(88,166,255,0.08)',fill:true,tension:0.4,pointBackgroundColor:'#58a6ff',pointRadius:5}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{ticks:{color:'#64748b'},grid:{display:false}},y:{ticks:{color:'#64748b',callback:function(v){return " + P + "+Math.round(v/1000)+'k'}},grid:{color:'#f1f5f9'}}}}}}); ",

        "product": "var d=" + j + ";new Chart(document.getElementById('c'),{type:'bar',data:{labels:d.pl,datasets:[{data:d.pv,backgroundColor:['#58a6ff','#3fb950','#f78166','#d2a8ff','#ffa657','#79c0ff'],borderRadius:6}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{ticks:{color:'#64748b',font:{size:11}},grid:{display:false}},y:{ticks:{color:'#64748b',callback:function(v){return " + P + "+Math.round(v/1000)+'k'}},grid:{color:'#f1f5f9'}}}}}}); ",

        "region": "var d=" + j + ";new Chart(document.getElementById('c'),{type:'bar',data:{labels:d.rl,datasets:[{data:d.rv,backgroundColor:['#3fb950','#58a6ff','#ffa657','#f78166'],borderRadius:6}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{ticks:{color:'#64748b'},grid:{display:false}},y:{ticks:{color:'#64748b',callback:function(v){return " + P + "+Math.round(v/1000)+'k'}},grid:{color:'#f1f5f9'}}}}}}); ",

        "salesperson": "var d=" + j + ";new Chart(document.getElementById('c'),{type:'bar',data:{labels:d.sl,datasets:[{data:d.sv,backgroundColor:['#d2a8ff','#58a6ff','#3fb950','#ffa657'],borderRadius:6}]},options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{y:{ticks:{color:'#64748b'},grid:{display:false}},x:{ticks:{color:'#64748b',callback:function(v){return " + P + "+Math.round(v/1000)+'k'}},grid:{color:'#f1f5f9'}}}}}}); ",

        "category": "var d=" + j + ";new Chart(document.getElementById('c'),{type:'doughnut',data:{labels:d.cl,datasets:[{data:d.cv,backgroundColor:['#58a6ff','#3fb950','#f78166'],borderWidth:0}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'bottom',labels:{color:'#64748b',font:{size:12}}}}}}); ",

        "heatmap_cat": "var d=" + j + ";var all=[];d.hc.forEach(function(c){d.hd[c].forEach(function(v){all.push(v);});});var mn=Math.min.apply(null,all),mx=Math.max.apply(null,all);function bg(v){var t=(v-mn)/(mx-mn);return 'rgb('+Math.round(13+t*75)+','+Math.round(17+t*149)+','+Math.round(23+t*232)+')';}function fg(v){return (v-mn)/(mx-mn)>0.5?'#0d1117':'#e6edf3';}var tbl=document.getElementById('c');var h='<tr><th style=\'background:#f8fafc;padding:8px 12px;color:#8b949e;border:1px solid #30363d\'>Month</th>'+d.hc.map(function(c){return '<th style=\'background:#f8fafc;padding:8px 12px;color:#8b949e;border:1px solid #30363d\'>'+c+'</th>';}).join('')+'<th style=\'background:#f8fafc;padding:8px 12px;color:#8b949e;border:1px solid #30363d\'>Total</th></tr>';d.hm.forEach(function(m,i){var tot=d.hc.reduce(function(s,c){return s+(d.hd[c][i]||0);},0);h+='<tr><td style=\'padding:8px 12px;color:#8b949e;border:1px solid #21262d\'>'+m+'</td>';d.hc.forEach(function(c){var v=d.hd[c][i]||0;h+='<td style=\'padding:8px 12px;text-align:center;border:1px solid #21262d;font-weight:500;background:'+bg(v)+';color:'+fg(v)+'\'>'+" + P + "+v.toLocaleString()+'</td>';});h+='<td style=\'padding:8px 12px;text-align:center;color:#8b949e;border:1px solid #21262d\'>'+" + P + "+tot.toLocaleString()+'</td></tr>';});tbl.innerHTML=h;",

        "heatmap_prod": "var d=" + j + ";var all=[];d.hpl.forEach(function(p){d.hpd[p].forEach(function(v){all.push(v);});});var mn=Math.min.apply(null,all),mx=Math.max.apply(null,all);function bg(v){var t=(v-mn)/(mx-mn);return 'rgb('+Math.round(13+t*234)+','+Math.round(17+t*112)+','+Math.round(23+t*79)+')';}function fg(v){return (v-mn)/(mx-mn)>0.5?'#0d1117':'#e6edf3';}var tbl=document.getElementById('c');var h='<tr><th style=\'background:#f8fafc;padding:6px 10px;color:#8b949e;border:1px solid #30363d;font-size:12px\'>Month</th>'+d.hpl.map(function(p){return '<th style=\'background:#f8fafc;padding:6px 8px;color:#8b949e;border:1px solid #30363d;font-size:11px\'>'+p+'</th>';}).join('')+'</tr>';d.ml.forEach(function(m,i){h+='<tr><td style=\'padding:6px 10px;color:#8b949e;border:1px solid #21262d;font-size:12px\'>'+m+'</td>';d.hpl.forEach(function(p){var v=d.hpd[p][i]||0;h+='<td style=\'padding:6px 8px;text-align:center;border:1px solid #21262d;font-size:11px;font-weight:500;background:'+bg(v)+';color:'+fg(v)+'\'>'+" + P + "+v.toLocaleString()+'</td>';});h+='</tr>';});tbl.innerHTML=h;",
    }

    if chart_type not in scripts:
        return ""

    script = scripts[chart_type]
    is_table = chart_type in ("heatmap_cat", "heatmap_prod")
    elem = '<table id="c" style="width:100%;border-collapse:collapse"></table>' if is_table else '<canvas id="c"></canvas>'
    height = "auto" if is_table else "320px"

    return (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        "<script src=\"" + CHARTJS + "\"></script>"
        "<style>body{margin:0;padding:8px;background:white;border-radius:12px}canvas{width:100%!important}</style>"
        "</head><body>"
        "<div style=\"height:" + height + ";position:relative\">" + elem + "</div>"
        "<script>" + script + "</script>"
        "</body></html>"
    )

def ask_claude(question, df_info):
    client = get_client()
    system = f"You are an expert data analyst.\nDataset:\n{df_info}\nBe concise. Use real numbers. Format nicely with bullet points."
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=system,
            messages=[{"role": "user", "content": question}]
        )
        return response.content[0].text
    except anthropic.AuthenticationError:
        return "Auth error — check your API key."
    except Exception as e:
        return f"Error: {e}"

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='color:white;font-size:18px'>🧠 DataSense AI</h2>", unsafe_allow_html=True)
    st.markdown("---")
    uploaded = st.file_uploader("Upload your data", type=["csv","xlsx","xls"], help="CSV or Excel files supported")
    if uploaded:
        st.success(f"✓ {uploaded.name}")
    st.markdown("---")
    st.markdown("<p style='color:rgba(255,255,255,0.6);font-size:11px;text-transform:uppercase;letter-spacing:0.08em;font-weight:600'>CHARTS</p>", unsafe_allow_html=True)
    chart_clicked = None
    cols = st.columns(2)
    charts = [
        ("📈 Trend",       "trend"),
        ("📦 Products",    "product"),
        ("🌍 Regions",     "region"),
        ("👤 Salespeople", "salesperson"),
        ("🍩 Categories",  "category"),
        ("🔥 Heat Cat",    "heatmap_cat"),
        ("🔥 Heat Prod",   "heatmap_prod"),
    ]
    for i, (label, key) in enumerate(charts):
        with cols[i % 2]:
            if st.button(label, key=f"btn_{key}", use_container_width=True):
                chart_clicked = key
    st.markdown("---")
    st.markdown("<p style='color:rgba(255,255,255,0.6);font-size:11px;text-transform:uppercase;letter-spacing:0.08em;font-weight:600'>SAMPLE QUESTIONS</p>", unsafe_allow_html=True)
    st.markdown("""
- Which product made the most revenue?
- Who is the top salesperson?
- What month had the biggest growth?
- Compare regions
- Any outliers in the data?
""")

# ── MAIN ─────────────────────────────────────────────────────────────────────
if not uploaded:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:#1e293b'>Welcome to DataSense AI 👋</h2>", unsafe_allow_html=True)
    st.markdown("Upload any CSV or Excel file in the sidebar — DataSense AI will automatically analyse it and build your dashboard.")
    st.markdown("""
**What you can do:**
- 📊 View 7 different interactive charts
- 💬 Ask questions about your data in plain English
- 🔥 Explore heatmaps by category and product
- 📈 Analyse trends, regions, and salesperson performance
""")
    st.stop()

# Load data
try:
    ext = os.path.splitext(uploaded.name)[1].lower()
    df = pd.read_csv(uploaded) if ext == ".csv" else pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Could not load file: {e}")
    st.stop()

df_info = f"""Dataset: {uploaded.name}
Shape: {df.shape[0]} rows x {df.shape[1]} columns
Columns: {', '.join(df.columns.tolist())}
Stats:\n{df.describe().to_string()}
Sample:\n{df.head(3).to_string()}"""

data = extract_data(df)

# ── METRICS ROW ──────────────────────────────────────────────────────────────
file_title = uploaded.name.replace(".csv","").replace(".xlsx","").replace(".xls","").replace("_"," ").replace("-"," ").title()
st.markdown(f"### 🧠 {file_title} — AI Dashboard")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">💰 Total Revenue</div><div class="metric-value">£{data.get("tr",0):,}</div><div class="metric-sub">Full year</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">📦 Units Sold</div><div class="metric-value" style="color:#10b981">{data.get("tu",0):,}</div><div class="metric-sub">All products</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">🛒 Total Orders</div><div class="metric-value" style="color:#f59e0b">{data.get("to",0):,}</div><div class="metric-sub">Transactions</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-label">📈 Avg Order Value</div><div class="metric-value" style="color:#8b5cf6">£{data.get("av",0):,}</div><div class="metric-sub">Per transaction</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊 Charts", "💬 Ask Claude"])

with tab1:
    active_chart = chart_clicked or st.session_state.get("active_chart", "trend")
    if chart_clicked:
        st.session_state["active_chart"] = chart_clicked

    chart_titles = {
        "trend":        "Monthly Revenue Trend",
        "product":      "Revenue by Product",
        "region":       "Revenue by Region",
        "salesperson":  "Revenue by Salesperson",
        "category":     "Revenue by Category",
        "heatmap_cat":  "Heatmap — Month & Category",
        "heatmap_prod": "Heatmap — Month & Product",
    }

    st.markdown(f"#### {chart_titles.get(active_chart, '')}")
    html = chart_html(active_chart, data)
    if html:
        height = 420 if active_chart not in ("heatmap_cat","heatmap_prod") else 500
        st.components.v1.html(html, height=height)
    else:
        st.info("Select a chart from the sidebar.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### All charts overview")
    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)
    for col_obj, ctype in [(r1c1,"trend"),(r1c2,"product"),(r2c1,"region"),(r2c2,"salesperson")]:
        with col_obj:
            st.markdown(f'<p style="font-size:12px;color:#8b949e;margin-bottom:6px">{chart_titles[ctype]}</p>', unsafe_allow_html=True)
            st.components.v1.html(chart_html(ctype, data), height=240)

with tab2:
    st.markdown("#### Ask anything about your data")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-claude">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        fc1, fc2 = st.columns([5,1])
        with fc1:
            question = st.text_input("", placeholder="e.g. Which product had the highest revenue?", label_visibility="collapsed")
        with fc2:
            submitted = st.form_submit_button("Ask →", use_container_width=True)

    if submitted and question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Thinking..."):
            answer = ask_claude(question, df_info)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.messages:
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()
