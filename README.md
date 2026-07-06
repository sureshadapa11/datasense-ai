# DataSense AI

An AI-powered data analytics platform that turns any CSV or Excel file into an interactive dashboard — complete with smart visualizations, KPI cards, anomaly detection, and a natural language Data Agent powered by Claude.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-UI-red) ![Claude](https://img.shields.io/badge/Claude-AI-orange)

---

## What It Does

Upload a dataset and DataSense AI will:

- **Auto-detect** the dataset type (Sales, HR, Finance, Marketing, Inventory)
- **Generate KPIs** tailored to that domain — revenue totals, headcount, campaign ROAS, etc.
- **Build visualizations** — trend lines, bar charts, pie charts, heatmaps, scatter plots, histograms
- **Surface AI insights** grounded in exact numbers from your data via Claude
- **Answer questions** in plain English through the Data Agent (runs pandas under the hood)

---

## Apps

This repo contains two Streamlit apps:

| App | File | Description |
|-----|------|-------------|
| **DataSense AI** | `datasense_ai.py` | Full-featured analytics platform with 9 views and a Data Agent |
| **Sales App** | `sales_app.py` | Lightweight sales dashboard with Chart.js visuals and Claude chat |

---

## Features

### Auto Dataset Detection
Column names are scanned to classify the dataset into one of six categories:

| Type | Detected From |
|------|--------------|
| Sales | revenue, product, region, discount, order |
| HR | employee, salary, department, performance, leave |
| Finance | budget, expense, profit, cashflow, income |
| Marketing | campaign, conversion, ROAS, CTR, channel |
| Inventory | stock, SKU, warehouse, supplier, reorder |
| Generic | fallback for everything else |

### Navigation Views (datasense_ai.py)
1. **Overview** — Top KPIs + trend charts at a glance
2. **Smart Insights** — Domain-specific insight cards with exact metric values
3. **KPI Report** — Detailed statistics and value distributions
4. **Trends** — Time series + monthly performance heatmaps
5. **Categories** — Segment breakdowns by categorical columns
6. **Correlations** — Metric relationships and scatter plots
7. **Anomalies** — Data quality report: outliers, gaps, and missing values
8. **AI Analysis** — Deep Claude-generated narrative tailored to your dataset type
9. **Data Table** — Searchable and filterable raw data view

### Data Agent
Ask questions in plain English:
- "What was the top-performing region last quarter?"
- "Show me products with revenue above 50,000"
- "Calculate average salary by department"

The agent converts your question to pandas code, executes it, and returns results as tables or charts. Results can be exported as CSV.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | [Streamlit](https://streamlit.io) |
| AI | [Claude API](https://anthropic.com) (`anthropic`) |
| Data | pandas, numpy, openpyxl |
| Charts | Plotly, Matplotlib (datasense_ai.py) · Chart.js (sales_app.py) |

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/sureshadapa11/datasense-ai.git
cd datasense-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your Claude API key

Create a `.streamlit/secrets.toml` file:

```toml
ANTHROPIC_API_KEY = "your-anthropic-api-key"
```

Get your API key at [console.anthropic.com](https://console.anthropic.com).

### 4. Run the app

```bash
# Full analytics platform
streamlit run datasense_ai.py

# Sales-focused dashboard
streamlit run sales_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Dev Container (GitHub Codespaces)

This repo includes a `.devcontainer` config for one-click development in Codespaces or VS Code Dev Containers:

- **Base image**: Python 3.11 (Debian Bookworm)
- **Auto-installs**: all `requirements.txt` dependencies
- **Auto-launches**: `sales_app.py` on port 8501
- **Extensions**: Python + Pylance

Click **"Open in Codespace"** on GitHub and you're ready in under a minute.

---

## Supported File Formats

- `.csv`
- `.xlsx` / `.xls`

---

## License

MIT
