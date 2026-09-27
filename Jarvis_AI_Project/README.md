# ⚡ JARVIS AI
**"An AI that reverse-engineers reality from data."**

An autonomous data-analysis agent: upload any CSV, and JARVIS AI profiles it,
finds hidden correlations and anomalies, writes plain-English insights, lets
you ask it follow-up questions in natural language (it writes and runs the
pandas/plotly code live), and exports everything as a PDF report.

## Team roles (matches the pitch deck)
| Role | Owns |
|---|---|
| 🧠 AI Engineer | `backend/agent.py` — LLM prompts, insight generation, code-gen agent |
| ⚙️ Backend Dev | `backend/pipeline.py` — CSV parsing, stats, correlations, anomaly detection |
| 🎨 Frontend Dev | `app.py` — Streamlit UI, KPI cards, charts, neon theme |
| 📋 Product Lead | `backend/report.py` + testing — PDF export, demo flow |

*(This table is developer-facing documentation only — the running app's
sidebar is kept for actually useful, judge-facing controls: AI provider,
accent theme, dataset info, live filters, and data export — see below.)*

## 1. Setup

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Pick an AI provider
JARVIS AI's "AI Insights" and "Ask JARVIS" tabs work with several providers —
pick one from the sidebar dropdown (CSV profiling, charts and anomaly
detection all work with none of them, no key needed):

| Provider | Cost | Setup |
|---|---|---|
| **Anthropic (Claude)** | Paid | Key from console.anthropic.com |
| **Groq** — Llama 3.3, Mixtral, Gemma (open-source models) | Free tier | Key from console.groq.com |
| **Ollama** — fully local, fully open-source | Free, no key | Install from ollama.com, run `ollama pull llama3.1`, then `ollama serve` |
| **OpenRouter** — many open-source models | Free tier available | Key from openrouter.ai |
| **Together AI** — open-source models | Free tier available | Key from together.ai |

Groq is the best pick for a hackathon demo: it's free, fast, and runs
real open-source models (Llama 3.3 70B) rather than a toy model. Ollama is
the fully-offline option if you don't want to depend on Wi-Fi at the venue,
but it needs the model downloaded ahead of time and is slower on a laptop.

Whichever you choose, the key is only pasted into the sidebar field at
runtime — it's never saved to disk.

## 3. Run the app

```bash
streamlit run app.py
```

This opens JARVIS AI at `http://localhost:8501`.

## 4. Sidebar controls
The sidebar is fully functional, not decorative:
- **🔑 AI Configuration** — collapsed by default to keep the workspace clean. Expand it to pick an AI provider, choose a model from a preset dropdown (or select "Custom..." to type your own), enter an API key, and hit **🔌 Test Connection** to confirm it works before running any analysis
- **🎨 Dashboard Theme** — pick an accent color that re-colors the whole dashboard live, including the title, badges, buttons, and every chart palette
- **📁 Dataset Info** — file name, size, and shape at a glance
- **Filters** — filter by category (multiselect) or numeric range (slider) on up to 5 columns each; every tab, chart, and KPI updates live against the filtered data, with a banner showing how many rows are currently in view
- **📥 Data Export** — download the currently filtered data as a CSV
- **🔄 Load a different dataset** — clears the current data and returns to the upload screen

## 5. Getting started (no file needed)
The main screen leads with a drag-and-drop CSV upload zone. If you don't have
a file handy, click **📊 Load Sample Sales Data** or **🧑‍💼 Load Sample HR Data**
to explore the full dashboard instantly with realistic demo data (each
includes a few missing values and outliers, like real data does).

Once data is loaded, the top of the page shows:
- **Quick metric cards** — total rows, columns, memory usage, and missing value count
- **Column types & missing values** — an expandable breakdown of every column's detected type (numeric/categorical) and missing %
- **⚡ Quick Insights** — one-click buttons for Summary Statistics, Detect Outliers, and Correlation Matrix that answer instantly with no AI call and no typing

## 6. Demo it

The dashboard now has **7 tabs** and an 8-card KPI strip (rows, columns,
numeric/categorical counts, missing %, duplicates, anomalies, and a live
Data Health Score):

1. **Overview** — data preview, correlation heatmap, column-type donut,
   distribution explorer, category breakdown bar chart
2. **Data Health** — a gauge showing the 0-100 health score + verdict,
   a quality breakdown chart (missing/duplicate/outlier %), and a
   per-column missing-values chart
3. **Deep Dive** — interactive scatter matrix across numeric columns
   (color by any category), box plots grouped by category, and a treemap
   of category composition
4. **AI Insights** — click "Generate Insights" and watch the live status
   steps (reading profile → asking the model → rendering) — each finding
   renders as its own glowing insight card, with the exact prompt available
   in a "Show prompt" expander for transparency
5. **Ask JARVIS** — ask a real question live, e.g. *"Which column correlates
   most with [target column]?"* — the agent shows its steps, writes and runs
   the code in front of the judges, and the generated code is always visible
   in the "Show generated code" expander
6. **Anomalies** — flagged outlier rows plus a scatter plot highlighting them
7. **Export Report** — downloads a PDF with the health-score badge and
   full findings, as a leave-behind for judges

## 7. Presentation tips
- Open with the tagline: *"JARVIS AI reverse-engineers reality from data — it
  doesn't just chart your numbers, it explains what they mean."*
- Show the agent writing code live rather than only pre-baked charts — that's
  the differentiator over a plain BI dashboard.
- Mention the architecture split (LLM agent / execution pipeline / UI / QA)
  to show it's a real team build, not a single script.
- Close with the PDF export — a tangible artifact judges can keep.

## Project structure
```
jarvis_ai/
├── app.py                  # Streamlit frontend
├── backend/
│   ├── pipeline.py         # CSV parsing + statistics + anomaly detection
│   ├── agent.py            # LLM insight + code-generation agent
│   └── report.py           # PDF report builder
├── requirements.txt
└── .streamlit/config.toml  # Dark neon theme
```

## Notes / things to double-check before the demo
- Test with 2-3 different CSVs beforehand (one with only numeric columns,
  one with categorical columns, one with missing data) to make sure every
  tab has something to show.
- If the venue has no internet, the AI Insights / Ask JARVIS tabs won't work
  (they call the Anthropic API) — Overview and Anomalies tabs still work
  fully offline, so plan the demo order accordingly.
