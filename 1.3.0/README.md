# WiFi Site Survey Automation

## Updated workflow

This project now runs in five clear stages:

1. Extract heatmap and scale images from DOCX reports
2. Convert extracted heatmaps into CSV outputs
3. Run parameter-vs-range analysis to generate graphs and supporting tables
4. Run local AI as a separate graph-summary stage
5. Chat with a local-AI knowledge base built from the generated outputs

The local AI stage is no longer embedded inside the range-analysis step.

---

## What changed in this update

### Local AI is now its own stage

The app no longer treats local AI as an optional toggle inside the parameter-vs-range run.

Instead:
- Step 3 generates the raw analytical outputs only
- Step 4 reads those outputs and creates graph-by-graph reasoning summaries
- Step 5 opens a knowledge-base chat over the generated artifacts

### Two local AI functions are now separated

#### 1. Graph summaries with reasoning

Step 4 uses a local model to produce:
- one reasoning summary for each generated graph, with its graph embedded in the DOCX
- one metric-level summary for each selected parameter, with the related graphs embedded
- one consolidated overall report across the selected routers, floors, and bands

The summaries are saved under:

```text
runs/<router_name>/ai_reports/
```

Typical output layout:

```text
ai_reports/
  <metric>/
    graphs/
      <graph_name>__graph_summary.md
      <graph_name>__graph_summary.docx
      <graph_name>__graph_summary.json
    <metric>__overall_summary.md
    <metric>__overall_summary.docx
    <metric>__overall_summary.json
  _overall/
    router_overall_summary.md
    router_overall_summary.docx
    router_overall_summary.json
```

#### 2. Knowledge-base chat

Step 5 builds a local knowledge base from:
- AI reports
- RvR tables
- generated CSV summaries
- logs

Then it lets the user ask grounded questions such as:
- which router looks strongest on 5 GHz in upper floor
- which graphs show the fastest drop with distance
- what weak points are repeated across metrics
- where the current outputs are missing evidence

---

## Main files

### `app.py`

Streamlit interface for all five stages.

### `local_ai.py`

Local AI helper module. It now handles:
- connection checks for local model servers
- graph-by-graph summary generation
- metric-level summary generation
- lightweight knowledge-base building
- retrieval-based local chat over generated project outputs

### `parameter_vs_range.py`

Still responsible only for the graph and table generation stage.

---

## Local AI expectation

This project expects a local model server.

Supported local modes in the UI:
- `ollama`
- `openai_compatible` for a local server that exposes an OpenAI-style API

Recommended default:
- Provider: `ollama`
- Base URL: `http://127.0.0.1:11434`
- Model: `gemma3:4b`

---

## Install

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install Python packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install Tesseract on Ubuntu or Debian:

```bash
sudo apt update
sudo apt install -y tesseract-ocr libgl1 libglib2.0-0
```

Install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start Ollama:

```bash
ollama serve
```

Pull a model:

```bash
ollama pull gemma3:4b
```

Run the app:

```bash
streamlit run app.py
```

---

## Recommended usage order

### Step 1
Upload DOCX reports and extract the embedded images.

### Step 2
Generate CSV outputs from the extracted heatmaps.

### Step 3
Upload the master ESX, router ESX files, and run one or more selected parameters.

### Step 4
Generate graph summaries, metric summaries, and one consolidated overall report. The DOCX reports use one consistent format and strip assistant-style endings.

### Step 5
Build or refresh the knowledge base, then ask questions grounded in the generated outputs.

---

## Notes on the knowledge-base chat

The knowledge-base chat is intentionally lightweight and local.

It does not require an external vector database. Instead it builds a simple retrieval set from your generated project files and sends the most relevant chunks to the local model.

This keeps the full workflow local to the machine.
