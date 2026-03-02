---
name: setup
description: "Interactive project setup wizard. From a clean codebase, guides user through provider selection (OpenAI/Azure/DeepSeek/Ollama), API key configuration, dependency installation, config generation, and launches the dashboard. Auto-diagnoses and fixes startup failures with up to 3 retry rounds. Use when user says 'setup', 'set up', 'configure', 'init project', '初始化', '环境配置', '项目配置', 'first run', 'get started', 'quick start', or wants to configure and launch the project from scratch."
---

# Setup

Interactive wizard: configure providers → install deps → generate config → launch dashboard → auto-fix issues.

---

## Pipeline

```
Preflight → Ask User → Generate Config → Install Deps → Validate → Launch → Usage Guide
```

> Auto-fix loop: if any step fails, diagnose → fix → retry (≤3 rounds).

---

## Step 1: Preflight Checks

Verify prerequisites before asking the user anything:

```powershell
python --version          # Require >=3.10
pip --version             # Verify pip available
```

If Python < 3.10, stop and inform user. Check if `.venv` exists; if not, create it:

```powershell
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
```

---

## Step 2: Ask User for Configuration

Use the `ask_questions` tool to gather provider choices. Ask in batches (max 4 questions per call).

### Batch 1: Core Providers

Ask these questions together:

1. **LLM Provider** — Which LLM provider?
   - Options: `OpenAI`, `Azure OpenAI`, `DeepSeek`, `Ollama (local)`
   - Recommended: `OpenAI`

2. **Embedding Provider** — Which embedding provider?
   - Options: `OpenAI`, `Azure OpenAI`, `Ollama (local)`
   - Recommended: `OpenAI` (should match LLM provider when possible)

3. **Vision** — Enable vision/image captioning?
   - Options: `Yes`, `No`
   - Recommended: `Yes`

4. **Rerank** — Enable reranking?
   - Options: `No (fastest)`, `Cross-Encoder (local model)`, `LLM-based`
   - Recommended: `No (fastest)`

### Batch 2: Credentials (based on Batch 1 answers)

Ask for credentials based on selected providers. Refer to [references/provider_profiles.md](references/provider_profiles.md) for required fields per provider.

**If OpenAI selected:**
- Ask: OpenAI API Key
- Ask: LLM model (default: `gpt-4o`)
- Ask: Embedding model (default: `text-embedding-ada-002`)

**If Azure OpenAI selected:**
- Ask: Azure API Key
- Ask: Azure Endpoint URL
- Ask: LLM deployment name (default: `gpt-4o`)
- Ask: Embedding deployment name (default: `text-embedding-ada-002`)

**If DeepSeek selected:**
- Ask: DeepSeek API Key
- Ask: Embedding provider separately (DeepSeek has no embeddings — must use OpenAI/Ollama)

**If Ollama selected:**
- Ask: Ollama base URL (default: `http://localhost:11434`)
- Ask: LLM model name (default: `llama3`)
- Ask: Embedding model name (default: `nomic-embed-text`)
- Verify Ollama is running: `curl http://localhost:11434/api/tags` or equivalent

---

## Step 3: Generate Config

Read the template from [references/settings_template.yaml](references/settings_template.yaml) and fill in values based on user answers.

Key rules:
- Look up `dimensions` from the model→dimensions table in [references/provider_profiles.md](references/provider_profiles.md)
- For Ollama: set `base_url`, leave `api_key`/`azure_endpoint`/`deployment_name` empty
- For OpenAI: leave `azure_endpoint`/`deployment_name`/`api_version` empty
- If vision disabled: set `vision_llm.enabled: false`
- For rerank: set `enabled`, `provider`, and `model` accordingly

Write the generated config to `config/settings.yaml`.

Also ensure required directories exist:

```powershell
python -c "from pathlib import Path; [Path(d).mkdir(parents=True, exist_ok=True) for d in ['data/db/chroma', 'data/images/default', 'logs', 'config/prompts']]"
```

---

## Step 4: Install Dependencies

```powershell
pip install -e ".[dev]"
```

If specific providers need extra packages:
- **Cross-Encoder rerank**: `pip install sentence-transformers`
- **Streamlit dashboard**: `pip install streamlit`
- **OpenAI**: `pip install openai`

Verify critical imports:

```powershell
python -c "import chromadb; import mcp; import yaml; print('Core deps OK')"
python -c "import streamlit; print('Streamlit OK')"
python -c "import openai; print('OpenAI SDK OK')"
```

---

## Step 5: Validate Configuration

Test that the config loads correctly:

```powershell
python -c "from src.core.settings import load_settings; s = load_settings(); print(f'Config OK: LLM={s.llm.provider}/{s.llm.model}, Embed={s.embedding.provider}/{s.embedding.model}')"
```

If this fails, enter **auto-fix loop**:

### Auto-Fix Loop (≤3 rounds)

```
Round 0..2:
  Read error message
  Diagnose root cause (missing field, wrong type, bad provider name, etc.)
  Fix config/settings.yaml or install missing dependency
  Re-validate
  If pass → continue to Step 6
  If fail → next round
```

Common fixes:
- `SettingsError: Missing required field` → add the field to settings.yaml
- `ModuleNotFoundError` → `pip install <package>`
- `Connection refused` (Ollama) → inform user to start Ollama service
- Wrong `dimensions` value → look up correct value from provider_profiles.md

If 3 rounds fail, report the issue to the user with diagnosis and ask for help.

---

## Step 6: Launch Dashboard

```powershell
python scripts/start_dashboard.py --port 8501
```

Run this as a **background process**. Wait a few seconds, then verify it's accessible:

```powershell
python -c "
import urllib.request
try:
    r = urllib.request.urlopen('http://localhost:8501/_stcore/health')
    print('Dashboard is running!' if r.status == 200 else f'Status: {r.status}')
except Exception as e:
    print(f'Dashboard not yet ready: {e}')
"
```

If the dashboard fails to start, enter auto-fix loop:
- Read the error output from the background terminal
- Common issues: missing `streamlit`, port already in use, import errors
- Fix and retry

---

## Step 7: Usage Guide

After successful launch, present this to the user:

```
🎉 Setup Complete!

Dashboard: http://localhost:8501

Quick Start:
  1. Ingest documents:  python scripts/ingest.py <path-to-pdf-or-folder>
  2. Query:             python scripts/query.py "your question here"
  3. Dashboard:         python scripts/start_dashboard.py
  4. MCP Server:        python main.py

Configuration: config/settings.yaml
Logs:          logs/traces.jsonl

Provider: {provider} / Model: {model}
```

Adapt the message based on the user's chosen providers and language (Chinese if user communicates in Chinese).
