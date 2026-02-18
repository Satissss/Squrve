# Squrve Text-to-SQL Demo

A Gradio-based web interface for the Squrve Text-to-SQL system. Upload databases, ask natural language questions, and generate executable SQL queries.

## Features

### 1. Database Upload

- **Single .sqlite / .db file:** Upload one SQLite database; schema is extracted automatically in Spider format.
- **Multiple .xlsx / .csv files:** Merge into one SQLite database
  - Each file becomes one table; table name = filename (without extension)
  - First row = column names; remaining rows = data

### 2. Database Selection & Persistence

- Uploaded databases are stored in `files/uploaded_db/{db_id}/`
- A `manifest.json` tracks all uploaded databases
- Select from uploaded databases across sessions for Text-to-SQL queries

### 3. Text-to-SQL Generation

- **Direct Generator:** Choose a single generator (e.g., DINSQLGenerator, LinkAlignGenerator) to generate SQL.
- **Custom Workflow:** Select a pipeline skeleton (e.g., `parser → generator`, `decomposer → parser → generator → optimizer`) and configure actors for each step. Supports Parser, Generator, Optimizer, Decomposer, Scaler, and Selector.

### 4. SQL Execution

- Execute generated SQL directly in the interface and view query results in a table.

## Setup

### Environment

```bash
# Install project dependencies (from project root)
pip install -r requirements.txt
```

### Configuration

1. Configure LLM API keys in `startup_run/startup_config.json` (e.g., OpenAI, Qwen, DeepSeek).
2. Optionally edit `demo/demo_config.yaml` for paths, port, etc.

### Launch

```bash
# From project root
python demo/gradio_demo.py
```

Runs at `http://0.0.0.0:7860` by default.

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config` | Router config path | `demo/startup_config.json` |
| `--server-name` | Server host | `0.0.0.0` |
| `--server-port` | Server port | `7860` |
| `--share` | Create public share link | - |

Example:

```bash
python demo/gradio_demo.py --config demo/startup_config.json --share --server-port 8080
```

### Usage

1. **Upload tab:** Upload .sqlite or .xlsx/.csv files, then click **Process & Create Database**.
2. **Query tab:**
   - Select a database from the dropdown
   - Enter a natural language question
   - Choose generation mode (Direct Generator or Custom Workflow)
   - Click **Generate SQL**
   - Click **Execute SQL** to run and view results

## Directory Structure

```
demo/
├── README.md           # Chinese documentation
├── README_EN.md        # English documentation (this file)
├── demo_config.yaml    # Demo config (paths, port, etc.)
├── file_to_db.py       # File-to-database conversion
├── gradio_demo.py      # Gradio UI
└── startup_config.json # LLM / router config
```

## Configuration

Main options in `demo_config.yaml`:

| Option | Description |
|--------|-------------|
| `paths.uploaded_db_root` | Root directory for uploaded databases |
| `paths.temp_data_dir` | Temporary data directory |
| `router_config` | Main config path (LLM, API keys) |
| `server.name` / `server.port` | Server host and port |
