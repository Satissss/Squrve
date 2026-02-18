"""
Squrve Text-to-SQL Demo

Gradio UI for:
- Upload databases (xlsx/csv or sqlite)
- Select database for query
- Generate SQL via direct Generator or custom Workflow
- Execute SQL
"""

import sys
import uuid
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import gradio as gr
import pandas as pd
from loguru import logger

from core.base import Router
from core.engine import Engine
import core.actor.agent  # ensure all actors are registered
from core.data_manage import DataLoader
from core.utils import save_dataset, load_dataset
from demo.file_to_db import (
    process_uploaded_files,
    load_upload_manifest,
)
from core.db_connect import get_sql_exec_result

# Actor type -> NAME list (ensure actors are registered via imports)
ACTOR_BY_TYPE = {
    "parser": [
        "LinkAlignParser",
        "CHESSSelectorParser",
        "RSLSQLBiDirParser",
        "MACSQLCoTParser",
        "DINSQLCoTParser",
        "OpenSearchCoTParser",
    ],
    "generator": [
        "LinkAlignGenerator",
        "DINSQLGenerator",
        "DAILSQLGenerator",
        "CHESSGenerator",
        "MACSQLGenerator",
        "RSLSQLGenerator",
        "ReFoRCEGenerator",
        "OpenSearchSQLGenerator",
        "RecursiveGenerator",
    ],
    "optimizer": [
        "LinkAlignOptimizer",
        "RSLSQLOptimizer",
        "CHESSOptimizer",
        "AdaptiveOptimizer",
        "OpenSearchSQLOptimizer",
        "MACSQLOptimizer",
        "DINSQLOptimizer",
    ],
    "decomposer": [
        "DINSQLDecomposer",
        "MACSQLDecomposer",
        "RecursiveDecomposer",
    ],
    "scaler": [
        "ChessScaler",
        "DINSQLScaler",
        "MACSQLScaler",
        "RSLSQLScaler",
        "OpenSearchSQLScaler",
    ],
    "selector": [
        "FastExecSelector",
        "ChaseSelector",
        "CHESSSelector",
        "AgentDebateSelector",
        "OpenSearchSQLSelector",
    ],
}

WORKFLOW_SKELETONS = [
    ["generator"],
    ["parser", "generator"],
    ["parser", "generator", "optimizer"],
    ["parser", "generator", "scaler", "selector"],
    ["parser", "generator", "optimizer", "scaler", "selector"],
    ["decomposer", "parser", "generator"],
    ["decomposer", "parser", "generator", "optimizer"],
    ["decomposer", "parser", "generator", "scaler", "selector"],
    ["decomposer", "parser", "generator", "optimizer", "scaler", "selector"],
]

DEMO_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="teal",
    neutral_hue="slate",
)

DEMO_CSS = """
.gradio-container {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}
.welcome-header {
    text-align: center;
    background: rgba(255, 255, 255, 0.9);
    border-radius: 20px;
    padding: 30px;
    margin: 20px 0;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
}
.card-container {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    padding: 25px;
    margin: 15px 0;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(0, 0, 0, 0.05);
}
.status-badge {
    display: inline-flex;
    align-items: center;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 500;
    margin: 5px 0;
}
.status-success {
    background: #dcfce7;
    color: #166534;
    border: 1px solid #bbf7d0;
}
.status-error {
    background: #fef2f2;
    color: #991b1b;
    border: 1px solid #fecaca;
}
.status-warning {
    background: #fffbeb;
    color: #92400e;
    border: 1px solid #fed7aa;
}
.icon-btn {
    font-size: 16px;
    margin-right: 8px;
}
.section-title {
    color: #1e40af;
    font-weight: 600;
    margin-bottom: 15px;
    font-size: 18px;
}
.result-table {
    max-height: 400px;
    overflow: auto;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
}
"""


def load_demo_config() -> dict:
    config_path = _project_root / "demo" / "demo_config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def get_uploaded_db_root() -> Path:
    cfg = load_demo_config()
    p = cfg.get("paths", {}).get("uploaded_db_root", "files/uploaded_db")
    return _project_root / p


def get_temp_data_dir() -> Path:
    cfg = load_demo_config()
    p = cfg.get("paths", {}).get("temp_data_dir", "files/temp_demo_data")
    return _project_root / p


def get_router_config_path() -> str:
    cfg = load_demo_config()
    return cfg.get("router_config", "startup_run/startup_config.json")


class SqurveDemo:
    def __init__(self, config_path: Optional[str] = None):
        config_path = config_path or get_router_config_path()
        if not Path(config_path).is_absolute():
            config_path = str(_project_root / config_path)
        self.router = Router(config_path=config_path)
        self.engine = Engine(self.router)
        logger.info("SqurveDemo initialized")

    def generate_sql(
        self,
        question: str,
        db_id: str,
        schema_path: Optional[str] = None,
        db_path: Optional[str] = None,
        use_workflow: bool = False,
        workflow_actor_lis: Optional[List] = None,
        generate_type: str = "DINSQLGenerator",
    ) -> Dict:
        if not question or not question.strip():
            return {"sql": "", "status": "error", "message": "Please provide a question"}
        if not db_id or not db_id.strip():
            return {"sql": "", "status": "error", "message": "Please select a database"}

        try:
            instance_id = str(uuid.uuid4())[:8]
            db_size = _compute_db_size_from_schema_path(schema_path or "", db_id.strip()) if schema_path else 0
            data_item = {
                "question": question.strip(),
                "db_id": db_id.strip(),
                "instance_id": instance_id,
                "db_type": "sqlite",
                "db_size": db_size,
            }

            temp_dir = get_temp_data_dir()
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_file = temp_dir / f"demo_{instance_id}.json"
            save_dataset(dataset=[data_item], new_data_source=temp_file)

            dataloader = DataLoader(self.router)
            dataloader.update_data_source(str(temp_file), "demo")

            schema_source_index = f"demo_{db_id}"
            schema_dir = Path(schema_path) if schema_path else None
            if schema_dir and schema_dir.exists():
                schema_file = schema_dir / "schema.json" if schema_dir.is_dir() else schema_dir
                if schema_file.exists():
                    dataloader.update_schema_save_source(
                        {schema_source_index: str(schema_file)},
                        multi_database=False,
                        vector_store=None,
                    )
                else:
                    dataloader.update_schema_save_source(
                        {schema_source_index: str(schema_dir)},
                        multi_database=False,
                        vector_store=None,
                    )
            else:
                return {"sql": "", "status": "error", "message": "Schema path not found"}

            if db_path:
                dataloader.set_db_path("demo", db_path)

            dataset = dataloader.generate_dataset(
                "demo",
                schema_source_index,
                is_schema_final=True,
            )
            if dataset is None:
                return {"sql": "", "status": "error", "message": "Failed to create dataset"}

            if db_path:
                dataset.db_path = db_path

            llm = self.engine.dataloader.llm

            if use_workflow and workflow_actor_lis:
                from core.actor.agent.WorkflowAgent import WorkflowAgent

                agent = WorkflowAgent(
                    dataset=dataset,
                    llm=llm,
                    actor_lis=workflow_actor_lis,
                    actor_args={},
                )
                result = agent.act(0)
            else:
                from core.task.meta.GenerateTask import GenerateTask

                task = GenerateTask(
                    llm=llm,
                    generate_type=generate_type,
                    dataset=dataset,
                    task_id=f"demo_{instance_id}",
                    eval_type=[],
                    open_parallel=False,
                    max_workers=1,
                    is_save_dataset=False,
                )
                actor = task.load_actor()
                if actor is None:
                    return {"sql": "", "status": "error", "message": f"Generator {generate_type} not found"}
                result = actor.act(0)

            sql = ""
            if isinstance(result, str):
                sql = result
            elif isinstance(result, dict):
                sql = result.get("pred_sql", result.get("sql", str(result)))
            else:
                sql = str(result)

            if sql and (sql.endswith(".sql") or "/" in sql.replace("\\", "/")):
                sql_path = Path(sql)
                if not sql_path.is_absolute():
                    sql_path = _project_root / sql_path
                if sql_path.exists() and sql_path.is_file():
                    try:
                        sql = sql_path.read_text(encoding="utf-8").strip()
                    except Exception:
                        pass

            return {"sql": sql, "status": "success", "message": "SQL generated", "instance_id": instance_id}

        except Exception as e:
            logger.exception(f"Error generating SQL: {e}")
            return {"sql": "", "status": "error", "message": str(e)}


def process_upload(files, base_root: Optional[Path] = None):
    if not files:
        return None, "No files selected"
    if not isinstance(files, list):
        files = [files]
    base_root = base_root or get_uploaded_db_root()
    paths = []
    for f in files:
        p = getattr(f, "name", f) if hasattr(f, "name") else f
        paths.append(Path(p) if isinstance(p, str) else Path(p))
    try:
        result = process_uploaded_files(paths, base_root)
        tables = result.get("schema_list", [])[:10]
        msg = (
            f"Database created: **{result['db_id']}**\n"
            f"Tables: {', '.join(tables)}"
            + ("..." if len(result.get("schema_list", [])) > 10 else "")
        )
        return result["db_id"], msg
    except Exception as e:
        logger.exception(f"Upload error: {e}")
        err_msg = str(e)
        if any("\u4e00" <= c <= "\u9fff" for c in err_msg):
            return None, "Upload failed. Please ensure files are valid .sqlite, .xlsx, or .csv format."
        return None, f"Upload failed: {err_msg}"


def _compute_db_size_from_schema_path(schema_path: str, db_id: Optional[str] = None) -> int:
    """
    Compute db_size from schema file (columns list length).
    db_size = number of columns across all tables (Spider format: column_names, excluding * placeholder).
    """
    path = Path(schema_path)
    schema_file = path / "schema.json" if path.is_dir() else path
    if not schema_file.exists():
        return 0
    try:
        data = load_dataset(schema_file)
        schemas = data if isinstance(data, list) else [data]
        for s in schemas:
            if not isinstance(s, dict):
                continue
            if db_id and s.get("db_id") != db_id:
                continue
            col_names = s.get("column_names") or s.get("column_names_original") or []
            if not col_names:
                return 0
            if len(col_names) > 1 and col_names[0][1] == "*":
                return len(col_names) - 1
            return len(col_names)
    except Exception:
        pass
    return 0


def get_available_databases() -> List[Tuple[str, str, str]]:
    """Returns [(db_id, db_path, schema_path), ...] from manifest."""
    base_root = get_uploaded_db_root()
    manifest = load_upload_manifest(base_root)
    out = []
    for e in manifest:
        db_path = e.get("db_path", "")
        if not Path(db_path).exists():
            continue
        schema_path = e.get("schema_path") or (Path(e.get("schema_base_dir", "")) / "schema.json")
        out.append((e["db_id"], db_path, str(schema_path)))
    return out


def create_demo(config_path: Optional[str] = None):
    demo_instance = SqurveDemo(config_path)
    base_root = get_uploaded_db_root()
    base_root.mkdir(parents=True, exist_ok=True)

    available_dbs = get_available_databases()
    db_choices = [d[0] for d in available_dbs]

    def on_upload(files):
        db_id, msg = process_upload(files, base_root)
        if db_id:
            dbs = get_available_databases()
            ch = [x[0] for x in dbs]
            upd = gr.update(choices=ch, value=db_id)
            return db_id, msg, upd
        upd = gr.update(choices=db_choices, value=None)
        return None, msg, upd

    def on_query(
        question,
        db_id,
        use_workflow,
        skeleton_val,
        parser_sel,
        generator_sel,
        optimizer_sel,
        decomposer_sel,
        scaler_sel,
        selector_sel,
        direct_generator,
    ):
        if not question or not db_id:
            return "", "Please provide question and select database", None, None, "⚠️", "error"

        dbs = get_available_databases()
        db_path, schema_path = None, None
        for d in dbs:
            if d[0] == db_id:
                db_path, schema_path = d[1], d[2]
                break
        if not db_path or not Path(db_path).exists():
            return "", "Database not found. Please upload first.", None, None, "❌", "error"

        workflow_actor_lis = None
        generate_type = direct_generator

        skeleton_idx = 1
        if skeleton_val:
            for i, s in enumerate(WORKFLOW_SKELETONS):
                if str(s) == str(skeleton_val):
                    skeleton_idx = i
                    break
        if use_workflow and 0 <= skeleton_idx < len(WORKFLOW_SKELETONS):
            skel = WORKFLOW_SKELETONS[skeleton_idx]
            actor_lis = []
            for t in skel:
                if t == "parser" and parser_sel:
                    actor_lis.append(parser_sel)
                elif t == "generator" and generator_sel:
                    actor_lis.append(generator_sel)
                elif t == "optimizer" and optimizer_sel:
                    actor_lis.append(optimizer_sel)
                elif t == "decomposer" and decomposer_sel:
                    actor_lis.append(decomposer_sel)
                elif t == "scaler" and scaler_sel:
                    actor_lis.append(scaler_sel)
                elif t == "selector" and selector_sel:
                    actor_lis.append(selector_sel)
            if actor_lis:
                workflow_actor_lis = actor_lis

        result = demo_instance.generate_sql(
            question=question,
            db_id=db_id,
            schema_path=schema_path,
            db_path=db_path,
            use_workflow=use_workflow and bool(workflow_actor_lis),
            workflow_actor_lis=workflow_actor_lis,
            generate_type=generate_type,
        )

        if result["status"] == "success":
            return result["sql"], "SQL generated successfully", db_path, "sqlite", "success", "success"
        else:
            msg = result["message"]
            if any("\u4e00" <= c <= "\u9fff" for c in msg):
                msg = "Generation failed. Please check your question and database, or try a different generator."
            return "", msg, None, None, "error", "error"

    def on_execute(sql, db_path, db_type):
        if not sql or not sql.strip():
            return "Please generate SQL first", None
        if not db_path:
            return "Database path not set", None
        sql_clean = sql.strip()
        try:
            result, err = get_sql_exec_result(db_type="sqlite", sql_query=sql_clean, db_path=db_path)
            if err:
                return f"Error: {err}", None
            if result is None:
                return "Query OK, 0 rows", pd.DataFrame()
            if isinstance(result, pd.DataFrame):
                row_count = len(result)
                status_msg = f"Query OK, {row_count} row{'s' if row_count != 1 else ''}"
                return status_msg, result
            # Fallback for non-DataFrame result (e.g. list of dicts)
            df = pd.DataFrame(result) if result else pd.DataFrame()
            return f"Query OK, {len(df)} rows", df
        except Exception as e:
            return str(e), None

    def on_skeleton_change(skeleton_val):
        """Show only actor dropdowns that appear in the selected workflow skeleton."""
        if not skeleton_val:
            return [gr.update(visible=True)] * 6
        skel = None
        for s in WORKFLOW_SKELETONS:
            if str(s) == skeleton_val:
                skel = s
                break
        if skel is None:
            return [gr.update(visible=True)] * 6
        return [
            gr.update(visible="parser" in skel),
            gr.update(visible="generator" in skel),
            gr.update(visible="optimizer" in skel),
            gr.update(visible="decomposer" in skel),
            gr.update(visible="scaler" in skel),
            gr.update(visible="selector" in skel),
        ]

    with gr.Blocks(title="Squrve Text-to-SQL") as demo:

        gr.Markdown(
            "## Squrve Text-to-SQL Demo\n\n"
            "Convert natural language questions into SQL queries. "
            "**Step 1:** Upload a database (.sqlite or .xlsx/.csv). "
            "**Step 2:** Select the database, enter your question, and generate SQL."
        )

        # Database selection (defined before Tabs so up_btn can update it)
        db_dropdown_q = gr.Dropdown(
            label="Database",
            choices=db_choices,
            value=db_choices[0] if db_choices else None,
            allow_custom_value=False,
            info="Select a database to query. Upload one first if the list is empty.",
        )

        with gr.Tabs():
            with gr.Tab("📤 Upload"):
                gr.Markdown(
                    "**Upload your database:**\n"
                    "- **Single .sqlite / .db file:** Upload one SQLite database; schema will be extracted automatically.\n"
                    "- **Multiple .xlsx / .csv files:** Each file becomes one table; the first row is used as column names."
                )
                file_up = gr.File(
                    label="Select files to upload",
                    file_count="multiple",
                    file_types=[".sqlite", ".db", ".xlsx", ".xls", ".csv"],
                )
                up_btn = gr.Button("Process & Create Database", variant="primary")
                up_status = gr.Markdown()
                up_db_id = gr.Textbox(label="Database ID", interactive=False)

                up_btn.click(
                    fn=on_upload,
                    inputs=[file_up],
                    outputs=[up_db_id, up_status, db_dropdown_q],
                )

            with gr.Tab("🔍 Query"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Input")
                        question = gr.Textbox(
                            label="Natural language question",
                            lines=3,
                            placeholder="e.g. How many singers are there? | List all stadiums with capacity > 5000 | What is the average age of singers from France?",
                        )

                        gr.Markdown("### Generation mode")
                        mode_radio = gr.Radio(
                            choices=["Direct Generator", "Custom Workflow"],
                            value="Direct Generator",
                            label="Mode",
                        )

                        with gr.Group(visible=False) as workflow_group:
                            gr.Markdown("**Workflow configuration**")
                            skeleton_drop = gr.Dropdown(
                                label="Workflow skeleton",
                                choices=[str(s) for s in WORKFLOW_SKELETONS],
                                value=str(WORKFLOW_SKELETONS[1]),
                            )
                            with gr.Row():
                                parser_drop = gr.Dropdown(choices=ACTOR_BY_TYPE["parser"], value=ACTOR_BY_TYPE["parser"][0], label="Parser")
                                gen_drop = gr.Dropdown(choices=ACTOR_BY_TYPE["generator"], value=ACTOR_BY_TYPE["generator"][1], label="Generator")
                            with gr.Row():
                                opt_drop = gr.Dropdown(choices=ACTOR_BY_TYPE["optimizer"], value=ACTOR_BY_TYPE["optimizer"][0], label="Optimizer")
                                dec_drop = gr.Dropdown(choices=ACTOR_BY_TYPE["decomposer"], value=ACTOR_BY_TYPE["decomposer"][0], label="Decomposer")
                            with gr.Row():
                                scaler_drop = gr.Dropdown(choices=ACTOR_BY_TYPE["scaler"], value=ACTOR_BY_TYPE["scaler"][0], label="Scaler")
                                selector_drop = gr.Dropdown(choices=ACTOR_BY_TYPE["selector"], value=ACTOR_BY_TYPE["selector"][0], label="Selector")

                        with gr.Group() as direct_group:
                            gr.Markdown("**Generator selection**")
                            direct_gen = gr.Dropdown(
                                label="Generator",
                                choices=ACTOR_BY_TYPE["generator"],
                                value="DINSQLGenerator",
                            )

                        submit_btn = gr.Button("Generate SQL", variant="primary")

                    with gr.Column(scale=1):
                        gr.Markdown("### Output")
                        sql_out = gr.Code(label="Generated SQL", language="sql", lines=8)
                        status_out = gr.Textbox(label="Status", interactive=False)
                        exec_btn = gr.Button("Execute SQL", variant="secondary")
                        exec_status = gr.Textbox(label="Execution status", interactive=False)
                        exec_result = gr.Dataframe(label="Query result (table)", interactive=False, wrap=True)

                db_path_state = gr.State()
                db_type_state = gr.State(value="sqlite")

                def on_mode_change(mode, skeleton_val):
                    use_wf = mode == "Custom Workflow"
                    wf_upd = gr.update(visible=use_wf)
                    direct_upd = gr.update(visible=not use_wf)
                    if use_wf and skeleton_val:
                        skel = None
                        for s in WORKFLOW_SKELETONS:
                            if str(s) == skeleton_val:
                                skel = s
                                break
                        if skel is not None:
                            return (
                                wf_upd, direct_upd,
                                gr.update(visible="parser" in skel),
                                gr.update(visible="generator" in skel),
                                gr.update(visible="optimizer" in skel),
                                gr.update(visible="decomposer" in skel),
                                gr.update(visible="scaler" in skel),
                                gr.update(visible="selector" in skel),
                            )
                    return wf_upd, direct_upd, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

                mode_radio.change(
                    fn=on_mode_change,
                    inputs=[mode_radio, skeleton_drop],
                    outputs=[workflow_group, direct_group, parser_drop, gen_drop, opt_drop, dec_drop, scaler_drop, selector_drop],
                )

                skeleton_drop.change(
                    fn=on_skeleton_change,
                    inputs=[skeleton_drop],
                    outputs=[parser_drop, gen_drop, opt_drop, dec_drop, scaler_drop, selector_drop],
                )

                def get_use_workflow(mode):
                    return mode == "Custom Workflow"

                def get_skeleton_idx(val):
                    for i, s in enumerate(WORKFLOW_SKELETONS):
                        if str(s) == val:
                            return i
                    return 1

                def on_query_wrapper(question_val, db_id, mode, skeleton_val, parser_sel, generator_sel, optimizer_sel, decomposer_sel, scaler_sel, selector_sel, direct_gen_val):
                    use_workflow = get_use_workflow(mode)
                    return on_query(
                        question_val, db_id, use_workflow, skeleton_val,
                        parser_sel, generator_sel, optimizer_sel, decomposer_sel, scaler_sel, selector_sel, direct_gen_val,
                    )

                submit_btn.click(
                    fn=on_query_wrapper,
                    inputs=[
                        question,
                        db_dropdown_q,
                        mode_radio,
                        skeleton_drop,
                        parser_drop,
                        gen_drop,
                        opt_drop,
                        dec_drop,
                        scaler_drop,
                        selector_drop,
                        direct_gen,
                    ],
                    outputs=[sql_out, status_out, db_path_state, db_type_state],
                )

                exec_btn.click(
                    fn=on_execute,
                    inputs=[sql_out, db_path_state, db_type_state],
                    outputs=[exec_status, exec_result],
                )

        def sync_db_dropdown():
            dbs = get_available_databases()
            ch = [x[0] for x in dbs]
            upd = gr.update(choices=ch, value=ch[0] if ch else None)
            return upd

        demo.load(fn=sync_db_dropdown, outputs=[db_dropdown_q])

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Router config path")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()

    cfg = load_demo_config()
    server = cfg.get("server", {})
    demo = create_demo(args.config)
    demo.launch(
        server_name=args.server_name or server.get("name", "0.0.0.0"),
        server_port=args.server_port or server.get("port", 7860),
        share=args.share,
        theme=DEMO_THEME,
        css=DEMO_CSS,
    )
