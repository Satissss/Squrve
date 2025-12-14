"""
Gradio Demo for Squrve Text-to-SQL Framework

This demo provides an interactive web interface for testing Text-to-SQL generation
based on the quick start example in startup_run/run.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent  # demo/gradio_demo.py -> project root

# # Add project root to Python path
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import gradio as gr
import json
import uuid
from typing import Optional, Dict, List
from loguru import logger

from core.base import Router
from core.engine import Engine
from core.data_manage import Dataset
from core.file_to_db import process_uploaded_file
from core.path_utils import get_project_root
from core.utils import save_dataset
from core.db_connect import get_sql_exec_result
import pandas as pd


class SqurveDemo:
    """Wrapper class for Squrve Text-to-SQL demo"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the demo with configuration
        
        Args:
            config_path: Path to configuration file (defaults to startup_config.json)
        """
        if config_path is None:
            config_path = "startup_run/startup_config.json"
        
        self.config_path = config_path
        self.router = None
        self.engine = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize Router and Engine from config"""
        try:
            self.router = Router(config_path=self.config_path)
            self.engine = Engine(self.router)
            logger.info("Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize engine: {e}")
            raise
    
    def generate_sql(
        self,
        question: str,
        db_id: str,
        schema_source: Optional[str] = None,
        generate_type: str = "DINSQLGenerator",
        db_path: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate SQL for a single question
        
        Args:
            question: Natural language question
            db_id: Database identifier
            schema_source: Schema source path (optional, uses router default if not provided)
            generate_type: Type of generator to use
            
        Returns:
            Dictionary with generated SQL and status information
        """
        if not question or not question.strip():
            return {
                "sql": "",
                "status": "error",
                "message": "Please provide a question"
            }
        
        if not db_id or not db_id.strip():
            return {
                "sql": "",
                "status": "error",
                "message": "Please provide a database ID"
            }
        
        try:
            # Create a single-item dataset
            instance_id = str(uuid.uuid4())[:8]
            data_item = {
                "question": question.strip(),
                "db_id": db_id.strip(),
                "instance_id": instance_id,
                "db_type": "sqlite"  # Default database type, can be overridden if needed
            }
            
            # Use DataLoader to properly handle schema_source
            from core.data_manage import DataLoader
            dataloader = DataLoader(self.router)
            
            # Save data item to a temporary JSON file and register it
            # This is required because update_data_source expects a file path
            temp_data_dir = get_project_root() / "files" / "temp_demo_data"
            temp_data_dir.mkdir(parents=True, exist_ok=True)
            temp_data_file = temp_data_dir / f"demo_{instance_id}.json"
            save_dataset(dataset=[data_item], new_data_source=temp_data_file)
            
            # Register the demo data source with the file path
            dataloader.update_data_source(str(temp_data_file), "demo")
            
            # Handle schema_source
            if schema_source is None:
                schema_source = self.router.schema_source
                logger.debug(f"Using default schema_source from router: {schema_source}")
            else:
                logger.debug(f"Using provided schema_source: {schema_source}")
            
            # Get schema source path from dataloader
            schema_source_index = schema_source
            
            # Check if schema_source is a directory path (for uploaded databases)
            # Convert to absolute path if it's a relative path
            schema_path = None
            if schema_source:
                schema_path = Path(schema_source)
                # Resolve to absolute path if it's relative
                if not schema_path.is_absolute():
                    schema_path = get_project_root() / schema_path
                schema_path = schema_path.resolve()
            
            logger.debug(f"Resolved schema_path: {schema_path}, exists: {schema_path.exists() if schema_path else None}, is_dir: {schema_path.is_dir() if schema_path else None}")
            
            if schema_path and schema_path.exists() and schema_path.is_dir():
                # Custom schema directory from uploaded file
                # For uploaded databases, schema files are in schema_base_dir/db_id/
                # (e.g., schemas/sales/sales/)
                # We pass schema_base_dir (schemas/sales) and set is_schema_final=True
                # to skip the automatic single_db suffix. Then get_db_schema will append
                # db_id to get schemas/sales/sales/
                schema_source_index = f"uploaded_{db_id}"
                # Pass schema_base_dir (schemas/{db_id}), not the full path
                # update_schema_save_source accepts dict format: {index: path}
                dataloader.update_schema_save_source(
                    {schema_source_index: str(schema_path)},
                    multi_database=False,
                    vector_store=None
                )
                logger.info(f"Registered uploaded schema - index: {schema_source_index}, path: {schema_path}")
                # Set is_schema_final=True to skip automatic single_db suffix
                is_schema_final = True
            elif isinstance(schema_source, str) and ":" in schema_source:
                # Handle format like "spider:dev"
                file_name_ = "_".join(schema_source.split(":"))
                save_schema_source = Path(dataloader.schema_source_dir) / file_name_
                if not dataloader.skip_schema_init:
                    dataloader.init_benchmark_schema(schema_source, dataloader.multi_database,
                                                     save_schema_source=save_schema_source)
                dataloader.update_schema_save_source({file_name_: str(save_schema_source)}, 
                                                    dataloader.multi_database,
                                                    dataloader.vector_store)
                schema_source_index = file_name_
                
                # Get database path from benchmark
                parts = schema_source.split(":")
                if len(parts) >= 2:
                    benchmark_id = parts[0]
                    sub_id = parts[1] if len(parts) > 1 else ""
                    db_path = self.router.get_benchmark_db_path(benchmark_id, sub_id)
                    if db_path:
                        dataloader.set_db_path("demo", db_path)
            
            # Set custom database path if provided
            if db_path:
                dataloader.set_db_path("demo", db_path)
            
            # Generate dataset using DataLoader
            # For uploaded databases, set is_schema_final=True to skip single_db suffix
            is_schema_final_flag = is_schema_final if 'is_schema_final' in locals() else False
            dataset = dataloader.generate_dataset(
                "demo", 
                schema_source_index,
                is_schema_final=is_schema_final_flag
            )
            
            if dataset is None:
                return {
                    "sql": "",
                    "status": "error",
                    "message": "Failed to create dataset. Please check db_id and schema_source configuration."
                }
            
            # Override dataset db_path if custom path provided
            if db_path:
                dataset.db_path = db_path
            
            # Create task
            from core.task.meta.GenerateTask import GenerateTask
            task = GenerateTask(
                llm=self.engine.dataloader.llm,
                generate_type=generate_type,
                dataset=dataset,
                task_id=f"demo_{instance_id}",
                eval_type=[],
                open_parallel=False,
                max_workers=1,
                is_save_dataset=False
            )
            
            # Execute the task for the single item
            actor = task.load_actor()
            if actor is None:
                return {
                    "sql": "",
                    "status": "error",
                    "message": f"Failed to load generator: {generate_type}"
                }
            
            # Generate SQL
            result = actor.act(0)  # Act on the first (and only) item
            
            # Extract SQL from result
            if isinstance(result, str):
                sql = result
            elif isinstance(result, dict):
                sql = result.get("pred_sql", result.get("sql", str(result)))
            else:
                sql = str(result)
            
            # Check if SQL is actually a file path (generators may save SQL to file and return path)
            # If it looks like a file path, read the file content
            if sql and (sql.endswith(".sql") or "../" in sql or "/" in sql.replace("\\", "/")):
                sql_path = Path(sql)
                if not sql_path.is_absolute():
                    sql_path = get_project_root() / sql_path
                if sql_path.exists() and sql_path.is_file():
                    try:
                        sql = sql_path.read_text(encoding="utf-8").strip()
                        logger.debug(f"Read SQL from file: {sql_path}")
                    except Exception as e:
                        logger.warning(f"Failed to read SQL from file {sql_path}: {e}")
                        # Keep the original value if file read fails
            
            # Also check dataset, but only if SQL is still empty or looks like a path
            if (not sql or sql.endswith(".sql")) and hasattr(dataset, '_dataset') and len(dataset._dataset) > 0:
                item = dataset._dataset[0]
                dataset_sql = item.get("pred_sql", "")
                if dataset_sql and not sql.endswith(".sql"):
                    # Only use dataset SQL if current SQL is empty or is a file path
                    if not sql or sql.endswith(".sql"):
                        # Check if dataset_sql is also a file path
                        if dataset_sql.endswith(".sql"):
                            dataset_sql_path = Path(dataset_sql)
                            if not dataset_sql_path.is_absolute():
                                dataset_sql_path = get_project_root() / dataset_sql_path
                            if dataset_sql_path.exists():
                                try:
                                    sql = dataset_sql_path.read_text(encoding="utf-8").strip()
                                    logger.debug(f"Read SQL from dataset file: {dataset_sql_path}")
                                except Exception as e:
                                    logger.warning(f"Failed to read SQL from dataset file {dataset_sql_path}: {e}")
                        else:
                            sql = dataset_sql
            
            return {
                "sql": sql,
                "status": "success",
                "message": f"Successfully generated SQL using {generate_type}",
                "instance_id": instance_id
            }
            
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            import traceback
            traceback.print_exc()
            return {
                "sql": "",
                "status": "error",
                "message": f"Error: {str(e)}"
            }


def create_demo(config_path: Optional[str] = None):
    """Create and launch the Gradio interface"""
    
    # Initialize demo
    demo_instance = SqurveDemo(config_path)
    
    # Get available generators from the config or use defaults
    available_generators = [
        "DINSQLGenerator",
        "LinkAlignGenerator",
        "DAILSQLGenerator",
        "CHESSGenerator",
        "MACSQLGenerator",
        "RSLSQLGenerator",
        "ReFoRCEGenerator",
        "OpenSearchSQLGenerator"
    ]
    
    # Store uploaded database info
    uploaded_db_info = {}
    
    def process_file_upload(file):
        """Process uploaded Excel/CSV file"""
        if file is None:
            return None, "No file uploaded"
        
        try:
            # Process the uploaded file
            result = process_uploaded_file(file.name)
            
            # Store for later use
            uploaded_db_info[result["db_id"]] = result
            
            status_msg = (
                f"✅ Database created successfully!\n\n"
                f"**Database ID:** {result['db_id']}\n"
                f"**Table Name:** {result['table_name']}\n"
                f"**Columns:** {len(result['schema_list'])}\n"
                f"**Database Path:** {result['db_path']}\n\n"
                f"You can now use database ID '{result['db_id']}' in the Query tab."
            )
            
            return result["db_id"], status_msg
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            import traceback
            traceback.print_exc()
            return None, f"❌ Error processing file: {str(e)}"
    
    def process_query(question: str, db_id: str, generator_type: str, use_uploaded: bool):
        """Process a single query"""
        # If using uploaded database, get its info
        db_path = None
        schema_source = None
        
        if use_uploaded and db_id in uploaded_db_info:
            db_info = uploaded_db_info[db_id]
            db_path = db_info["db_path"]
            # Use the schema base directory as schema_source
            # Dataset.get_db_schema will look for schema_source / db_id when multi_database=False
            schema_source = db_info.get("schema_base_dir", db_info["schema_dir"])
            logger.info(f"Using uploaded database - db_id: {db_id}, schema_source: {schema_source}, db_path: {db_path}")
        else:
            logger.info(f"Using benchmark database - db_id: {db_id}, use_uploaded: {use_uploaded}, in uploaded_db_info: {db_id in uploaded_db_info if 'uploaded_db_info' in globals() else 'N/A'}")
        
        result = demo_instance.generate_sql(
            question, 
            db_id, 
            schema_source=schema_source,
            generate_type=generator_type,
            db_path=db_path
        )
        
        if result["status"] == "success":
            return result["sql"], f"✅ {result['message']}", db_path, "sqlite"
        else:
            return "", f"❌ {result['message']}", None, None
    
    def execute_sql_query(sql: str, db_path: str, db_type: str):
        """Execute SQL query and return results"""
        if not sql or not sql.strip():
            return "⚠️ 请先生成 SQL 查询", None
        
        if not db_path:
            return "⚠️ 数据库路径未设置，无法执行 SQL", None
        
        # Clean SQL: remove any file path references if accidentally included
        sql_clean = sql.strip()
        
        # Check if SQL looks like a file path (contains path separators)
        if sql_clean.startswith("../") or sql_clean.startswith("./") or "/" in sql_clean or "\\" in sql_clean:
            # If it looks like a file path, try to read the file
            if sql_clean.endswith(".sql"):
                try:
                    from pathlib import Path
                    sql_file = Path(sql_clean)
                    if not sql_file.is_absolute():
                        sql_file = get_project_root() / sql_file
                    if sql_file.exists():
                        sql_clean = sql_file.read_text(encoding="utf-8").strip()
                        logger.info(f"Read SQL from file: {sql_file}")
                    else:
                        return f"⚠️ SQL 文件不存在: {sql_clean}", None
                except Exception as e:
                    logger.warning(f"Failed to read SQL from file {sql_clean}: {e}")
                    return f"⚠️ 无法读取 SQL 文件: {sql_clean}\n错误: {str(e)}", None
        
        logger.debug(f"Executing SQL (length={len(sql_clean)}): {sql_clean[:200]}...")
        logger.debug(f"Database path: {db_path}, type: {db_type}")
        
        try:
            # Execute SQL using Squrve's db_connect module
            if db_type == "sqlite":
                result, error = get_sql_exec_result(
                    db_type="sqlite",
                    sql_query=sql_clean,
                    db_path=db_path
                )
            else:
                return f"⚠️ 暂不支持 {db_type} 数据库类型的执行", None
            
            if error:
                return f"❌ SQL 执行错误: {error}", None
            
            if result is None:
                return "⚠️ 查询返回空结果", None
            
            if isinstance(result, pd.DataFrame):
                if result.empty:
                    return "✅ 查询成功，但结果为空（0 行）", None
                else:
                    # Format result as markdown table
                    result_str = f"✅ 查询成功！返回 {len(result)} 行数据\n\n"
                    result_str += result.to_markdown(index=False)
                    return result_str, result
            else:
                return f"✅ 查询成功！\n结果: {result}", None
                
        except Exception as e:
            logger.error(f"Error executing SQL: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ 执行 SQL 时发生错误: {str(e)}", None
    
    # Create Gradio interface
    with gr.Blocks(title="Squrve Text-to-SQL Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🚀 Squrve Text-to-SQL Demo
        
        This demo allows you to test the Squrve Text-to-SQL framework interactively.
        
        **Two ways to use:**
        1. **Upload your own data**: Upload an Excel (.xlsx) or CSV file to automatically create a database
        2. **Use existing database**: Enter a database ID from existing benchmarks
        
        **Based on the quick start example from `startup_run/run.py`**
        """)
        
        with gr.Tabs():
            with gr.Tab("📤 Upload Your Data"):
                gr.Markdown("### Upload Excel or CSV File")
                file_upload = gr.File(
                    label="Upload File (Excel .xlsx/.xls or CSV .csv)",
                    file_types=[".xlsx", ".xls", ".csv"]
                )
                upload_btn = gr.Button("Process File", variant="primary")
                upload_status = gr.Textbox(
                    label="Upload Status",
                    interactive=False,
                    lines=3
                )
                uploaded_db_id = gr.Textbox(
                    label="Database ID (auto-generated)",
                    interactive=False
                )
                
                upload_btn.click(
                    fn=process_file_upload,
                    inputs=[file_upload],
                    outputs=[uploaded_db_id, upload_status]
                )
            
            with gr.Tab("💬 Query Database"):
                with gr.Row():
                    with gr.Column(scale=2):
                        use_uploaded_checkbox = gr.Checkbox(
                            label="Use uploaded database",
                            value=False
                        )
                        
                        question_input = gr.Textbox(
                            label="Natural Language Question",
                            placeholder="e.g., Find all records where price is greater than 100",
                            lines=3
                        )
                        
                        db_id_input = gr.Textbox(
                            label="Database ID (db_id)",
                            placeholder="Enter database ID or use uploaded database",
                            lines=1
                        )
                        
                        generator_dropdown = gr.Dropdown(
                            choices=available_generators,
                            value="DINSQLGenerator",
                            label="Generator Type"
                        )
                        
                        submit_btn = gr.Button("Generate SQL", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        sql_output = gr.Code(
                            label="Generated SQL",
                            language="sql",
                            lines=10
                        )
                        
                        status_output = gr.Textbox(
                            label="Status",
                            interactive=False
                        )
                        
                        execute_btn = gr.Button("▶️ Execute SQL", variant="secondary", size="lg")
                        
                        execution_result = gr.Textbox(
                            label="Execution Result",
                            interactive=False,
                            lines=15,
                            placeholder="SQL 执行结果将显示在这里..."
                        )
                        
                        # Hidden components to store db_path and db_type
                        db_path_hidden = gr.State(value=None)
                        db_type_hidden = gr.State(value=None)
        
        # Event handlers for query tab
        submit_btn.click(
            fn=process_query,
            inputs=[question_input, db_id_input, generator_dropdown, use_uploaded_checkbox],
            outputs=[sql_output, status_output, db_path_hidden, db_type_hidden]
        )
        
        # Allow Enter key to submit
        question_input.submit(
            fn=process_query,
            inputs=[question_input, db_id_input, generator_dropdown, use_uploaded_checkbox],
            outputs=[sql_output, status_output, db_path_hidden, db_type_hidden]
        )
        
        # Execute SQL button handler
        def execute_sql_wrapper(sql, db_path, db_type):
            """Wrapper function for execute_sql_query"""
            result_text, result_df = execute_sql_query(sql, db_path, db_type)
            return result_text
        
        execute_btn.click(
            fn=execute_sql_wrapper,
            inputs=[sql_output, db_path_hidden, db_type_hidden],
            outputs=[execution_result]
        )
        
        # Auto-fill db_id when checkbox is checked
        def update_db_id(use_uploaded, current_db_id):
            if use_uploaded and uploaded_db_info:
                # Use the most recently uploaded database
                latest_db_id = list(uploaded_db_info.keys())[-1]
                return latest_db_id
            return current_db_id
        
        use_uploaded_checkbox.change(
            fn=update_db_id,
            inputs=[use_uploaded_checkbox, db_id_input],
            outputs=[db_id_input]
        )
    
    return demo


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Squrve Gradio Demo")
    parser.add_argument(
        "--config",
        type=str,
        default="startup_run/startup_config.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link"
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="0.0.0.0",
        help="Server name"
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="Server port"
    )
    
    args = parser.parse_args()
    
    # Create and launch demo
    demo = create_demo(config_path=args.config)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share
    )

