<div align="right">

[English](README.md) | [中文](README_zh.md)

</div>

# Gradio Demo for Squrve

This is an interactive web interface based on Gradio for testing the Squrve Text-to-SQL framework.

## Features

- 🚀 Based on the quick start example from `startup_run/run.py`
- 💬 Interactive natural language question input
- 🗄️ Support for specifying database ID (db_id)
- 🔧 Support for multiple generator types
- 📊 Real-time display of generated SQL queries
- 📤 Upload Excel/CSV files to create databases automatically
- ▶️ Execute SQL queries and view results directly

## Installation

First, ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Basic Launch

```bash
python gradio_demo.py
```

### 2. Using Custom Configuration File

```bash
python gradio_demo.py --config startup_run/startup_config.json
```

### 3. Create Public Link (for sharing)

```bash
python gradio_demo.py --share
```

### 4. Custom Server Port

```bash
python gradio_demo.py --server-port 8080
```

## Usage Examples

### 1. Upload Data File (e.g., .csv)
- Switch to the **"📤 Upload Your Data"** tab
- Click the upload area and select your Excel or CSV file, for example: `assets/sales.csv`

```csv
Product,Price,Quantity,Date
Widget A,10.99,100,2024-01-01
Widget B,15.99,50,2024-01-02
Widget C,8.99,200,2024-01-03
```
- Click the **"Process File"** button
- The system will automatically:Convert the file to a SQLite database; Generate schema files; Display the database ID for use in queries

### 2. Enter Your Natural Language Question
- Switch to the **"💬 Query Database"** tab
- Check the **"Use uploaded database"** checkbox
- Database ID will be auto-filled
- Enter a natural language question, for example: "Find all products with price greater than 10"

### 3. Select Generator Type
- Default: DINSQLGenerator
- Options: LinkAlignGenerator, CHESSGenerator, MACSQLGenerator, etc.

### 4. Click "Generate SQL" Button
- The system will generate the corresponding SQL query
- Results will be displayed in the output area on the right

### 5. Execute SQL Query
- After SQL is generated, click the **"▶️ Execute SQL"** button
- The query will be executed against the database
- Results will be displayed in the "Execution Result" area
- You can verify if the SQL is correct by checking the execution results

## Configuration

The demo uses `startup_run/startup_config.json` as the default configuration file. Make sure:

1. **API Keys are configured**
   ```json
   {
     "api_key": {
       "qwen": "your_api_key_here",
       "deepseek": "your_api_key_here"
     }
   }
   ```

2. **Database paths are correctly set** in the configuration file

## Supported Generator Types

- `DINSQLGenerator` - DIN-SQL method
- `LinkAlignGenerator` - LinkAlign method
- `DAILSQLGenerator` - DAIL-SQL method
- `CHESSGenerator` - CHESS method
- `MACSQLGenerator` - MAC-SQL method
- `RSLSQLGenerator` - RSL-SQL method
- `ReFoRCEGenerator` - ReFoRCE method
- `OpenSearchSQLGenerator` - OpenSearchSQL method

## Development Notes

The core logic of the demo is in the `SqurveDemo` class:

- `_initialize_engine()`: Initializes Router and Engine
- `generate_sql()`: Processes a single query and generates SQL
- `execute_sql_query()`: Executes SQL queries and returns results

The code structure follows Squrve framework design patterns, using:
- `Router` for configuration management
- `Engine` for task execution
- `Dataset` for data encapsulation
- `GenerateTask` and `Actor` for SQL generation
- `get_sql_exec_result` from `core.db_connect` for SQL execution

## File Upload Support

The demo supports uploading Excel (.xlsx, .xls) or CSV (.csv) files:

- Files are automatically converted to SQLite databases
- Schema files are generated in Squrve-compatible format
- The database ID is derived from the file name (without extension)
- Uploaded databases can be used immediately for SQL generation

## Contributing

Welcome to submit Issues or Pull Requests to improve this demo!

