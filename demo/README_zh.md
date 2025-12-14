<div align="right">

[English](README.md) | [中文](README_zh.md)

</div>

# Gradio Demo for Squrve

这是一个基于 Gradio 的交互式 Web 界面，用于测试 Squrve Text-to-SQL 框架。

## 功能特点

- 🚀 基于 `startup_run/run.py` 的快速启动示例
- 💬 交互式自然语言问题输入
- 🗄️ 支持指定数据库 ID (db_id)
- 🔧 支持多种生成器类型选择
- 📊 实时显示生成的 SQL 查询

## 安装依赖

首先确保已安装所有依赖：


```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 基本启动

```bash
python gradio_demo.py
```

### 2. 使用自定义配置文件

```bash
python gradio_demo.py --config startup_run/startup_config.json
```

### 3. 创建公共链接（用于分享）

```bash
python gradio_demo.py --share
```

### 4. 自定义服务器端口

```bash
python gradio_demo.py --server-port 8080
```

## 使用示例
1. **上传数据文件（如.csv）**
- 切换到 **"📤 Upload Your Data"** 标签页
- 点击上传区域，选择你的 Excel 或 CSV 文件，例如：`assets/sales.csv` 

```csv
Product,Price,Quantity,Date
Widget A,10.99,100,2024-01-01
Widget B,15.99,50,2024-01-02
Widget C,8.99,200,2024-01-03
```
- 点击 **"Process File"** 按钮

2. **输入你的自然语言问题**
- 切换到 **"💬 Query Database"** 标签页
- 勾选 **"Use uploaded database"** 复选框
- Database ID 会自动填充
- 输入自然语言问题，例如："Find all products with price greater than 10"

3. **选择生成器类型**
- 默认：DINSQLGenerator
- 可选：LinkAlignGenerator, CHESSGenerator, MACSQLGenerator 等

4. **点击 "Generate SQL" 按钮**
- 系统将生成对应的 SQL 查询
- 结果会显示在右侧的输出区域

## 配置说明

Demo 使用 `startup_run/startup_config.json` 作为默认配置文件。确保：

1. **API Keys 已配置**
   ```json
   {
     "api_key": {
       "qwen": "your_api_key_here",
       "deepseek": "your_api_key_here"
     }
   }
   ```


## 支持的生成器类型

- `DINSQLGenerator` - DIN-SQL 方法
- `LinkAlignGenerator` - LinkAlign 方法
- `DAILSQLGenerator` - DAIL-SQL 方法
- `CHESSGenerator` - CHESS 方法
- `MACSQLGenerator` - MAC-SQL 方法
- `RSLSQLGenerator` - RSL-SQL 方法
- `ReFoRCEGenerator` - ReFoRCE 方法
- `OpenSearchSQLGenerator` - OpenSearchSQL 方法

## 开发说明

Demo 的核心逻辑在 `SqurveDemo` 类中：

- `_initialize_engine()`: 初始化 Router 和 Engine
- `generate_sql()`: 处理单个查询并生成 SQL

代码结构遵循 Squrve 框架的设计模式，使用：
- `Router` 管理配置
- `Engine` 管理任务执行
- `Dataset` 封装数据
- `GenerateTask` 和 `Actor` 执行 SQL 生成

## 贡献

欢迎提交 Issue 或 Pull Request 来改进这个 Demo！
