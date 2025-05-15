# **SheetMind: Agentic AI System for CSV/Excel Data Analysis**

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🧠 Overview

**SheetMind** is a cutting-edge agentic AI framework designed specifically for autonomous and intelligent analysis of structured tabular data such as CSV or Excel files. Inspired by GenSpark-like systems, it orchestrates a multi-agent ecosystem capable of dynamic code generation, self-correction, safe execution, and deep analytical reasoning.

Built with modularity, safety, and continual learning in mind, the architecture powers intelligent agents that can:

- Interpret user queries in natural language
- Dynamically create new agents for tasks like data cleaning, analysis, transformation, and visualization
- Execute generated code safely in sandboxed environments
- Reflect, improve, and learn from errors and feedback

---

## 🚀 Features

- **Agent-Based Architecture**: Modular design with specialized agents for different tasks
- **Safe Execution**: Sandboxed environment for secure code execution
- **Automated Data Profiling**: Comprehensive analysis of data structure and statistics
- **Natural Language Interface**: Interact with your data using natural language queries
- **Extensible**: Easily add new agents and capabilities
- **Asynchronous Processing**: Efficient handling of multiple tasks concurrently

## 🧩 Core Components

### 1. **Executive Layer – Controller Agent**

- Orchestrates tasks across all agents.
- Implements ReAct (Reason + Act) pattern.
- Maintains session and memory state.
- Decomposes user intent into executable sub-tasks.

### 2. **Agent Factory Layer**

- Dynamically generates new agents based on task needs.
- Uses Groq/Gemini for safe code generation.
- Verifies agents in a sandbox before deployment.
- Learns from past successes and failures.

### 3. **Specialized Agent Layer**

Includes:

- `DataAnalysisAgent`: Insight extraction and profiling
- `FormulaAgent`: Spreadsheet formula evaluation
- `VisualizationAgent`: Chart and graph generation
- `TransformationAgent`: Format conversions and cleaning
- `CodeExecutionAgent`: Executes validated Python/R scripts
- `NaturalLanguageAgent`: Converts NL queries into executable logic

### 4. **Secure Execution Sandbox**

- Containerized runtime (Docker or VM-based)
- Enforces CPU/memory/time limits
- Validates all inputs and outputs
- Monitors logs, metrics, and exceptions

### 5. **Feedback and Self-Reflection System**

- Detects logical/syntactic/runtime errors
- Classifies issues and formulates improvement prompts
- Maintains learning history of successful/failed attempts

---

## 🧪 Advanced AI Capabilities

- **Chain-of-Thought Reasoning**: Transparent intermediate steps.
- **Tree-of-Thought Planning**: Exploratory strategy optimization.
- **ReAct Pattern**: Think → Act → Observe → Improve loop.
- **Retrieval-Augmented Generation**: Fetch past examples to guide agents.
- **Meta-Cognitive Reflection**: Test generation, performance evaluation, and self-correction.

---

## 📦 Project Structure

```bash
SheetMind/
│
├── src/
│   ├── agents/           # Specialized agents (data, formula, viz, etc.)
│   ├── chains/           # Agent orchestration flows
│   ├── tools/            # Utilities and helper tool interfaces
│   ├── prompts/          # Prompt templates for LLMs
│   ├── retrievers/       # Retrieval logic for context and RAG
│   ├── embeddings/       # Embedding generators for data/metadata
│   ├── utils/            # Common helper functions
│   └── main.py           # Entry point for orchestrator
│
├── config/
│   └── config.yaml       # Model configs, thresholds, env settings
│
├── .env                  # Environment variables
├── .env.example          # Template env
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Project and dependency metadata
├── uv.lock               # Locked deps for reproducibility
├── README.md             # This file
└── .gitignore            # Ignore rules
```

---

## 📎 Dependency Management

This project uses **[uv](https://github.com/uv-py/uv)** for fast and modern dependency handling.

### ⚙️ Setup

```bash
# Install uv
pip install uv

# Sync dependencies
uv sync

# Add a new dependency
uv add <package-name>

# Remove a dependency
uv remove <package-name>
```

- `pyproject.toml`: Dependency definitions
- `uv.lock`: Lock file for reproducibility

---

## 🚧 Roadmap

| Phase | Goal                                                  |
| ----- | ----------------------------------------------------- |
| 1     | Build controller agent + sandbox + core agents        |
| 2     | Develop agent factory + verification system           |
| 3     | Implement feedback loop, reflection, Tree of Thoughts |
| 4     | Optimize, expand agent skills, UX refinement          |

---

## 🔐 Security Design

- **Sandboxed Code Execution** (Docker/RestrictedPython)
- **Input & Output Sanitization**
- **Resource Isolation**
- **Permission Layers**
- **Rollback & Error Recovery**

---

## 🌟 Key Features

- Built for tabular data (CSV/Excel)
- Auto-generates specialized code agents on-the-fly
- Safe execution with detailed monitoring
- Feedback-driven self-improvement
- Modular, scalable, and domain-adaptive

---

## 🚀 Use Cases

- Business data analytics
- Automated spreadsheet reporting
- Visual dashboard generation
- Data cleaning pipelines
- Natural language to analysis conversion

---

## 🛠 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/sheetmind.git
   cd sheetmind
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚦 Quick Start

### Basic Usage

```bash
# Profile a CSV file
python main.py data/example.csv

# Analyze data with a specific task
python main.py data/example.csv --task "find outliers"

# Save results to a file
python main.py data/example.csv --output results.json
```

### Available Tasks

- `profile data`: Generate a comprehensive profile of the dataset (default)
- `clean data`: Clean the dataset by handling missing values and outliers
- `analyze data`: Perform statistical analysis on the dataset
- `transform data`: Apply transformations to the dataset

## 🏗 Architecture

SheetMind is built on a multi-agent system with the following key components:

1. **Controller Agent**: Orchestrates tasks across all agents
2. **Data Analysis Agent**: Handles data profiling and statistical analysis
3. **Visualization Agent**: Creates visual representations of data
4. **Formula Agent**: Evaluates spreadsheet-like formulas
5. **Sandbox Environment**: Secure execution environment for generated code

## 📚 Documentation

For detailed documentation, please refer to the [docs](docs/) directory.

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to contribute to this project.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For any questions or feedback, please open an issue or contact the maintainers.
