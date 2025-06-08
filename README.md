# SheetMind: AI-Powered Conversational Data Analysis

SheetMind transforms the way you interact with structured data. Instead of wrestling with complex formulas or code, simply ask questions in natural language and get instant insights, explanations, and the underlying analysis code.

## The Problem We Solve

Analyzing data locked in spreadsheets (CSVs, Excel files) or other structured formats can be a significant hurdle. Many individuals and teams lack the specialized skills (like advanced Excel, SQL, or Python programming) or the time to perform in-depth data exploration. This often leads to:
-   Underutilized data and missed opportunities.
-   Time-consuming manual analysis prone to errors.
-   Reliance on data specialists, creating bottlenecks.
-   A desire for quick answers without a steep learning curve.

SheetMind aims to break down these barriers, making data analysis accessible, intuitive, and efficient for everyone.

## Our Solution: SheetMind

SheetMind is an intelligent application that allows you to:
-   **Upload your data files** (CSV, Excel, Parquet).
-   **Ask questions in plain English** (e.g., "What are the average sales per product category?", "Show me customer growth trends.").
-   **Receive comprehensive results**:
    -   A direct textual answer.
    -   A clear explanation of how the answer was derived.
    -   The Python code generated and executed for the analysis.
    -   A preview of the resulting data (if applicable).

It's designed to feel like you're conversing with a data analyst who not only gives you answers but also shows you their work.

## Key Features

-   **Natural Language Queries:** No need to learn complex query languages.
-   **Multi-Format Support:** Handles CSV, Excel (.xlsx, .xls), and Parquet files.
-   **AI-Powered Code Generation:** Leverages Google's Gemini LLM to generate Python (pandas) code dynamically.
-   **Transparent Analysis:** See the exact code used, promoting trust and learning.
-   **Step-by-Step Explanations:** Understand the "how" behind the "what."
-   **Safe Code Execution:** AI-generated code runs in an isolated sandbox environment.
-   **User-Friendly Interface:** Built with Streamlit for an intuitive web experience.
-   **Extensible Agentic Architecture:** Built with a modular system of AI agents.

## How it Works (High-Level Architecture)

SheetMind employs an agent-based system:
1.  **Streamlit UI:** Provides the interface for file upload and query input.
2.  **ControllerAgent:** Orchestrates the workflow, manages requests, and interacts with other agents.
3.  **DataAnalysisAgent:**
    -   Loads the data.
    -   Communicates with the Gemini LLM (`GeminiClient`) to understand the query and generate appropriate Python code.
    -   Executes the generated code safely in an `IsolatedPythonExecutor`.
    -   Uses the LLM again to generate explanations of the results and code.
4.  **Results Display:** The UI presents the findings from the agents.

## Current Implementation: Streamlit Application

The primary way to interact with SheetMind is through its Streamlit web application.

### Setup and Running

1.  **Prerequisites**:
    *   Python 3.8+
    *   Pip (Python package installer)

2.  **Create a Virtual Environment (Recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    Navigate to the `SheetMinds` directory (where `requirements.txt` and `streamlit_app.py` are located) and run:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Google API Key**:
    *   Create a file named `.env` in the `SheetMinds` directory (the same directory as `streamlit_app.py`).
    *   Add your Google API key to this file:
        ```
        GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
        ```
    *   Replace `"YOUR_GOOGLE_API_KEY_HERE"` with your actual Gemini API key.

5.  **Run the Streamlit Application**:
    In your terminal, from the `SheetMinds` directory, run:
    ```bash
    streamlit run streamlit_app.py
    ```
    This will start the Streamlit server, and the application should open in your web browser.

### How to Use the Streamlit App

1.  **Upload Data**: Click on "Choose a data file" to upload your CSV, Excel, or Parquet file.
2.  **Enter Query**: Type your data analysis question in the text area (e.g., "What are the average sales per product category?").
3.  **Analyze**: Click the "Analyze Data" button.
4.  **View Results**: The application will display:
    *   A textual answer to your query.
    *   An explanation of how the answer was derived.
    *   The Python code generated and executed for the analysis.
    *   A preview of the resulting data (if applicable).

## Project Structure (Simplified)

```
SheetMinds/
├── streamlit_app.py      # The main Streamlit application file
├── requirements.txt      # Python dependencies
├── .env                  # For API keys (you need to create this)
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py
│   │   ├── controller_agent.py
│   │   └── data_analysis_agent.py
│   ├── llm/
│   │   ├── __init__.py
│   │   └── gemini_client.py
│   └── tools/
│       ├── __init__.py
│       └── isolated_python_executor.py
├── API_DOCS.md           # Detailed documentation for the legacy API
└── uploads/              # Default folder for uploads (used by legacy Flask app)
```
*(Note: The `uploads/` directory and `API_DOCS.md` primarily relate to a previous Flask-based API version of SheetMind.)*

## Future Plans

SheetMind is an evolving project. Some potential future enhancements include:
-   **Enhanced Visualization Capabilities:** Integration with more advanced charting libraries for richer data storytelling.
-   **Support for More Data Sources:** Connections to databases (SQL, NoSQL), cloud storage (S3, GCS), and other data platforms.
-   **Advanced Data Cleaning & Preparation Tools:** AI-assisted data wrangling features.
-   **Proactive Insights & Anomaly Detection:** Agents that can automatically identify interesting patterns or outliers.
-   **Collaboration Features:** Allowing multiple users to work on and discuss analyses.
-   **Customizable Agent Behaviors:** More fine-grained control over how agents perform tasks.
-   **Expanded LLM Support:** Option to use other leading Large Language Models.

*(Your contributions and ideas for future development are welcome!)*

## Development

This section is for developers looking to contribute or understand the codebase more deeply.

### Project Structure (Detailed for Developers)

```
src/
  agents/          # Core logic for ControllerAgent, DataAnalysisAgent, etc.
  llm/             # Client for interacting with Large Language Models (e.g., GeminiClient).
  tools/           # Utility components like the IsolatedPythonExecutor.

streamlit_app.py   # Main entry point for the Streamlit UI.
api.py             # Legacy Flask API (consider for deprecation or separate maintenance).
requirements.txt   # Python dependencies.
.env.example       # Example for environment variables.
```

### Adding New Features (General Workflow)

1.  Create a new branch for your feature:
    ```bash
    git checkout -b feature/your-feature-name
    ```
2.  Implement your changes. Ensure to add relevant tests.
3.  Test thoroughly:
    *(Consider adding specific test commands if available, or describe manual testing procedures for the Streamlit app).*
4.  Commit your changes with a descriptive message:
    ```bash
    git add .
    git commit -m "Add your feature description"
    ```
5.  Push your changes and create a pull request against the `main` branch.

## Contributing

Contributions are welcome! Please feel free to open an issue to discuss a bug or feature, or submit a pull request with your improvements.
*(Consider creating a `CONTRIBUTING.md` file with more detailed guidelines if the project grows).*

## License

This project is licensed under the MIT License - see the `LICENSE` file for details (if one exists, otherwise state "MIT Licensed" or chosen license).

## Contact

For questions, support, or collaboration inquiries, please open an issue on the GitHub repository or contact [your-email@example.com](mailto:your-email@example.com) (replace with actual contact).

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
