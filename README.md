# SheetMind Streamlit Application

This application provides a web interface for analyzing structured data files (CSV, Excel, Parquet) using AI.

## Setup and Running

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

## How to Use

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
└── uploads/              # Default folder for uploads (created by Flask app, not directly used by Streamlit temp files)
```

## Notes

*   The Streamlit app uses a temporary file for uploads, which is deleted after processing.
*   Ensure your `GOOGLE_API_KEY` is correctly set in the `.env` file for the AI analysis to work.
*   The `src` directory containing the agent logic must be in the same directory as `streamlit_app.py` or accessible via `PYTHONPATH`.


## 🚀 Quick Start

### Running the API Server

You can start the API server using the provided control script:

```bash
# Make the script executable (only needed once)
chmod +x run.sh

# Start the server
./run.sh start

# Or use the interactive menu
./run.sh
```

This will start the Flask API server on port 5000 by default. If port 5000 is in use, it will automatically find the next available port.

### API Endpoints

The API provides the following endpoints:

- `GET /api/health` - Health check
- `POST /api/upload` - Upload a file for analysis
- `POST /api/analyze` - Analyze data with a natural language query
- `POST /api/profile` - Generate a data profile

For detailed API documentation, see [API_DOCS.md](API_DOCS.md).

### Running Tests

To run the test suite:

```bash
./run.sh test
```

This will execute a series of test queries against the API.

## 📚 API Documentation

For complete API documentation, including request/response formats and examples, see [API_DOCS.md](API_DOCS.md).

## 🧪 Testing

### Automated Tests

Run the test suite to verify everything is working:

```bash
./run.sh test
```

### Manual Testing

You can use tools like `curl` or Postman to test the API endpoints:

```bash
# Health check
curl http://localhost:5000/api/health

# Upload a file
curl -X POST -F "file=@test_data/sample_data.csv" http://localhost:5000/api/upload

# Analyze data
curl -X POST -H "Content-Type: application/json" -d '{
  "query": "What is the average salary?",
  "file_path": "path/from/upload/endpoint"
}' http://localhost:5000/api/analyze
```

## 🛠 Development

### Project Structure

```
src/
  agents/          # Specialized agents (data, formula, viz, etc.)
  chains/          # Agent orchestration flows
  tools/           # Utilities and helper tool interfaces
  llm/             # LLM client implementations
  
api.py            # Main API application
test_api.py       # API test script
run.sh            # Control script for the API
requirements.txt   # Python dependencies
.env.example      # Example environment variables
```

### Adding New Features

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and test them:
   ```bash
   ./run.sh test
   ```

3. Commit your changes with a descriptive message:
   ```bash
   git add .
   git commit -m "Add your feature description"
   ```

4. Push your changes and create a pull request.

## 🤝 Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) before submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or support, please open an issue on GitHub or contact [your-email@example.com](mailto:your-email@example.com).

## 📊 Basic Usage

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
