#!/usr/bin/env python3
"""
SheetMind API - Simplified REST API for CSV data analysis
"""
import os
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any
from functools import wraps

from flask import Flask, request, jsonify
from flask_cors import CORS

# Add the src directory to the Python path
import sys
sys.path.append(str(Path(__file__).parent.absolute()))

from src.agents.controller_agent import ControllerAgent
from src.agents.data_analysis_agent import DataAnalysisAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Global variable to store the CSV file path
csv_file_path = None
# Global controller agent
controller = None
# Global event loop
loop = None


def async_to_sync(async_func):
    """Helper to run async functions in Flask synchronous context."""
    global loop
    if loop is None:
        # Create a new event loop if none exists
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    @wraps(async_func)
    def wrapped(*args, **kwargs):
        return loop.run_until_complete(async_func(*args, **kwargs))
    
    return wrapped


def init_controller():
    """Initialize the controller if not already initialized."""
    global controller
    if controller is None:
        controller = ControllerAgent(
            agent_id="controller_001",
            name="SheetMind Controller",
            description="Simple controller for CSV analysis"
        )
    return controller


@app.route("/api/set_csv_path", methods=["POST"])
def set_csv_path():
    """Set the path to the CSV file."""
    data = request.json
    if "path" not in data:
        return jsonify({"success": False, "error": "No path provided"}), 400

    global csv_file_path
    csv_file_path = data["path"]

    if not os.path.exists(csv_file_path):
        return jsonify({"success": False, "error": f"File not found: {csv_file_path}"}), 404

    return jsonify({
        "success": True,
        "message": f"CSV file path set to: {csv_file_path}"
    })


@app.route("/api/ask", methods=["POST"])
def ask_question():
    """Ask a question about the CSV data."""
    data = request.json
    if "question" not in data:
        return jsonify({"success": False, "error": "No question provided"}), 400

    question = data["question"]
    
    # Check if CSV path is set
    if not csv_file_path:
        return jsonify({"success": False, "error": "CSV file path not set. Please set a path first."}), 400

    if not os.path.exists(csv_file_path):
        return jsonify({"success": False, "error": f"File not found: {csv_file_path}"}), 404

    try:
        # Initialize the controller
        agent = init_controller()
        
        # Call the async function to execute the query
        result = execute_query(question, csv_file_path)
        
        # Return the results
        if result.success:
            return jsonify({"success": True, "result": result.output})
        else:
            return jsonify({"success": False, "error": result.error}), 500
            
    except Exception as e:
        logger.exception("Error processing question")
        return jsonify({"success": False, "error": str(e)}), 500


@async_to_sync
async def execute_query(question, file_path):
    """Execute a query using the controller agent."""
    # Get the controller
    agent = init_controller()
    
    # Execute the query and return the result
    return await agent.execute(
        task="custom analysis",
        query=question,
        data_path=file_path
    )


if __name__ == "__main__":
    # Apply a custom analysis patch if needed for data analysis agent
    async def custom_analysis_patch(self, query: str, **kwargs):
        """Custom analysis implementation for DataAnalysisAgent."""
        data_path = kwargs.get("data_path")
        if not data_path:
            return self._create_error_response("No data path provided for custom analysis")
        
        try:
            data = await self._load_data(data_path)
            
            # Generate a response to the query
            response = {
                "query": query,
                "answer": f"Analysis of {data_path} based on query: {query}",
                "data_summary": {
                    "rows": len(data),
                    "columns": len(data.columns),
                    "column_names": list(data.columns),
                    "sample_data": data.head(5).to_dict(orient="records")
                }
            }
            
            return self._create_success_response(response)
        except Exception as e:
            return self._create_error_response(f"Error in custom analysis: {str(e)}")
    
    # Apply the patch
    DataAnalysisAgent._custom_analysis = custom_analysis_patch
    
    # Run the Flask app
    print("Starting SheetMind API server...")
    print("Use the following endpoints:")
    print("  POST /api/set_csv_path - Set the CSV file path")
    print("  POST /api/ask - Ask a question about the CSV data")
    app.run(host="0.0.0.0", port=5000, debug=True)

