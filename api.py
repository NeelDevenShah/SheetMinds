#!/usr/bin/env python3
"""
SheetMind API - AI-Powered Data Analysis API

This API provides endpoints for analyzing CSV data using AI-generated code.
It uses Google's Gemini for code generation and executes the code in an isolated environment.
"""
import os
import logging
import asyncio
import json
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
from functools import wraps
from datetime import datetime

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# Add the src directory to the Python path
import sys
sys.path.append(str(Path(__file__).parent.absolute()))

from src.agents.controller_agent import ControllerAgent
from src.agents.data_analysis_agent import DataAnalysisError, CodeExecutionError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Configuration
CONFIG = {
    "UPLOAD_FOLDER": "uploads",
    "ALLOWED_EXTENSIONS": {"csv", "xlsx", "xls", "parquet"},
    "MAX_CONTENT_LENGTH": 50 * 1024 * 1024,  # 50MB max file size
    "DEFAULT_PORT": 5000,
    "HOST": "0.0.0.0",
    "DEBUG": False # True
}

# Global variables
global_controller = None
app.config['UPLOAD_FOLDER'] = CONFIG["UPLOAD_FOLDER"]
app.config['MAX_CONTENT_LENGTH'] = CONFIG["MAX_CONTENT_LENGTH"]

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def async_to_sync(async_func):
    """Helper to run async functions in Flask synchronous context."""
    @wraps(async_func)
    def wrapped(*args, **kwargs):
        try:
            # Create a new event loop for this thread if it doesn't exist
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the coroutine and return its result
            return loop.run_until_complete(async_func(*args, **kwargs))
        except Exception as e:
            logger.exception("Error in async_to_sync")
            raise
    
    return wrapped


def get_controller() -> ControllerAgent:
    """Get or create the global controller instance."""
    global global_controller
    if global_controller is None:
        logger.info("Initializing ControllerAgent")
        global_controller = ControllerAgent(
            agent_id="controller_001",
            name="SheetMind Controller",
            description="Manages data analysis workflows"
        )
    return global_controller


def allowed_file(filename: str) -> bool:
    """Check if the file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in CONFIG["ALLOWED_EXTENSIONS"]


def create_error_response(message: str, status_code: int = 400, **kwargs) -> Response:
    """Create a standardized error response."""
    response = {
        "success": False,
        "error": message,
        "timestamp": datetime.utcnow().isoformat(),
        **kwargs
    }
    return jsonify(response), status_code


def create_success_response(data: Any, status_code: int = 200, **kwargs) -> Response:
    """Create a standardized success response."""
    response = {
        "success": True,
        "data": data,
        "timestamp": datetime.utcnow().isoformat(),
        **kwargs
    }
    return jsonify(response), status_code

@app.route('/api/upload', methods=['POST'])
@async_to_sync
async def upload_file():
    """
    Upload a file to the server.
    
    Returns:
        JSON response with the file path and metadata
    """
    # Check if the post request has the file part
    if 'file' not in request.files:
        return create_error_response("No file part in the request")
    
    file = request.files['file']
    
    # If user does not select file, browser might
    # submit an empty part without filename
    if file.filename == '':
        return create_error_response("No selected file")
    
    if file and allowed_file(file.filename):
        # Ensure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Create a unique filename to prevent overwriting
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4()}.{file_ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get file info
        file_size = os.path.getsize(filepath)
        
        return create_success_response({
            "filename": filename,
            "filepath": filepath,
            "size": file_size,
            "content_type": file.content_type
        })
    
    return create_error_response("File type not allowed")

@app.route("/api/analyze", methods=["POST"])
@async_to_sync
async def analyze_data():
    """
    Analyze data based on the provided query.
    
    Request JSON format:
    {
        "query": "Your analysis question",
        "file_path": "/path/to/your/file.csv"
    }
    
    Returns:
        JSON response with analysis results
    """
    # Get and validate request data
    data = request.get_json()
    
    if not data:
        return create_error_response("No JSON data provided")
    
    query = data.get("query")
    file_path = data.get("file_path")
    
    if not query or not isinstance(query, str):
        return create_error_response("Query is required and must be a string")
    
    if not file_path or not isinstance(file_path, str):
        return create_error_response("file_path is required and must be a string")
    
    # Check if file exists
    if not os.path.exists(file_path):
        return create_error_response(f"File not found: {file_path}", status_code=404)
    
    # Get the controller
    controller = get_controller()
    
    # Execute the analysis
    result = await controller.execute(
        task="analyze",
        query=query,
        data_path=file_path
    )
    
    # Return the result
    if result.success:
        return create_success_response({
            "result": result.output.get("result"),
            "explanation": result.output.get("explanation"),
            "code": result.output.get("code"),
            "data_preview": result.output.get("data_preview")
        })
    else:
        return create_error_response(
            result.error or "Analysis failed",
            status_code=500
        )

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return create_error_response("The requested resource was not found", status_code=404)


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return create_error_response("An internal server error occurred", status_code=500)


if __name__ == "__main__":
    # Print API documentation
    print("\n" + "="*50)
    print("SheetMind API Server")
    print("="*50)
    print("\nAvailable endpoints:")
    print("  POST /api/upload          - Upload a file")
    print("  POST /api/analyze         - Analyze data with a query")
    print("\nConfiguration:")
    print(f"  Host: {CONFIG['HOST']}")
    print(f"  Port: {CONFIG['DEFAULT_PORT']}")
    print(f"  Debug: {CONFIG['DEBUG']}")
    print("\nStarting server...\n")
    
    # Run the Flask app
    app.run(
        host=CONFIG["HOST"],
        port=CONFIG["DEFAULT_PORT"],
        debug=CONFIG["DEBUG"]
    )

