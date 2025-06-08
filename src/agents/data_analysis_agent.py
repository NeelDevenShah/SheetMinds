import pandas as pd
import numpy as np
import json
import logging
import sys
import traceback
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import textwrap
import tempfile
import shutil
import os

from .base_agent import BaseAgent, AgentResponse
from ..tools.isolated_python_executor import IsolatedPythonExecutor
from ..llm.gemini_client import GeminiClient, CodeGenerationResult

logger = logging.getLogger(__name__)


class DataAnalysisError(Exception):
    """Custom exception for data analysis errors."""
    pass


class CodeExecutionError(Exception):
    """Custom exception for code execution errors."""
    pass

class DataAnalysisAgent(BaseAgent):
    """
    Specialized agent for performing data analysis tasks on tabular data.
    Handles tasks like data profiling, statistical analysis, and data cleaning.
    Uses Gemini for code generation and isolated execution for safety.
    """
    
    def __init__(self, 
                 agent_id: str,
                 name: str = "Data Analysis Agent",
                 description: str = "Performs data analysis and profiling on tabular data",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the DataAnalysisAgent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name
            description: Description of the agent's purpose
            config: Configuration dictionary
        """
        default_config = {
            "max_rows_to_display": 10,
            "max_columns_to_display": 10,
            "profile_sample_size": 1000,
            "allowed_file_types": ["csv", "xlsx", "xls", "parquet"],
            "max_file_size_mb": 50,
            "default_encoding": "utf-8",
            "sandbox_timeout": 300,  # seconds
            "sandbox_memory_limit_mb": 1024,
            "max_code_execution_attempts": 3,
            "max_code_length": 5000,
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(agent_id, name, description, default_config)
        
        # Initialize the isolated Python executor
        self.executor = IsolatedPythonExecutor("data_analysis_sandbox")
        
        # Initialize Gemini client
        self.llm_client = GeminiClient()
        
        # Cache for storing loaded data
        self._data_cache = {}
        self.initialized = True
    
    def _create_response(
        self,
        success: bool,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        traceback: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a standardized response dictionary.
        
        Args:
            success: Whether the operation was successful
            result: Result data (if successful)
            error: Error message (if failed)
            traceback: Traceback information (if failed)
            
        Returns:
            Dictionary with the response data
        """
        response = {
            "success": success,
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "agent_id": self.agent_id,
            "agent_name": self.name
        }
        
        if success and result is not None:
            response["result"] = result
        elif not success:
            response["error"] = error
            if traceback:
                response["traceback"] = traceback
        
        return response
    
    def _get_traceback(self) -> str:
        """Get the current exception traceback as a string."""
        return traceback.format_exc()
    
    async def _load_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from a file into a pandas DataFrame with caching.
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments to pass to the pandas read function
            
        Returns:
            Loaded DataFrame
            
        Raises:
            DataAnalysisError: If there's an error loading the data
        """
        cache_key = f"{file_path}_{str(kwargs)}"
        
        # Return cached data if available
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]
            
        file_path = Path(file_path)
        
        # Check file exists
        if not file_path.exists():
            raise DataAnalysisError(f"File not found: {file_path}")
            
        # Check file extension
        file_ext = file_path.suffix.lower()[1:]  # Remove the dot
        if file_ext not in self.config["allowed_file_types"]:
            raise DataAnalysisError(
                f"Unsupported file type: {file_ext}. "
                f"Supported types: {', '.join(self.config['allowed_file_types'])}"
            )
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config["max_file_size_mb"]:
            raise DataAnalysisError(
                f"File size ({file_size_mb:.2f} MB) exceeds the maximum allowed size "
                f"({self.config['max_file_size_mb']} MB)"
            )
        
        # Read the file based on its type
        if file_ext == 'csv':
            df = pd.read_csv(file_path, **kwargs)
        elif file_ext in ('xlsx', 'xls'):
            df = pd.read_excel(file_path, **kwargs)
        elif file_ext == 'parquet':
            df = pd.read_parquet(file_path, **kwargs)
        else:
            raise DataAnalysisError(f"Unhandled file type: {file_ext}")
            
        # Cache the loaded data
        self._data_cache[cache_key] = df
        return df
    
    async def _get_data_preview(self, df: pd.DataFrame, max_rows: int = 5) -> Dict[str, Any]:
        """Create a preview of the data for code generation.
        
        Args:
            df: DataFrame to create preview for
            max_rows: Maximum number of rows to include in preview
            
        Returns:
            Dictionary with data preview
        """
        preview = {
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_data": df.head(max_rows).to_dict(orient='records'),
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
        }
        
        # Add basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if not numeric_cols.empty:
            preview["numeric_stats"] = df[numeric_cols].describe().to_dict()
            
        return preview
        
    async def _generate_analysis_code(
        self,
        query: str,
        data_preview: Dict[str, Any],
        max_attempts: int = 3
    ) -> CodeGenerationResult:
        logger.info(f"{self.agent_id}: Entering _generate_analysis_code for query: '{query[:50]}...'")
        """Generate analysis code using Gemini.
        
        Args:
            query: User's query about the data
            data_preview: Preview of the data for context
            max_attempts: Maximum number of generation attempts
            
        Returns:
            CodeGenerationResult containing the generated code and metadata
        """
        # CodeGenerationResult is imported at the top of the file (from ..llm.gemini_client import ...)
        
        attempt = 0
        last_error = "Unknown error"
        code_gen_result: Optional[CodeGenerationResult] = None # Ensure it's defined for final return

        while attempt < max_attempts:
            self.logger.info(f"{self.agent_id}: Generating analysis code (attempt {attempt + 1}/{max_attempts})...")
            
            logger.info(f"{self.agent_id}: Calling llm_client.generate_analysis_code in _generate_analysis_code. Attempt {attempt + 1}")
            try:
                code_gen_result = await self.llm_client.generate_analysis_code(
                    query=query,
                    data_preview=data_preview
                )
            except Exception as e:
                self.logger.error(f"{self.agent_id}: Exception during llm_client.generate_analysis_code: {e}", exc_info=True)
                last_error = f"Exception during LLM call: {str(e)}"
                attempt += 1
                if attempt < max_attempts:
                    self.logger.info(f"{self.agent_id}: Retrying code generation after exception (attempt {attempt + 1}/{max_attempts})...")
                    await asyncio.sleep(1) 
                continue

            if not isinstance(code_gen_result, CodeGenerationResult):
                last_error = f"Unexpected result type from code generation: {type(code_gen_result)}. Expected CodeGenerationResult."
                self.logger.error(f"{self.agent_id}: {last_error}")
                attempt += 1
                if attempt < max_attempts:
                    await asyncio.sleep(1)
                continue

            if code_gen_result.success and code_gen_result.code is not None:
                self.logger.info(f"{self.agent_id}: Code generation successful on attempt {attempt + 1}")
                # Validate the generated code
                if len(code_gen_result.code) > self.config["max_code_length"]:
                    error_msg = f"Generated code is too long ({len(code_gen_result.code)} characters > {self.config['max_code_length']})"
                    self.logger.error(f"{self.agent_id}: {error_msg}")
                    return CodeGenerationResult(
                        success=False,
                        code="", # Ensure code is always a string
                        error=error_msg,
                        explanation=code_gen_result.explanation,
                        model_name=code_gen_result.model_name,
                        usage_metadata=code_gen_result.usage_metadata,
                        raw_response=code_gen_result.raw_response
                    )
                    
                # Ensure the code has the required result variable for the executor
                if 'result = ' not in code_gen_result.code and 'return ' not in code_gen_result.code:
                    self.logger.info(f"{self.agent_id}: Adding default 'result = df' to generated code as no 'result =' or 'return ' found.")
                    code_gen_result.code += "\n\n# Ensure a 'result' variable is available for the executor\nresult = df"
                    
                return code_gen_result
            
            last_error = code_gen_result.error or 'No error message provided from LLM client'
            self.logger.warning(f"{self.agent_id}: Code generation failed on attempt {attempt + 1}/{max_attempts}: {last_error}")
                
            attempt += 1
            if attempt < max_attempts:
                self.logger.info(f"{self.agent_id}: Retrying code generation (attempt {attempt + 1}/{max_attempts})...")
                await asyncio.sleep(1) 
            
        final_error_msg = f"Failed to generate valid code after {max_attempts} attempts. Last error: {last_error}"
        self.logger.error(f"{self.agent_id}: {final_error_msg}")
        
        # Ensure a CodeGenerationResult is returned even if all attempts fail or code_gen_result was never successfully assigned
        raw_resp = getattr(code_gen_result, 'raw_response', None) if code_gen_result else None
        model_n = getattr(code_gen_result, 'model_name', None) if code_gen_result else None
        usage_m = getattr(code_gen_result, 'usage_metadata', None) if code_gen_result else None
        expl = getattr(code_gen_result, 'explanation', None) if code_gen_result else None

        return CodeGenerationResult(
            success=False,
            code="",
            error=final_error_msg,
            explanation=expl,
            model_name=model_n,
            usage_metadata=usage_m,
            raw_response=raw_resp
        )
            
        error_msg = f"Failed to generate valid code after {max_attempts} attempts: {last_error}"
        self.logger.error(error_msg)
        return CodeGenerationResult(
            success=False,
            error=error_msg
        )
    
    async def _explain_results(
        self,
        query: str,
        code: str,
        result: Any,
        data_preview: Dict[str, Any]
    ) -> str:
        """Generate a natural language explanation of the analysis results.
        
        Args:
            query: The original user query
            code: The generated analysis code
            result: The result of executing the code
            data_preview: Preview of the input data
            
        Returns:
            A natural language explanation of the results
        """
        self.logger.info("Generating explanation for analysis results...")
        
        # Convert the result to a string representation if it's not already
        if hasattr(result, 'to_string'):
            result_str = result.to_string()
        elif isinstance(result, (dict, list, pd.DataFrame, pd.Series, np.ndarray)):
            result_str = str(result)
        else:
            result_str = str(result)
        
        # Generate the explanation using the LLM
        explanation = await self.llm_client.explain_results(
            query=query,
            code=code,
            execution_result=result_str,
            data_preview=data_preview
        )
        
        return explanation or "No explanation available."
    
    async def _execute_analysis_code(
        self,
        code: str,
        df: pd.DataFrame,
        max_attempts: int = 3
    ) -> Dict[str, Any]:
        """Execute the generated analysis code in an isolated environment.
        
        Args:
            code: Python code to execute
            df: DataFrame to analyze
            max_attempts: Maximum number of execution attempts
            
        Returns:
            Dictionary containing the execution results with keys:
            - success: bool indicating if execution was successful
            - result: the output of the analysis (if successful)
            - error: error message (if execution failed)
        """
        # Create a temporary directory for the execution
        import tempfile
        import json
        from pathlib import Path
        
        temp_dir = Path(tempfile.mkdtemp(prefix="analysis_"))
        
        # Create a dictionary to store the data
        data_dict = df.to_dict(orient='records')
        
        # Create the execution script with the data embedded
        script = """
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("AI GENERATED CODE EXECUTION STARTED")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

import pandas as pd
import numpy as np
import json
import traceback


# Load the data from the embedded JSON
data = {data_dict}
df = pd.DataFrame(data)

# User's analysis code
{code}
    
# Ensure result is defined
if 'result' not in locals() and 'result' not in globals():
    result = df

# Convert result to a serializable format
if hasattr(result, 'to_dict'):
    if hasattr(result, 'index'):
        if hasattr(result, 'reset_index'):
            result = result.reset_index()
        if hasattr(result, 'to_dict'):
            output = result.to_dict(orient='records' if hasattr(result, 'columns') else None)
    else:
        output = result.to_dict()
elif isinstance(result, (np.ndarray, list, tuple)):
    output = result.tolist() if hasattr(result, 'tolist') else list(result)
elif isinstance(result, (str, int, float, bool)) or result is None:
    output = result
else:
    output = str(result)

# Ensure the output is JSON serializable
def make_serializable(obj):
    if isinstance(obj, (np.integer, int, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, float, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item') and callable(getattr(obj, 'item')):
        return obj.item()
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {{k: make_serializable(v) for k, v in obj.items()}}
    return str(obj)

# Apply serialization
if output is not None:
    if isinstance(output, (dict, list, tuple)):
        output = make_serializable(output)

# Save the output
with open('result.json', 'w') as f:
    json.dump(output, f, default=str)

print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('AI GENERATED CODE EXECUTION COMPLETED')
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    
""".format(data_dict=repr(data_dict), code=code)
        # Save the script
        script_path = temp_dir / "analysis_script.py"
        script_path.write_text(script)
        
        # Execute the script
        self.logger.info("Executing analysis script...")
        exec_result = self.executor.execute_code(
            code=script,
            source_dir=str(temp_dir)
        )
        
        if exec_result.get("status") == "executed" and not exec_result.get("errors"):
            # Load the result
            result_path = "result.json"
            if os.path.exists(result_path):
                with open(result_path, 'r') as f:
                    output = json.load(f)
                    
                # Delete the file at the result path if exists
                if os.path.exists(result_path):
                    os.remove(result_path)
                    print('RESULT FILE REMOVED')
                
                return {
                    "success": True,
                    "result": output,
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "result": None,
                    "error": "No result file was generated"
                }
        else:
            error_msg = exec_result.get("errors", "Unknown execution error")
            if not isinstance(error_msg, str):
                error_msg = str(error_msg)
            return {
                "success": False,
                "result": None,
                "error": f"Execution error: {error_msg}"
            }

    async def analyze_data(
        self,
        query: str,
        data_path: str,
        **kwargs
    ) -> AgentResponse:
        """
        Analyze data based on the user's query using AI-generated code.
        
        Args:
            query: The user's question or analysis request
            data_path: Path to the data file
            **kwargs: Additional arguments for data loading
            
        Returns:
            AgentResponse with the analysis results
        """
        self.logger.info(f"Analyzing data from {data_path} for query: {query}")
        
        self.logger.info("Loading data...")
        # Load the data
        df = await self._load_data(data_path, **kwargs)
        self.logger.info(f"Loaded data with shape: {df.shape}")
        
        # Create a preview of the data for code generation
        self.logger.info("Creating data preview...")
        data_preview = await self._get_data_preview(df)
        
        # Generate analysis code
        self.logger.info("Generating analysis code...")
        result = None # Initialize result
        try:
            self.logger.info(f"{self.agent_id}: Attempting to call and await self._generate_analysis_code")
            result = await self._generate_analysis_code(
                query=query,
                data_preview=data_preview
            )
            self.logger.info(f"{self.agent_id}: Call to self._generate_analysis_code completed. Result success: {result.success if result else 'N/A'}")
        except Exception as e_gen_code:
            self.logger.error(f"{self.agent_id}: Exception occurred DIRECTLY during 'await self._generate_analysis_code': {str(e_gen_code)}", exc_info=True)
            # Ensure a valid AgentResponse is returned or an error is raised appropriately
            # For now, let's re-raise to see if it gets caught by a higher-level handler or stops execution here
            # Depending on desired behavior, you might return an AgentResponse(success=False, error=...)
            raise CodeExecutionError(f"Direct exception from _generate_analysis_code call: {str(e_gen_code)}")
        
        if not result.success:
            error_msg = f"Failed to generate analysis code: {result.error}"
            self.logger.error(error_msg)
            raise CodeExecutionError(error_msg)
            
        code = result.code
        explanation = result.explanation or "No explanation provided"
        
        self.logger.info("Generated analysis code successfully")
        
        # Execute the generated code
        self.logger.info("Executing analysis code...")
        
        execution_result = await self._execute_analysis_code(code, df)
        self.logger.info(f"Code execution completed. Success: {execution_result['success']}")
        
        if not execution_result['success']:
            error_msg = f"Error executing analysis: {execution_result['error']}"
            self.logger.error(error_msg)
            raise CodeExecutionError(error_msg)
        
        # Get the explanation for the results
        explanation = ""
        self.logger.info("Generating explanation for results...")
        explanation = await self._explain_results(query, code, execution_result['result'], data_preview)
        self.logger.info("Explanation generated successfully")

        # Return the results as an AgentResponse
        return AgentResponse(
            success=True,
            output={
                "result": execution_result['result'],
                "code": code,
                "explanation": explanation,
                "data_preview": data_preview
            },
            error=None
        )
    
    async def profile_data(
        self,
        data_path: str,
        sample_size: Optional[int] = None,
        **kwargs
    ) -> AgentResponse:
        """
        Profile the data at the given path and return statistics and insights.
        
        Args:
            data_path: Path to the data file
            sample_size: Number of rows to sample for profiling (None for all)
            **kwargs: Additional arguments to pass to the data loader
            
        Returns:
            AgentResponse with the profiling results
        """
        self.logger.info(f"Profiling data from {data_path}")
        
        # Load the data
        df = await self._load_data(data_path, **kwargs)
        
        if sample_size is None:
            sample_size = self.config.get("profile_sample_size", 1000)
            
        if sample_size and len(df) > sample_size:
            df = df.sample(min(sample_size, len(df)))
        
        # Basic info
        profile = {
            "file_path": str(data_path),
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().mean() * 100).round(2).to_dict(),
        }
        
        # Numeric columns statistics
        numeric_cols = df.select_dtypes(include=['number']).columns
        if not numeric_cols.empty:
            profile["numeric_stats"] = df[numeric_cols].describe(
                percentiles=[.05, .25, .5, .75, .95]
            ).to_dict()
        
        # Categorical columns statistics
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if not cat_cols.empty:
            profile["categorical_stats"] = {}
            for col in cat_cols:
                value_counts = df[col].value_counts()
                profile["categorical_stats"][col] = {
                    "unique_values": value_counts.count(),
                    "top_values": value_counts.head(10).to_dict(),
                    "freq": (value_counts / len(df)).mul(100).round(2).head(10).to_dict()
                }
        
        # Memory usage
        profile["memory_usage"] = {
            "total_bytes": int(df.memory_usage(deep=True).sum()),
            "by_column": df.memory_usage(deep=True).to_dict(),
            "total_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        }
        
        # Sample data
        profile["sample_data"] = {
            "head": df.head(5).to_dict(orient='records'),
            "tail": df.tail(5).to_dict(orient='records')
        }
        
        return self._create_response(
            success=True,
            result={"profile": profile}
        )

    async def _initialize_sandbox_environment(self) -> None:
        """Initialize the sandbox environment with required libraries."""
        # This will be called during initialization to set up the sandbox

        # Create a basic environment setup script with proper indentation
        setup_script = """
                    # Import required libraries
                    import pandas as pd
                    import numpy as np
                    from typing import Dict, List, Any, Optional, Union
                    import json
                    import io
                    import sys
                    import traceback

                    # Store any helper functions or variables needed for analysis
                    class DataAnalysisHelper:
                        @staticmethod
                        def safe_eval(expr: str, local_vars: Dict[str, Any]) -> Any:
                            "Safely evaluate an expression with restricted globals."
                            allowed_globals = {
                                'pd': pd,
                                'np': np,
                                'json': json,
                                'Dict': Dict,
                                'List': List,
                                'Any': Any,
                                'Optional': Optional,
                                'Union': Union,
                                'sys': sys,
                                'traceback': traceback,
                                'io': io
                            }
                            try:
                                return eval(expr, {'__builtins__': {}}, {**allowed_globals, **local_vars})
                            except Exception as e:
                                return f"Error evaluating expression: {str(e)}"

                    # Make helper available
                    helper = DataAnalysisHelper()"""
        # Dedent the setup script to avoid indentation errors
        dedented_script = textwrap.dedent(setup_script)
        # Execute the setup script in the sandbox
        result = self.executor.execute_code(
            code=dedented_script,
            source_dir="."
        )
        if result["errors"]:
            self.logger.error(f"Failed to initialize sandbox environment: {result['errors']}")

    async def execute(
        self,
        task: str,
        **kwargs
    ) -> AgentResponse:
        """
        Execute a data analysis task.
            
        Returns:
            AgentResponse with the cleaned data
        """
        # Implementation for data cleaning
        data_path = kwargs.get("data_path")
        if not data_path:
            return AgentResponse(
                success=False,
                output=None,
                error="'data_path' parameter is required for data cleaning"
            )
        df = await self._load_data(data_path)
        # Basic cleaning: drop rows with any missing values and remove duplicates
        cleaned_df = df.dropna().drop_duplicates()
        # Return a preview of cleaned data and its shape
        preview = cleaned_df.head(10).to_dict(orient="records")
        return AgentResponse(
            success=True,
            output={
                "preview": preview,
                "shape": cleaned_df.shape
            },
            error=None
        )
    
    async def _analyze_data(self, **kwargs) -> AgentResponse:
        """
        Perform analysis on the data based on the provided query.
        
        Args:
            **kwargs: Should contain 'data' or 'data_path' and analysis parameters
            
        Returns:
            AgentResponse with the analysis results
        """
        # Implementation for data analysis
        return AgentResponse(
            success=False,
            output=None,
            error="Data analysis not yet implemented"
        )
    
    async def _transform_data(self, **kwargs) -> AgentResponse:
        """
        Transform the data based on the provided transformation rules.
        
        Args:
            **kwargs: Should contain 'data' or 'data_path' and transformation rules
            
        Returns:
            AgentResponse with the transformed data
        """
        # Implementation for data transformation
        return AgentResponse(
            success=False,
            output=None,
            error="Data transformation not yet implemented"
        )
    
    async def _custom_analysis(self, query: str, **kwargs) -> AgentResponse:
        """
        Perform a custom analysis based on a natural language query.
        
        Args:
            query: Natural language query describing the analysis to perform
            **kwargs: Additional parameters including 'data' or 'data_path'
            
        Returns:
            AgentResponse with the analysis results
        """
        # This would typically use an LLM to generate code for the analysis
        # For now, we'll return a placeholder response
        return AgentResponse(
            success=False,
            output=None,
            error="Custom analysis not yet implemented"
        )
    

