import pandas as pd
import numpy as np
import json
import logging
import sys
import traceback
import asyncio
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import textwrap

from .base_agent import BaseAgent, AgentResponse
from ..tools.isolated_python_executor import IsolatedPythonExecutor

logger = logging.getLogger(__name__)

class DataAnalysisAgent(BaseAgent):
    """
    Specialized agent for performing data analysis tasks on tabular data.
    Handles tasks like data profiling, statistical analysis, and data cleaning.
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
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(agent_id, name, description, default_config)
        
        # Initialize the isolated Python executor
        self.executor = IsolatedPythonExecutor("data_analysis_sandbox")
        
        # We'll initialize the environment later when needed
        # DO NOT use asyncio.create_task here as it requires a running event loop
        self.initialized = False
    
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
        Load data from a file into a pandas DataFrame.
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments to pass to the pandas read function
            
        Returns:
            Loaded DataFrame
            
        Raises:
            ValueError: If the file type is not supported or the file is too large
            Exception: For other loading errors
        """
        file_path = Path(file_path)
        
        # Check file extension
        file_ext = file_path.suffix.lower()[1:]  # Remove the dot
        if file_ext not in self.config["allowed_file_types"]:
            raise ValueError(
                f"Unsupported file type: {file_ext}. "
                f"Supported types: {', '.join(self.config['allowed_file_types'])}"
            )
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config["max_file_size_mb"]:
            raise ValueError(
                f"File size ({file_size_mb:.2f} MB) exceeds the maximum allowed size "
                f"({self.config['max_file_size_mb']} MB)"
            )
        
        # Read the file based on its type
        try:
            if file_ext == 'csv':
                return pd.read_csv(file_path, **kwargs)
            elif file_ext in ('xlsx', 'xls'):
                return pd.read_excel(file_path, **kwargs)
            elif file_ext == 'parquet':
                return pd.read_parquet(file_path, **kwargs)
            else:
                raise ValueError(f"Unhandled file type: {file_ext}")
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {str(e)}")
            raise
    
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
        
        try:
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
            
        except Exception as e:
            self.logger.exception(f"Error profiling data from {data_path}")
            return self._create_response(
                success=False,
                error=f"Error profiling data: {str(e)}",
                traceback=self._get_traceback()
            )
    
    async def _initialize_sandbox_environment(self) -> None:
        """Initialize the sandbox environment with required libraries."""
        # This will be called during initialization to set up the sandbox
        try:
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
        except Exception as e:
            self.logger.error(f"Error initializing sandbox environment: {str(e)}")

        
    async def execute(
        self,
        task: str,
        **kwargs
    ) -> AgentResponse:
        """
        Execute a data analysis task.
    if query and not task:
        task = "custom analysis"
    elif not task and not query:
        return AgentResponse(
            success=False,
            output=None,
            error="Either 'query' or 'task' parameter is required"
        )
    
    self.logger.info(f"Executing data analysis task: {task}")
    
    try:
        # Make sure the sandbox environment is initialized
        if not self.initialized:
            await self._initialize_sandbox_environment()
            self.initialized = True
        
        # Route to the appropriate handler based on the task
        task = task.lower().strip()
        
        if task == "profile data":
            return await self.profile_data(**kwargs)
        elif task == "clean data":
            return await self._clean_data(**kwargs)
        elif task == "analyze data":
            return await self._analyze_data(**kwargs)
        elif task == "transform data":
            return await self._transform_data(**kwargs)
        elif task == "custom analysis":
            query = kwargs.get("query")
            if not query:
                return AgentResponse(
                    success=False,
                    output=None,
                    error="Custom analysis requires a 'query' parameter"
                )
            return await self._custom_analysis(query, **kwargs)
        else:
            return AgentResponse(
                success=False,
                output=None,
                error=f"Unknown task: {task}"
            )
    
    except Exception as e:
        self.logger.exception(f"Error executing task: {task}")
        return AgentResponse(
            success=False,
            output=None,
            error=f"Error executing task: {str(e)}",
            metadata={"traceback": self._get_traceback()}
        )
            
        Returns:
            AgentResponse with the cleaned data
        """
        # Implementation for data cleaning
        try:
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
        except Exception as e:
            self.logger.exception("Error during data cleaning")
            return AgentResponse(
                success=False,
                output=None,
                error=f"Data cleaning failed: {str(e)}"
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
    
    async def _generate_analysis_code(self, query: str, **kwargs) -> str:
        """
        Generate Python code to perform the requested analysis.
        
        Args:
            query: Natural language query
            **kwargs: Additional parameters
            
        Returns:
            Generated Python code as a string
        """
        # In a real implementation, this would use an LLM to generate the code
        # For now, we'll return a simple template
        return """
        # This is a placeholder for generated analysis code
        # In a real implementation, this would be generated based on the query
        
        import pandas as pd
        import numpy as np
        
        # Load data
        # data = pd.read_csv('data.csv')
        
        # Perform analysis
        # result = data.describe()
        
        # Save results
        # result.to_csv('analysis_result.csv')
        print("Analysis completed successfully")
        """
