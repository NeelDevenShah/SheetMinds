"""
Controller Agent for managing data analysis workflows.

This agent coordinates between the API and the data analysis agent,
handling task management, error handling, and result formatting.
"""

import logging
from typing import Dict, Any, Optional, Union, List
import json

from .base_agent import BaseAgent, AgentResponse
from .data_analysis_agent import DataAnalysisAgent, DataAnalysisError, CodeExecutionError

logger = logging.getLogger(__name__)


class ControllerAgent(BaseAgent):
    """
    Controller agent that manages data analysis workflows.
    
    This agent serves as the main entry point for analysis requests,
    coordinating between the API and specialized analysis agents.
    """
    
    def __init__(self, 
                 agent_id: str = "controller_001",
                 name: str = "SheetMind Controller",
                 description: str = "Manages data analysis workflows and coordinates agents",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ControllerAgent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name
            description: Description of the agent's purpose
            config: Configuration dictionary
        """
        default_config = {
            "max_query_length": 1000,
            "allowed_file_types": ["csv", "xlsx", "xls", "parquet"],
            "max_file_size_mb": 50,
            "enable_caching": True,
            "cache_ttl_seconds": 3600,  # 1 hour
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(agent_id, name, description, default_config)
        
        # Initialize the data analysis agent
        self._data_analysis_agent = DataAnalysisAgent(
            agent_id="data_analysis_001",
            name="Data Analysis Agent",
            description="Agent for analyzing tabular data with AI-generated code",
            config={
                "max_code_length": 5000,
                "max_code_execution_attempts": 3,
            }
        )
        
        # Cache for storing analysis results
        self._result_cache = {}
        self.initialized = True
    
    async def execute(self, task: str, **kwargs) -> AgentResponse:
        """
        Execute a task with the given parameters.
        
        Args:
            task: The task to perform (e.g., 'analyze', 'profile')
            **kwargs: Task-specific parameters
            
        Returns:
            AgentResponse containing the result or error
        """
        self.logger.info(f"Executing task: {task}")
        
        # Route to the appropriate handler based on the task
        if task == "analyze":
            # Remove the explicitly passed args from kwargs to avoid duplication
            analysis_kwargs = kwargs.copy()
            if 'query' in analysis_kwargs:
                del analysis_kwargs['query']
            if 'data_path' in analysis_kwargs:
                del analysis_kwargs['data_path']
                
            return await self._handle_analysis(
                query=kwargs.get("query"),
                data_path=kwargs.get("data_path"),
                **analysis_kwargs
            )
        elif task == "profile":
            # Remove the explicitly passed args from kwargs to avoid duplication
            profile_kwargs = kwargs.copy()
            if 'data_path' in profile_kwargs:
                del profile_kwargs['data_path']
                
            return await self._handle_profiling(
                data_path=kwargs.get("data_path"),
                **profile_kwargs
            )
        else:
            raise ValueError(f"Unknown task: {task}")
    
    async def _handle_analysis(
        self,
        query: str,
        data_path: str,
        **kwargs
    ) -> AgentResponse:
        """
        Handle a data analysis request.
        
        Args:
            query: The user's question or analysis request
            data_path: Path to the data file
            **kwargs: Additional parameters
            
        Returns:
            AgentResponse with the analysis results
        """
        # Validate inputs
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
            
        if len(query) > self.config["max_query_length"]:
            raise ValueError(
                f"Query is too long. Maximum length is {self.config['max_query_length']} characters"
            )
            
        if not data_path or not isinstance(data_path, str):
            raise ValueError("data_path must be a non-empty string")
        
        # Check cache if enabled
        cache_key = self._generate_cache_key("analysis", query, data_path)
        if self.config["enable_caching"] and cache_key in self._result_cache:
            self.logger.info("Returning cached analysis result")
            return self._result_cache[cache_key]
        
        # Execute the analysis
        result = await self._data_analysis_agent.analyze_data(
            query=query,
            data_path=data_path,
            **kwargs
        )
        
        # Cache the result if successful
        if result.success and self.config["enable_caching"]:
            self._result_cache[cache_key] = result
            
        return result
    
    async def _handle_profiling(
        self,
        data_path: str,
        **kwargs
    ) -> AgentResponse:
        """
        Handle a data profiling request.
        
        Args:
            data_path: Path to the data file
            **kwargs: Additional parameters
            
        Returns:
            AgentResponse with the profiling results
        """
        # Validate inputs
        if not data_path or not isinstance(data_path, str):
            raise ValueError("data_path must be a non-empty string")
        
        # Check cache if enabled
        cache_key = self._generate_cache_key("profile", data_path)
        if self.config["enable_caching"] and cache_key in self._result_cache:
            self.logger.info("Returning cached profile result")
            return self._result_cache[cache_key]
        
        # Execute the profiling
        result = await self._data_analysis_agent.profile_data(
            data_path=data_path,
            **kwargs
        )
        
        # Cache the result if successful
        if result.success and self.config["enable_caching"]:
            self._result_cache[cache_key] = result
            
        return result
    
    def _generate_cache_key(self, prefix: str, *args) -> str:
        """Generate a cache key from the given arguments.
        
        Args:
            prefix: Cache key prefix
            *args: Values to include in the key
            
        Returns:
            A string cache key
        """
        return f"{prefix}:{':'.join(str(arg) for arg in args)}"
    
    def clear_cache(self) -> None:
        """Clear the result cache."""
        self._result_cache.clear()
        self.logger.info("Cache cleared")

