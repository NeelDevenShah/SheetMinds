# Simple controller_agent.py - Direct CSV analysis without complex task management
from typing import Dict, Any, Optional
import logging
from .base_agent import BaseAgent, AgentResponse
from .data_analysis_agent import DataAnalysisAgent

logger = logging.getLogger(__name__)


class ControllerAgent(BaseAgent):
    """
    A super simple controller that directly calls the DataAnalysisAgent.
    No task management, IDs, or complex structures - just direct execution.
    """
    
    def __init__(self, 
                 agent_id: str = "controller_001",
                 name: str = "SheetMind Controller",
                 description: str = "Simple controller for CSV analysis",
                 config: Optional[Dict[str, Any]] = None):
        """Initialize the ControllerAgent."""
        default_config = {}
        if config:
            default_config.update(config)
        
        super().__init__(agent_id, name, description, default_config)
        
        # We'll initialize the data analysis agent later when needed
        self._data_analysis_agent = None
    
    async def execute(self, query: str, **kwargs) -> AgentResponse:
        """
        Execute a query by directly passing it to the data analysis agent.
        
        Args:
            query: The question or analysis request about the CSV data
            **kwargs: Additional parameters including data_path
            
        Returns:
            AgentResponse containing the result
        """
        self.logger.info(f"Executing query: {query}")
        
        try:
            # Initialize the data analysis agent if not already done
            if self._data_analysis_agent is None:
                self.logger.info("Initializing DataAnalysisAgent")
                self._data_analysis_agent = DataAnalysisAgent(
                    agent_id="data_analysis_001",
                    name="Data Analysis Agent",
                    description="Agent for analyzing CSV data"
                )
                # Manually initialize the sandbox environment
                await self._initialize_data_analysis_agent()
            
            # Call the data analysis agent
            task = kwargs.pop("task", "custom analysis")
            result = await self._data_analysis_agent.execute(task=task, query=query, **kwargs)
            self.logger.info("Query executed successfully")
            return result
            
        except Exception as e:
            error_msg = f"Error executing query: {str(e)}"
            self.logger.error(error_msg)
            return AgentResponse(
                success=False, 
                output=None,
                error=error_msg
            )
            
    async def _initialize_data_analysis_agent(self):
        """
        Safely initialize the data analysis agent's sandbox environment.
        """
        try:
            # If the agent has an _initialize_sandbox_environment method, call it
            if hasattr(self._data_analysis_agent, '_initialize_sandbox_environment'):
                await self._data_analysis_agent._initialize_sandbox_environment()
        except Exception as e:
            self.logger.error(f"Error initializing data analysis agent: {str(e)}")
            # We'll still continue, as some functionality might work

