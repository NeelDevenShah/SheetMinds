#!/usr/bin/env python3
"""
SheetMind - Agentic AI System for CSV/Excel Data Analysis

This is the main entry point for the SheetMind system.
"""
import asyncio
import argparse
import logging
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.absolute()))

from src.agents.controller_agent import ControllerAgent
from src.agents.data_analysis_agent import DataAnalysisAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sheetmind.log')
    ]
)
logger = logging.getLogger(__name__)


class SheetMind:
    """Main class for the SheetMind system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the SheetMind system.
        
        Args:
            config_path: Path to a configuration file (optional)
        """
        self.config = self._load_config(config_path)
        self.controller: Optional[ControllerAgent] = None
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dictionary containing configuration settings
        """
        default_config = {
            "controller": {
                "agent_id": "controller_001",
                "name": "SheetMind Controller",
                "description": "Manages and coordinates all agents in the SheetMind system",
                "max_parallel_tasks": 5,
                "task_timeout": 300,
                "state_file": "sheetmind_state.json"
            },
            "data_analysis": {
                "max_rows_to_display": 10,
                "max_columns_to_display": 10,
                "profile_sample_size": 1000,
                "allowed_file_types": ["csv", "xlsx", "xls", "parquet"],
                "max_file_size_mb": 50,
                "default_encoding": "utf-8"
            },
            "logging": {
                "level": "INFO",
                "file": "sheetmind.log"
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    # Merge with default config
                    self._merge_dicts(default_config, user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {str(e)}")
        
        return default_config
    
    def _merge_dicts(self, base: Dict[Any, Any], update: Dict[Any, Any]) -> None:
        """
        Recursively merge two dictionaries.
        
        Args:
            base: The base dictionary to update
            update: Dictionary with updates to apply
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_dicts(base[key], value)
            else:
                base[key] = value
    
    async def initialize(self) -> None:
        """Initialize the SheetMind system and all components."""
        logger.info("Initializing SheetMind system...")
        
        try:
            # Initialize the controller agent
            self.controller = ControllerAgent(
                agent_id=self.config["controller"]["agent_id"],
                name=self.config["controller"]["name"],
                description=self.config["controller"]["description"],
                config={
                    "max_parallel_tasks": self.config["controller"]["max_parallel_tasks"],
                    "task_timeout": self.config["controller"]["task_timeout"],
                    "state_file": self.config["controller"]["state_file"]
                }
            )
            
            logger.info("SheetMind system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize SheetMind system: {str(e)}", exc_info=True)
            raise
    
    async def process_file(self, file_path: str, task: str = "profile data") -> Dict[str, Any]:
        """
        Process a file with the specified task.
        
        Args:
            file_path: Path to the file to process
            task: The task to perform (e.g., "profile data", "analyze data")
            
        Returns:
            Dictionary with the processing results
        """
        if not self.controller:
            raise RuntimeError("SheetMind system not initialized. Call initialize() first.")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Processing file: {file_path} with task: {task}")
        
        try:
            # Create a task for the data analysis
            task_id = self.controller.create_task(
                description=f"{task} for {os.path.basename(file_path)}",
                agent_type="data_analysis",
                params={
                    "task": task,
                    "data_path": file_path
                }
            )
            
            # Execute the task
            results = await self.controller.execute_tasks()
            
            if task_id in results:
                result = results[task_id]
                if result.success:
                    logger.info(f"Task completed successfully: {task_id}")
                    return {"success": True, "result": result.output}
                else:
                    logger.error(f"Task failed: {task_id} - {result.error}")
                    return {"success": False, "error": result.error}
            else:
                error_msg = f"Task {task_id} not found in results"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            error_msg = f"Error processing file {file_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"success": False, "error": error_msg}
    
    async def shutdown(self) -> None:
        """Shut down the SheetMind system and all components."""
        logger.info("Shutting down SheetMind system...")
        
        if self.controller:
            await self.controller.shutdown()
        
        logger.info("SheetMind system shutdown complete")


async def main():
    """Main entry point for the SheetMind CLI."""
    parser = argparse.ArgumentParser(description="SheetMind - Agentic AI System for CSV/Excel Data Analysis")
    parser.add_argument("file", nargs="?", help="Path to the file to analyze")
    parser.add_argument("--task", default="profile data", help="Task to perform (default: profile data)")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output", help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    if not args.file:
        parser.print_help()
        return
    
    sheetmind = SheetMind(config_path=args.config)
    
    try:
        # Initialize the system
        await sheetmind.initialize()
        
        # Process the file
        result = await sheetmind.process_file(args.file, args.task)
        
        # Output the results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
        
    finally:
        # Clean up
        await sheetmind.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
