"""
Formula Agent - Handles spreadsheet-like formula evaluation.
"""
import logging
from typing import Dict, Any, Optional, Union, List

from .base_agent import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)

class FormulaAgent(BaseAgent):
    """
    Agent responsible for evaluating spreadsheet-like formulas on data.
    """
    
    def __init__(self, 
                 agent_id: str,
                 name: str = "Formula Agent",
                 description: str = "Evaluates spreadsheet-like formulas on data",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the FormulaAgent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name
            description: Description of the agent's purpose
            config: Configuration dictionary
        """
        default_config = {
            "max_formula_length": 1000,
            "allowed_functions": [
                "sum", "avg", "mean", "min", "max", "count", "if", "and", "or", 
                "not", "concat", "left", "right", "mid", "len", "find", "substitute",
                "trim", "lower", "upper", "proper", "round", "roundup", "rounddown"
            ],
            "max_iterations": 1000,
            "timeout_seconds": 30
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(agent_id, name, description, default_config)
    
    async def initialize(self):
        """Initialize the agent and any required resources."""
        logger.info(f"Initializing {self.name} (ID: {self.agent_id})")
        self.state = "ready"
    
    async def shutdown(self):
        """Clean up resources used by the agent."""
        logger.info(f"Shutting down {self.name} (ID: {self.agent_id})")
        self.state = "shutdown"
    
    async def evaluate_formula(
        self, 
        formula: str, 
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Evaluate a formula against the provided data.
        
        Args:
            formula: The formula to evaluate (e.g., "=SUM(A1:A10)")
            data: The data to evaluate the formula against
            context: Additional context for formula evaluation
            
        Returns:
            AgentResponse containing the result of the formula evaluation
        """
        # Remove leading '=' if present
        if formula.startswith('='):
            formula = formula[1:]
        
        # Validate formula length
        if len(formula) > self.config["max_formula_length"]:
            return self._create_response(
                success=False,
                error=f"Formula exceeds maximum length of {self.config['max_formula_length']} characters"
            )
        
        # Prepare the evaluation context
        eval_globals = self._get_safe_globals()
        eval_locals = {"data": data}
        
        if context:
            eval_locals.update(context)
        
        # Evaluate the formula in a restricted environment
        result = eval(formula, eval_globals, eval_locals)
        
        return self._create_response(
            success=True,
            result=result
        )
    
    def _get_safe_globals(self) -> Dict[str, Any]:
        """
        Get a dictionary of safe globals for formula evaluation.
        
        Returns:
            Dictionary of safe globals
        """
        import math
        import statistics
        import re
        
        # Basic math functions
        safe_globals = {
            # Math functions
            'abs': abs,
            'round': round,
            'sum': sum,
            'min': min,
            'max': max,
            'len': len,
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
            # Math module
            'math': {
                'pi': math.pi,
                'e': math.e,
                'sqrt': math.sqrt,
                'pow': math.pow,
                'exp': math.exp,
                'log': math.log,
                'log10': math.log10,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'degrees': math.degrees,
                'radians': math.radians,
            },
            # Statistics
            'stats': {
                'mean': statistics.mean,
                'median': statistics.median,
                'mode': statistics.mode,
                'stdev': statistics.stdev,
                'variance': statistics.variance,
            },
            # String operations
            're': re,
            # List operations
            'list': list,
            'dict': dict,
            'range': range,
        }
        
        # Add allowed functions from config
        for func_name in self.config["allowed_functions"]:
            if func_name in safe_globals:
                continue
                
            if hasattr(math, func_name):
                safe_globals[func_name] = getattr(math, func_name)
            elif hasattr(statistics, func_name):
                safe_globals[func_name] = getattr(statistics, func_name)
        
        return safe_globals
    
    async def batch_evaluate(
        self, 
        formulas: Dict[str, str], 
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Evaluate multiple formulas in a batch.
        
        Args:
            formulas: Dictionary of formula names to formulas
            data: The data to evaluate the formulas against
            context: Additional context for formula evaluation
            
        Returns:
            AgentResponse containing the results of all formula evaluations
        """
        results = {}
        errors = {}
        
        for name, formula in formulas.items():
            result = await self.evaluate_formula(formula, data, context)
            if result["success"]:
                results[name] = result["result"]
            else:
                errors[name] = result["error"]
        
        return self._create_response(
            success=len(errors) == 0,
            result=results,
            error=errors if errors else None
        )
    
    async def validate_formula(self, formula: str) -> AgentResponse:
        """
        Validate that a formula is syntactically correct.
        
        Args:
            formula: The formula to validate
            
        Returns:
            AgentResponse indicating if the formula is valid
        """
        # Try to compile the formula
        if formula.startswith('='):
            formula = formula[1:]
        compile(formula, "<string>", "eval")
        return self._create_response(success=True)