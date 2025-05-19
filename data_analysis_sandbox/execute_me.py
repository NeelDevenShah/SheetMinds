
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
helper = DataAnalysisHelper()