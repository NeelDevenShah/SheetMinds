import os
import json
import asyncio
from typing import Dict, Any, Optional, Union
import traceback
import logging # Added
import time # Added
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

load_dotenv()

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Added

@dataclass
class CodeGenerationResult:
    """Container for code generation results."""
    success: bool
    code: Optional[str] = None
    explanation: Optional[str] = None
    error: Optional[str] = None


class GeminiClient:
    """Client for interacting with Google's Gemini API."""
    
    def __init__(self, api_key: str = None):
        """Initialize the Gemini client.
        
        Args:
            api_key: Optional API key. If not provided, will look for GEMINI_API_KEY env var.
        """
        _env_api_key = os.getenv("GEMINI_API_KEY")
        logger.info(f"GeminiClient __init__: GEMINI_API_KEY from env: {'******' if _env_api_key else 'Not found'}")
        self.api_key = api_key or _env_api_key
        if not self.api_key:
            logger.error("GeminiClient __init__: Gemini API key is missing. Ensure GEMINI_API_KEY is in .env or passed directly.")
            raise ValueError("Gemini API key not set. Please set GEMINI_API_KEY environment variable or pass it to the client.")
        else:
            logger.info("GeminiClient __init__: API key loaded successfully.")

    async def _get_llm(self):
        """Get an instance of the LLM client."""
        logger.info("GeminiClient _get_llm: Attempting to import and instantiate ChatGoogleGenerativeAI.")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm_instance = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash", 
                api_key=self.api_key,
                temperature=0.2  # Lower temperature for more deterministic code generation
            )
            logger.info("GeminiClient _get_llm: ChatGoogleGenerativeAI instantiated successfully.")
            return llm_instance
        except ImportError as e:
            logger.error(f"GeminiClient _get_llm: Failed to import langchain_google_genai. Ensure it's installed. Error: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"GeminiClient _get_llm: Failed to instantiate ChatGoogleGenerativeAI. Error: {e}", exc_info=True)
            raise

    async def generate_content(
        self, 
        prompt: str, 
        data: Optional[Dict] = None, 
        system_instruction: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate content using Gemini.
        
        Args:
            prompt: The prompt/question to send to Gemini
            data: Optional data to include as context
            system_instruction: Optional system instruction
            **kwargs: Additional arguments
            
        Returns:
            The generated content as a string
        """
        llm = await self._get_llm()
        
        # Format the prompt with data if provided
        if data:
            data_str = json.dumps(data, indent=2)
            full_prompt = f"""Given the following data context:
```json
{data_str}
```

{prompt}

Please provide a clear and concise response."""
        else:
            full_prompt = prompt
            
        if system_instruction:
            full_prompt = f"{system_instruction}\n\n{full_prompt}"
            
        logger.info(f"GeminiClient: Sending prompt to LLM (first 100 chars): {full_prompt[:100]}...")
        start_time = time.time()
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: llm.invoke(full_prompt)
            )
            end_time = time.time()
            logger.info(f"GeminiClient: LLM call completed in {end_time - start_time:.2f} seconds.")
        except Exception as e:
            end_time = time.time()
            logger.error(f"GeminiClient: LLM call failed after {end_time - start_time:.2f} seconds: {e}", exc_info=True)
            raise # Re-raise the exception to be handled by the caller
        return response.content.strip()

    async def generate_analysis_code(
        self,
        query: str,
        data_preview: Dict[str, Any],
        **kwargs
    ) -> CodeGenerationResult:
        """Generate Python code for data analysis.
        
        Args:
            query: The user's query about the data
            data_preview: A preview of the data (first few rows, column names, dtypes)
            **kwargs: Additional arguments
            
        Returns:
            CodeGenerationResult containing the generated code and metadata
        """
        system_prompt = """You are an expert data analyst. Generate Python code to analyze the given data 
        based on the user's query. Follow these rules:
        
        1. Only use pandas and numpy for data manipulation
        2. Include all necessary imports
        3. The data is already loaded in a variable called 'df'
        4. The code should return the result in a variable called 'result'
        5. Add comments to explain the analysis steps
        6. Handle potential errors gracefully
        7. The code should be a complete, runnable script
        8. The last line should be: result = <your_result_variable>
        
        Format your response as a markdown code block with language 'python'.
        
        Make sure your code is correct and functional. Test it before submitting.
        Pay attention to indentation (use 4 spaces for each level of indentation).
        Do not include any blank lines at the end of your code.
        """
        
        data_preview_str = json.dumps(data_preview, indent=2)
        
        prompt = f"""# Data Analysis Task
        
## User Query:
{query}

## Data Preview:
```json
{data_preview_str}
```

Please generate Python code to analyze this data according to the query. The code should be complete and ready to execute."""

        response = await self.generate_content(
            prompt=prompt,
            system_instruction=system_prompt
        )
        
        # Extract code from markdown code block if present
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0].strip()
        elif "```" in response:
            code = response.split("```")[1].strip()
        else:
            code = response.strip()
        
        return CodeGenerationResult(
            success=True,
            code=code,
            explanation=response
        )

    async def explain_results(
        self,
        query: str,
        code: str,
        execution_result: Any,
        **kwargs
    ) -> str:
        """Generate a natural language explanation of the analysis results.
        
        Args:
            query: The original user query
            code: The code that was executed
            execution_result: The result of executing the code
            **kwargs: Additional arguments
            
        Returns:
            Natural language explanation of the results
        """
        system_prompt = """You are a helpful data analysis assistant. Explain the results of the data analysis 
        in clear, non-technical language. Focus on answering the user's original question and highlighting 
        key insights from the data. Be concise but thorough."""
        
        result_str = str(execution_result)
        if len(result_str) > 2000:  # Truncate very long results
            result_str = result_str[:2000] + "... [truncated]"
            
        prompt = f"""# Data Analysis Results Explanation
        
## User's Original Question:
{query}

## Analysis Code:
```python
{code}
```

## Execution Result:
```
{result_str}
```

Please explain these results in a clear, non-technical way that directly answers the user's question."""

        return await self.generate_content(
            prompt=prompt,
            system_instruction=system_prompt
        )
