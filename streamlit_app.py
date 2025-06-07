import streamlit as st
import os
import tempfile
import asyncio
from pathlib import Path
import sys
import logging
import pandas as pd
from dotenv import load_dotenv

# Determine the directory of this script
SCRIPT_DIR = Path(__file__).resolve().parent

# Add the script's directory (SheetMinds) to sys.path
# This allows imports like `from src.agents...`
sys.path.append(str(SCRIPT_DIR))

# Attempt to import agent classes
try:
    from src.agents.controller_agent import ControllerAgent, AgentResponse
except ImportError as e:
    st.error(f"Error importing agent classes: {e}. Ensure the 'src' directory is correctly placed relative to this script and sys.path is set up correctly. Current SCRIPT_DIR: {SCRIPT_DIR}")
    st.stop()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler('streamlit_app.log') # Optional: log to file
    ]
)
logger = logging.getLogger(__name__)

@st.cache_resource # Cache the controller agent instance
def get_controller_agent_instance():
    logger.info("Attempting to initialize ControllerAgent for Streamlit app...")
    
    # Load .env file from the SCRIPT_DIR (SheetMinds directory)
    dotenv_path = SCRIPT_DIR / '.env'
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        logger.info(f"Loaded .env file from: {dotenv_path}")
        # Verify if GOOGLE_API_KEY is loaded (optional check)
        # if not os.getenv("GOOGLE_API_KEY"):
        #     logger.warning("GOOGLE_API_KEY not found in environment after loading .env")
        # else:
        #     logger.info("GOOGLE_API_KEY found in environment.")
    else:
        logger.warning(f".env file not found at {dotenv_path}. Gemini API key might not be available. Make sure it's in the same directory as streamlit_app.py.")

    try:
        controller = ControllerAgent(
            agent_id="streamlit_controller_001",
            name="Streamlit SheetMind Controller",
            description="Manages data analysis workflows for Streamlit UI"
        )
        logger.info("ControllerAgent initialized successfully.")
        return controller
    except Exception as e:
        logger.error(f"Failed to initialize ControllerAgent: {e}", exc_info=True)
        st.error(f"Critical Error: Could not initialize the AI Controller Agent: {e}")
        st.stop()

async def run_analysis_async(query: str, file_path: str) -> AgentResponse:
    controller = get_controller_agent_instance()
    if not controller:
        return AgentResponse(success=False, error="Controller Agent not available.", output=None)
        
    logger.info(f"Running analysis for query: '{query}' on file: '{file_path}'")
    try:
        response = await controller.execute(
            task="analyze",
            query=query,
            data_path=file_path
        )
        return response
    except Exception as e:
        logger.error(f"Error during analysis execution: {e}", exc_info=True)
        return AgentResponse(success=False, error=str(e), output=None)

def display_results(analysis_response: AgentResponse):
    if analysis_response.success:
        st.success("Analysis complete! 🎉")
        output = analysis_response.output
        
        if output:
            if output.get("result"):
                st.subheader("💡 Analysis Result:")
                st.write(output["result"])
            
            if output.get("explanation"):
                st.subheader("📄 Explanation:")
                st.markdown(output["explanation"])
            
            if output.get("code"):
                st.subheader("🐍 Generated Code:")
                st.code(output["code"], language="python")
            
            if output.get("data_preview"):
                st.subheader("📊 Data Preview (Result):")
                preview_data = output["data_preview"]
                if isinstance(preview_data, dict) and "columns" in preview_data and "data" in preview_data:
                    try:
                        df_preview = pd.DataFrame(preview_data["data"], columns=preview_data["columns"])
                        st.dataframe(df_preview)
                    except Exception as e:
                        st.error(f"Could not display data preview as DataFrame: {e}")
                        st.json(preview_data) # Show raw JSON if DataFrame conversion fails
                elif isinstance(preview_data, (list, pd.DataFrame)):
                    try:
                        st.dataframe(pd.DataFrame(preview_data))
                    except Exception as e:
                        st.error(f"Could not display data preview: {e}")
                        st.json(preview_data)
                elif isinstance(preview_data, str):
                     st.text(preview_data)
                else:
                    st.json(preview_data) # Show raw JSON if not in expected format
            else:
                st.info("No data preview available for this result.")
        else:
            st.warning("Analysis succeeded but returned no output.")
            
    else:
        st.error(f"Analysis Failed: {analysis_response.error}")
        # Attempt to access traceback if it's nested in output as per original Flask error handling
        tb = None
        if analysis_response.output and isinstance(analysis_response.output, dict):
            tb = analysis_response.output.get('traceback')
        elif isinstance(analysis_response.error, dict):
            tb = analysis_response.error.get('traceback') # if error itself is a dict
        
        if tb:
            st.subheader("Traceback:")
            st.code(tb)

def main():
    st.set_page_config(page_title="SheetMind AI Data Analysis", layout="wide")
    st.title("SheetMind: AI-Powered Data Analysis 🧠📊")
    st.markdown("Upload your data file (CSV, Excel, Parquet) and ask a question in natural language.")

    uploaded_file = st.file_uploader(
        "Choose a data file",
        type=["csv", "xlsx", "xls", "parquet"],
        help="Supported formats: CSV, Excel (xlsx, xls), Parquet"
    )

    query = st.text_area(
        "Enter your analysis query:",
        placeholder="e.g., What is the average sales per region? or Show me the top 5 products by profit.",
        height=100
    )

    if st.button("Analyze Data 🚀", type="primary"):
        if uploaded_file is not None and query.strip():
            with st.spinner("Analyzing your data... This might take a moment. 🤖"):
                temp_file_path = ""
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_file_path = tmp_file.name
                    
                    logger.info(f"File '{uploaded_file.name}' saved to temporary path: '{temp_file_path}'")
                    
                    # Run analysis: Streamlit button callbacks are sync, use asyncio.run
                    analysis_response = asyncio.run(run_analysis_async(query, temp_file_path))
                    display_results(analysis_response)

                except Exception as e:
                    logger.error(f"An unexpected error occurred in Streamlit app: {e}", exc_info=True)
                    st.error(f"An unexpected error occurred: {e}")
                finally:
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                        logger.info(f"Temporary file '{temp_file_path}' removed.")
        
        elif not uploaded_file:
            st.warning("Please upload a data file. 📁")
        elif not query.strip():
            st.warning("Please enter an analysis query. ❓")

if __name__ == "__main__":
    # Ensure ControllerAgent can be initialized before running the main app logic
    # This also ensures .env is loaded early if needed by GeminiClient during ControllerAgent init
    # get_controller_agent_instance() 
    # No, call it inside main or where needed to benefit from Streamlit's caching and error display.
    main()
