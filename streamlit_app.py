import streamlit as st
import pandas as pd
import os
import tempfile
import uuid
import asyncio # Required if running async methods outside Streamlit's direct support

# Make sure SheetMinds is in PYTHONPATH or run from Csv-QA root
# e.g., export PYTHONPATH=$PYTHONPATH:$(pwd)
# from SheetMinds.src.agents.controller_agent import ControllerAgent # This will be cached
# from SheetMinds.src.agents.base_agent import AgentResponse

# Placeholder for actual imports - will be dynamically checked if possible or assumed
# For local development, ensure these paths are correct and SheetMinds is in PYTHONPATH
try:
    from SheetMinds.src.agents.controller_agent import ControllerAgent
    from SheetMinds.src.agents.base_agent import AgentResponse
    # Ensure GeminiClient can be initialized; it might need GOOGLE_API_KEY environment variable
except ImportError as e:
    st.error(f"Failed to import required agent modules: {e}. "
             f"Please ensure the 'SheetMinds' directory is in your PYTHONPATH "
             f"and you are running Streamlit from the project root (e.g., Csv-QA). "
             f"Also, ensure all dependencies like 'google-generativeai' are installed.")
    st.stop()

st.set_page_config(layout="wide", page_title="SheetMind Data Analyzer")

st.title("SheetMind: AI-Powered Data Analyzer 📊")
st.markdown("""
Welcome to SheetMind! Upload your data file (CSV, Excel, Parquet) and ask questions in natural language.
The AI will analyze your data and provide insights, explanations, and even the code used.
""")

# --- Agent Initialization (Cached) ---
@st.cache_resource
def get_controller_agent():
    """Initializes and returns the ControllerAgent."""
    # IMPORTANT: GeminiClient (used by DataAnalysisAgent) typically requires an API key.
    # Ensure GEMINI_API_KEY (or GOOGLE_API_KEY, depending on client) is set in your environment.
    # Example: os.environ['GOOGLE_API_KEY'] = 'YOUR_API_KEY'
    # If not set, GeminiClient initialization might fail within the agent.
    try:
        agent = ControllerAgent(agent_id="streamlit_controller_001")
        # You might need to explicitly initialize parts of the agent if not done in __init__
        # For example, if the sandbox for DataAnalysisAgent needs setup:
        # asyncio.run(agent._data_analysis_agent._initialize_sandbox_environment()) # If needed and sync
        return agent
    except Exception as e:
        st.error(f"Error initializing ControllerAgent: {e}. Check API keys and configurations.")
        st.stop()
        return None

controller = get_controller_agent()

# --- Session State Initialization ---
if 'uploaded_file_local_path' not in st.session_state:
    st.session_state.uploaded_file_local_path = None
if 'original_filename' not in st.session_state:
    st.session_state.original_filename = None
if 'analysis_result_output' not in st.session_state:
    st.session_state.analysis_result_output = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'analysis_in_progress' not in st.session_state:
    st.session_state.analysis_in_progress = False

# --- UI Layout ---
st.sidebar.header("Controls")

# Get the absolute path to the project root (Csv-QA)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# Define a consistent temporary uploads directory within the project if desired, or use system temp
# Using system temp is generally safer and cleaner
TEMP_DIR = tempfile.gettempdir()
SHEETMIND_TEMP_UPLOADS = os.path.join(TEMP_DIR, "sheetmind_uploads")
os.makedirs(SHEETMIND_TEMP_UPLOADS, exist_ok=True)

uploaded_file_obj = st.sidebar.file_uploader("1. Upload your Data File", type=["csv", "xlsx", "xls", "parquet"], key="file_uploader")

if uploaded_file_obj:
    st.sidebar.info(f"File '{uploaded_file_obj.name}' selected.")
    if st.sidebar.button("Process Uploaded File", key="process_upload_button"):
        with st.spinner(f"Processing '{uploaded_file_obj.name}'..."):
            try:
                # Save uploaded file to a temporary local path
                _, file_extension = os.path.splitext(uploaded_file_obj.name)
                # Generate a unique filename to avoid collisions
                temp_filename = f"{uuid.uuid4().hex}{file_extension}"
                local_save_path = os.path.join(SHEETMIND_TEMP_UPLOADS, temp_filename)
                
                with open(local_save_path, "wb") as f:
                    f.write(uploaded_file_obj.getvalue())
                
                st.session_state.uploaded_file_local_path = local_save_path
                st.session_state.original_filename = uploaded_file_obj.name
                st.sidebar.success(f"File '{uploaded_file_obj.name}' processed and ready.")
                # st.sidebar.caption(f"Temp path: {local_save_path}") # Optional: for debugging
                st.session_state.analysis_result_output = None # Clear previous results
                st.session_state.error_message = None
            except Exception as e:
                st.session_state.error_message = f"Error processing file: {e}\n{traceback.format_exc()}"
                st.sidebar.error(st.session_state.error_message)

if st.session_state.uploaded_file_local_path:
    st.sidebar.markdown("---")
    st.sidebar.header(f"2. Ask about '{st.session_state.original_filename}'")
    query = st.sidebar.text_area("Enter your question about the data:", height=100, key="query_input")
    
    if st.sidebar.button("Analyze Data", key="analyze_button", disabled=st.session_state.analysis_in_progress):
        if not query.strip():
            st.sidebar.warning("Please enter a question.")
        elif controller is None: # Check if agent initialization failed
             st.error("Analysis cannot proceed: Controller Agent is not available. Check earlier errors.")
        else:
            st.session_state.analysis_in_progress = True
            st.session_state.error_message = None
            st.session_state.analysis_result_output = None
            # Use st.empty() for dynamic updates within the spinner context if needed
            # status_placeholder = st.empty()
            with st.spinner(" SheetMind is thinking... This may take a moment."):
                # status_placeholder.info("Initializing analysis...")
                try:
                    # Streamlit can run asyncio.run implicitly for async button callbacks
                    # but for clarity or if issues arise, explicit asyncio.run can be used.
                    # status_placeholder.info(f"Sending query for {st.session_state.original_filename}...")
                    agent_response: AgentResponse = asyncio.run(controller.execute(
                        task="analyze", 
                        query=query, 
                        data_path=st.session_state.uploaded_file_local_path
                    ))
                    # status_placeholder.info("Analysis complete. Processing results...")
                    
                    if agent_response.success:
                        st.session_state.analysis_result_output = agent_response.output
                        st.session_state.error_message = None
                    else:
                        st.session_state.analysis_result_output = None
                        err_msg = agent_response.error or "Analysis failed with an unknown error from the agent."
                        if agent_response.metadata and 'traceback' in agent_response.metadata:
                            err_msg += f"\n\nAgent Traceback:\n{agent_response.metadata['traceback']}"
                        st.session_state.error_message = err_msg
                except Exception as e:
                    st.session_state.analysis_result_output = None
                    st.session_state.error_message = f"An critical error occurred during analysis: {e}\n{traceback.format_exc()}"
                finally:
                    st.session_state.analysis_in_progress = False
                    # status_placeholder.empty()
                    # Use st.experimental_rerun() to ensure UI updates immediately after async operation
                    st.experimental_rerun()

st.markdown("---")

# --- Display Error Messages centrally ---
if st.session_state.error_message:
    st.error(st.session_state.error_message)

# --- Display Analysis Results ---
if st.session_state.analysis_result_output:
    st.subheader("Analysis Results")
    
    output_data = st.session_state.analysis_result_output
    
    if output_data.get("explanation"):
        st.markdown("#### Explanation")
        st.info(output_data["explanation"])
    
    # Display 'result' which can be varied (text, number, list of dicts for table)
    if "result" in output_data: # Check for key existence
        st.markdown("#### Result")
        res_val = output_data["result"]
        if res_val is None:
            st.text("(No direct result value returned)")
        elif isinstance(res_val, (list, tuple)):
            if res_val and all(isinstance(item, dict) for item in res_val): # List of dicts -> DataFrame
                try:
                    df_result = pd.DataFrame(list(res_val))
                    st.dataframe(df_result)
                except Exception as e:
                    st.warning(f"Could not display list of dicts as DataFrame: {e}")
                    st.json(res_val)
            else: # Other lists/tuples
                st.json(res_val)
        elif isinstance(res_val, dict): # Dictionary -> JSON or DataFrame
            try:
                # Attempt to make a DataFrame if it's a dict of lists/series
                df_result = pd.DataFrame(res_val)
                st.dataframe(df_result)
            except ValueError: # If not suitable for DataFrame (e.g. mixed types, scalars)
                st.json(res_val)
            except Exception as e:
                 st.warning(f"Could not display dict as DataFrame: {e}")
                 st.json(res_val)
        else: # Scalar values (string, number, bool)
            st.text(str(res_val))

    if output_data.get("data_preview"):
        st.markdown("#### Data Preview (from analysis)")
        preview = output_data["data_preview"]
        # data_preview from DataAnalysisAgent is typically a list of dicts (records)
        if isinstance(preview, list) and preview and isinstance(preview[0], dict):
            try:
                df_preview = pd.DataFrame(preview)
                st.dataframe(df_preview)
            except Exception as e:
                st.warning(f"Could not display data preview as DataFrame: {e}")
                st.json(preview) # Fallback to JSON
        elif isinstance(preview, dict) and preview.get("columns") and preview.get("data"):
             # Old format, handle for compatibility if necessary
            try:
                df_preview = pd.DataFrame(preview["data"], columns=preview["columns"])
                st.dataframe(df_preview)
            except Exception as e:
                st.warning(f"Could not display structured data preview: {e}")
                st.json(preview)
        else:
            st.json(preview) # Fallback to JSON if structure is not as expected

    if output_data.get("plot_image_path"):
        st.markdown("#### Generated Plot")
        # The DataAnalysisAgent saves plots to a path; we need to display it.
        # Ensure this path is accessible by Streamlit.
        # If IsolatedPythonExecutor saves it within its own temp space, this path needs to be handled.
        # For now, assuming it's an accessible path or base64 encoded string.
        plot_path_or_data = output_data["plot_image_path"]
        if os.path.exists(plot_path_or_data):
            st.image(plot_path_or_data)
        # Add handling if it's base64 data later if needed
        else:
            st.warning(f"Plot image not found at: {plot_path_or_data}. It might be in a sandboxed environment.")

    if output_data.get("code"):
        with st.expander("View Generated Code"):
            st.code(output_data["code"], language="python")
            
else:
    if st.session_state.uploaded_file_local_path and not st.session_state.analysis_in_progress:
        st.info("Enter a question in the sidebar and click 'Analyze Data'.")
    elif not st.session_state.analysis_in_progress:
        st.info("Upload a data file using the sidebar to begin.")

st.markdown("---")
st.caption("SheetMind - Powered by AI")

# Add a traceback import for error handling
import traceback
