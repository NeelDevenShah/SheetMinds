"""
Visualization Agent - Handles data visualization tasks.
"""
import logging
import base64
import io
from typing import Dict, Any, Optional, List, Union, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from .base_agent import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)

class VisualizationAgent(BaseAgent):
    """
    Agent responsible for creating visualizations from data.
    """
    
    def __init__(self, 
                 agent_id: str,
                 name: str = "Visualization Agent",
                 description: str = "Creates visualizations from data",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the VisualizationAgent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name
            description: Description of the agent's purpose
            config: Configuration dictionary
        """
        default_config = {
            "allowed_plot_types": [
                "line", "bar", "barh", "hist", "box", "kde", "density",
                "area", "pie", "scatter", "hexbin"
            ],
            "default_figsize": (10, 6),
            "default_style": "whitegrid",
            "default_context": "notebook",
            "max_categories": 20,
            "max_points": 10000,
            "dpi": 100,
            "output_format": "base64",  # or 'figure', 'bytes', 'file'
            "output_file": "plot.png"
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(agent_id, name, description, default_config)
        
        # Set matplotlib style and context
        plt.style.use(default_config["default_style"])
        sns.set_context(default_config["default_context"])
    
    async def initialize(self):
        """Initialize the agent and any required resources."""
        logger.info(f"Initializing {self.name} (ID: {self.agent_id})")
        self.state = "ready"
    
    async def shutdown(self):
        """Clean up resources used by the agent."""
        logger.info(f"Shutting down {self.name} (ID: {self.agent_id})")
        plt.close('all')
        self.state = "shutdown"
    
    async def create_visualization(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
        plot_type: str,
        x: Optional[str] = None,
        y: Optional[Union[str, List[str]]] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        output_format: Optional[str] = None,
        **kwargs
    ) -> AgentResponse:
        """
        Create a visualization from the provided data.
        
        Args:
            data: Input data as DataFrame or list of dictionaries
            plot_type: Type of plot to create (e.g., 'line', 'bar', 'hist')
            x: Column name for x-axis
            y: Column name(s) for y-axis
            title: Plot title
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            figsize: Figure size as (width, height) in inches
            output_format: Format of the output ('base64', 'figure', 'bytes', 'file')
            **kwargs: Additional arguments for the plot
            
        Returns:
            AgentResponse containing the visualization
        """
        try:
            # Convert data to DataFrame if it's not already
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
            # Validate plot type
            plot_type = plot_type.lower()
            if plot_type not in self.config["allowed_plot_types"]:
                return self._create_response(
                    success=False,
                    error=f"Plot type '{plot_type}' is not allowed. "
                          f"Allowed types: {', '.join(self.config['allowed_plot_types'])}"
                )
            
            # Set default values
            if figsize is None:
                figsize = self.config["default_figsize"]
            
            if output_format is None:
                output_format = self.config["output_format"]
            
            # Create the plot
            fig, ax = plt.subplots(figsize=figsize, dpi=self.config["dpi"])
            
            # Handle different plot types
            if plot_type == 'line':
                data.plot.line(x=x, y=y, ax=ax, **kwargs)
            elif plot_type == 'bar':
                data.plot.bar(x=x, y=y, ax=ax, **kwargs)
            elif plot_type == 'barh':
                data.plot.barh(x=x, y=y, ax=ax, **kwargs)
            elif plot_type == 'hist':
                data.plot.hist(y=y, ax=ax, **kwargs)
            elif plot_type == 'box':
                data.plot.box(y=y, ax=ax, **kwargs)
            elif plot_type in ['kde', 'density']:
                data.plot.kde(y=y, ax=ax, **kwargs)
            elif plot_type == 'area':
                data.plot.area(x=x, y=y, ax=ax, **kwargs)
            elif plot_type == 'pie':
                data.plot.pie(y=y, ax=ax, **kwargs)
            elif plot_type == 'scatter':
                if x and y:
                    data.plot.scatter(x=x, y=y, ax=ax, **kwargs)
                else:
                    return self._create_response(
                        success=False,
                        error="Both x and y must be specified for scatter plot"
                    )
            elif plot_type == 'hexbin':
                if x and y:
                    data.plot.hexbin(x=x, y=y, ax=ax, **kwargs)
                else:
                    return self._create_response(
                        success=False,
                        error="Both x and y must be specified for hexbin plot"
                    )
            else:
                return self._create_response(
                    success=False,
                    error=f"Unsupported plot type: {plot_type}"
                )
            
            # Set plot title and labels
            if title:
                ax.set_title(title)
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)
            
            # Adjust layout
            plt.tight_layout()
            
            # Handle output format
            result = None
            if output_format == 'figure':
                result = fig
            elif output_format == 'base64':
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=self.config["dpi"], bbox_inches='tight')
                buf.seek(0)
                result = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
            elif output_format == 'bytes':
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=self.config["dpi"], bbox_inches='tight')
                buf.seek(0)
                result = buf.read()
                buf.close()
            elif output_format == 'file':
                output_file = self.config["output_file"]
                fig.savefig(output_file, dpi=self.config["dpi"], bbox_inches='tight')
                result = output_file
            
            plt.close(fig)
            
            return self._create_response(
                success=True,
                result={
                    'figure': result,
                    'format': output_format,
                    'plot_type': plot_type
                }
            )
            
        except Exception as e:
            logger.exception(f"Error creating {plot_type} plot")
            return self._create_response(
                success=False,
                error=f"Error creating visualization: {str(e)}",
                traceback=self._get_traceback()
            )
    
    async def create_correlation_heatmap(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
        method: str = 'pearson',
        figsize: Optional[Tuple[int, int]] = None,
        output_format: Optional[str] = None,
        **kwargs
    ) -> AgentResponse:
        """
        Create a correlation heatmap for the given data.
        
        Args:
            data: Input data as DataFrame or list of dictionaries
            method: Correlation method ('pearson', 'kendall', 'spearman')
            figsize: Figure size as (width, height) in inches
            output_format: Format of the output ('base64', 'figure', 'bytes', 'file')
            **kwargs: Additional arguments for the heatmap
            
        Returns:
            AgentResponse containing the correlation heatmap
        """
        try:
            # Convert data to DataFrame if it's not already
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
            # Calculate correlation matrix
            corr = data.corr(method=method, numeric_only=True)
            
            # Set default values
            if figsize is None:
                # Make the figure larger for better readability
                size = max(8, len(corr.columns) * 0.8)
                figsize = (size, size)
            
            if output_format is None:
                output_format = self.config["output_format"]
            
            # Create the plot
            fig, ax = plt.subplots(figsize=figsize, dpi=self.config["dpi"])
            
            # Create heatmap
            sns.heatmap(
                corr,
                annot=True,
                fmt=".2f",
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                ax=ax,
                **kwargs
            )
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            # Adjust layout
            plt.tight_layout()
            
            # Handle output format
            result = None
            if output_format == 'figure':
                result = fig
            elif output_format == 'base64':
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=self.config["dpi"], bbox_inches='tight')
                buf.seek(0)
                result = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
            elif output_format == 'bytes':
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=self.config["dpi"], bbox_inches='tight')
                buf.seek(0)
                result = buf.read()
                buf.close()
            elif output_format == 'file':
                output_file = self.config["output_file"].replace('.png', '_corr.png')
                fig.savefig(output_file, dpi=self.config["dpi"], bbox_inches='tight')
                result = output_file
            
            plt.close(fig)
            
            return self._create_response(
                success=True,
                result={
                    'figure': result,
                    'format': output_format,
                    'plot_type': 'correlation_heatmap',
                    'correlation_matrix': corr.to_dict()
                }
            )
            
        except Exception as e:
            logger.exception("Error creating correlation heatmap")
            return self._create_response(
                success=False,
                error=f"Error creating correlation heatmap: {str(e)}",
                traceback=self._get_traceback()
            )
    
    async def create_pair_plot(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
        hue: Optional[str] = None,
        vars: Optional[List[str]] = None,
        diag_kind: str = 'auto',
        height: float = 2.5,
        aspect: float = 1.0,
        output_format: Optional[str] = None,
        **kwargs
    ) -> AgentResponse:
        """
        Create a pair plot (scatter plot matrix) for the given data.
        
        Args:
            data: Input data as DataFrame or list of dictionaries
            hue: Variable in data to map plot aspects to different colors
            vars: Variables to include in the plot
            diag_kind: Kind of plot for the diagonal subplots ('auto', 'hist', 'kde', None)
            height: Height (in inches) of each facet
            aspect: Aspect ratio of each facet
            output_format: Format of the output ('base64', 'figure', 'bytes', 'file')
            **kwargs: Additional arguments for the pair plot
            
        Returns:
            AgentResponse containing the pair plot
        """
        try:
            # Convert data to DataFrame if it's not already
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
            if output_format is None:
                output_format = self.config["output_format"]
            
            # Limit the number of variables for performance
            if vars is None:
                # Use only numeric columns
                numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                vars = numeric_cols[:6]  # Limit to first 6 numeric columns
                if len(numeric_cols) > 6:
                    logger.warning(f"Using first 6 of {len(numeric_cols)} numeric columns for pair plot")
            
            if not vars or len(vars) < 2:
                return self._create_response(
                    success=False,
                    error="At least 2 numeric variables are required for pair plot"
                )
            
            # Sample data if too large for better performance
            plot_data = data
            if len(plot_data) > self.config["max_points"]:
                plot_data = plot_data.sample(n=self.config["max_points"], random_state=42)
                logger.warning(f"Sampled {self.config['max_points']} points out of {len(data)} for pair plot")
            
            # Create the pair plot
            pair_grid = sns.pairplot(
                plot_data,
                vars=vars,
                hue=hue,
                diag_kind=diag_kind,
                height=height,
                aspect=aspect,
                plot_kws={"s": 15, "alpha": 0.7, "edgecolor": "k"},
                diag_kws={"edgecolor": "k"},
                **kwargs
            )
            
            # Adjust title and labels
            pair_grid.fig.suptitle("Pair Plot", y=1.02)
            
            # Handle output format
            result = None
            if output_format == 'figure':
                result = pair_grid.fig
            else:
                # For other formats, we need to save to a buffer first
                buf = io.BytesIO()
                pair_grid.savefig(buf, format='png', dpi=self.config["dpi"], bbox_inches='tight')
                buf.seek(0)
                
                if output_format == 'base64':
                    result = base64.b64encode(buf.read()).decode('utf-8')
                elif output_format == 'bytes':
                    result = buf.read()
                elif output_format == 'file':
                    output_file = self.config["output_file"].replace('.png', '_pairplot.png')
                    with open(output_file, 'wb') as f:
                        f.write(buf.getvalue())
                    result = output_file
                
                buf.close()
            
            plt.close(pair_grid.fig)
            
            return self._create_response(
                success=True,
                result={
                    'figure': result,
                    'format': output_format,
                    'plot_type': 'pair_plot',
                    'variables': vars,
                    'hue': hue
                }
            )
            
        except Exception as e:
            logger.exception("Error creating pair plot")
            return self._create_response(
                success=False,
                error=f"Error creating pair plot: {str(e)}",
                traceback=self._get_traceback()
            )
