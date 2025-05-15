from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type, TypeVar
from dataclasses import dataclass, asdict
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T', bound='BaseAgent')

@dataclass
class AgentResponse:
    """Standard response format for agent operations."""
    success: bool
    output: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentResponse':
        return cls(**data)

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class BaseAgent(ABC):
    """Base class for all agents in the SheetMind system."""
    
    def __init__(self, 
                 agent_id: str, 
                 name: str, 
                 description: str,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name of the agent
            description: Description of the agent's purpose and capabilities
            config: Configuration dictionary for the agent
        """
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.config = config or {}
        self.memory = {}
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Set up logging for the agent."""
        self.logger = logging.getLogger(f"{self.__class__.__name__}:{self.agent_id}")
    
    @abstractmethod
    async def execute(self, task: str, **kwargs) -> AgentResponse:
        """
        Execute the agent's primary task.
        
        Args:
            task: The task to perform
            **kwargs: Additional arguments specific to the agent implementation
            
        Returns:
            AgentResponse containing the result of the execution
        """
        pass
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update the agent's configuration.
        
        Args:
            new_config: Dictionary containing configuration updates
        """
        self.config.update(new_config)
        self.logger.info(f"Updated configuration: {new_config}")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the agent.
        
        Returns:
            Dictionary containing the agent's state
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "config": self.config,
            "memory": self.memory
        }
    
    def save_state(self, file_path: str) -> bool:
        """
        Save the agent's state to a file.
        
        Args:
            file_path: Path to save the state file
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            state = self.get_state()
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
            self.logger.info(f"Saved state to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}")
            return False
    
    @classmethod
    def load_state(cls: Type[T], file_path: str) -> Optional[T]:
        """
        Load an agent's state from a file.
        
        Args:
            file_path: Path to the state file
            
        Returns:
            An instance of the agent with the loaded state, or None if loading failed
        """
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            agent = cls(
                agent_id=state['agent_id'],
                name=state['name'],
                description=state['description'],
                config=state.get('config', {})
            )
            agent.memory = state.get('memory', {})
            return agent
        except Exception as e:
            logger.error(f"Error loading agent state: {str(e)}")
            return None
    
    def __str__(self) -> str:
        """Return a string representation of the agent."""
        return (f"{self.__class__.__name__}(agent_id='{self.agent_id}', "
                f"name='{self.name}', description='{self.description}')")


class AgentFactory:
    """Factory class for creating and managing agents."""
    
    _registry: Dict[str, Type[BaseAgent]] = {}
    
    @classmethod
    def register(cls, agent_type: str, agent_class: Type[BaseAgent]) -> None:
        """
        Register an agent class with the factory.
        
        Args:
            agent_type: String identifier for the agent type
            agent_class: The agent class to register
        """
        if not issubclass(agent_class, BaseAgent):
            raise ValueError(f"Agent class must be a subclass of BaseAgent")
        cls._registry[agent_type] = agent_class
    
    @classmethod
    def create_agent(cls, 
                    agent_type: str, 
                    agent_id: str, 
                    name: str, 
                    description: str,
                    config: Optional[Dict[str, Any]] = None) -> BaseAgent:
        """
        Create a new agent instance.
        
        Args:
            agent_type: Type of agent to create
            agent_id: Unique identifier for the agent
            name: Human-readable name of the agent
            description: Description of the agent's purpose
            config: Configuration dictionary for the agent
            
        Returns:
            An instance of the requested agent
            
        Raises:
            ValueError: If the agent type is not registered
        """
        agent_class = cls._registry.get(agent_type)
        if agent_class is None:
            raise ValueError(f"Unknown agent type: {agent_type}")
        return agent_class(agent_id, name, description, config or {})
