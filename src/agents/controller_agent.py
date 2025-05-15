from typing import Dict, Any, Optional, List, Type, TypeVar
from dataclasses import dataclass
import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
import json

from .base_agent import BaseAgent, AgentResponse, AgentFactory
from ..tools.isolated_python_executor import IsolatedPythonExecutor

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Represents a task to be executed by an agent."""
    task_id: str
    description: str
    agent_type: str
    priority: int = 1
    dependencies: List[str] = None
    params: Dict[str, Any] = None
    created_at: str = None
    status: str = "pending"
    result: Any = None
    error: str = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.params is None:
            self.params = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "agent_type": self.agent_type,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "params": self.params,
            "created_at": self.created_at,
            "status": self.status,
            "result": self.result,
            "error": self.error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        return cls(**data)


class ControllerAgent(BaseAgent):
    """
    The ControllerAgent is responsible for managing and coordinating all other agents
    in the SheetMind system. It handles task delegation, dependency resolution, and
    result aggregation.
    """
    
    def __init__(self, 
                 agent_id: str = "controller_001",
                 name: str = "SheetMind Controller",
                 description: str = "Manages and coordinates all agents in the SheetMind system",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ControllerAgent.
        
        Args:
            agent_id: Unique identifier for the controller
            name: Human-readable name
            description: Description of the controller's purpose
            config: Configuration dictionary
        """
        default_config = {
            "max_retries": 3,
            "task_timeout": 300,  # seconds
            "max_parallel_tasks": 5,
            "state_file": "controller_state.json"
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(agent_id, name, description, default_config)
        
        # Initialize task tracking
        self.tasks: Dict[str, Task] = {}
        self.agents: Dict[str, BaseAgent] = {}
        self.task_queue = asyncio.PriorityQueue()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.failed_tasks: Dict[str, Task] = {}
        
        # Initialize the isolated Python executor
        self.executor = IsolatedPythonExecutor("sheetmind_execution")
        
        # Register built-in agent types
        self._register_builtin_agents()
    
    async def register_agent(self, agent: BaseAgent) -> None:
        """
        Register an agent instance with the controller.
        
        Args:
            agent: The agent instance to register
            
        Returns:
            None
        """
        if agent.agent_id in self.agents:
            self.logger.warning(f"Agent with ID {agent.agent_id} is already registered")
            return
            
        self.agents[agent.agent_id] = agent
        self.logger.info(f"Registered agent: {agent.name} (ID: {agent.agent_id})")
    
    def _register_builtin_agents(self) -> None:
        """Register built-in agent types with the factory."""
        # This will be populated as we implement more agents
        from .data_analysis_agent import DataAnalysisAgent
        from .formula_agent import FormulaAgent
        from .visualization_agent import VisualizationAgent
        
        AgentFactory.register("data_analysis", DataAnalysisAgent)
        AgentFactory.register("formula", FormulaAgent)
        AgentFactory.register("visualization", VisualizationAgent)
    
    async def execute(self, task: str, **kwargs) -> AgentResponse:
        """
        Execute a high-level task by coordinating multiple agents.
        
        Args:
            task: The high-level task to perform
            **kwargs: Additional parameters for the task
            
        Returns:
            AgentResponse with the result of the task
        """
        self.logger.info(f"Executing task: {task}")
        
        try:
            # Step 1: Plan the task execution
            plan = await self._plan_task_execution(task, **kwargs)
            
            # Step 2: Create and schedule subtasks
            task_ids = []
            for subtask in plan.get("subtasks", []):
                task_id = self.create_task(
                    description=subtask["description"],
                    agent_type=subtask["agent_type"],
                    priority=subtask.get("priority", 1),
                    params=subtask.get("params", {})
                )
                task_ids.append(task_id)
            
            # Step 3: Execute tasks in parallel with dependency resolution
            results = await self.execute_tasks()
            
            # Step 4: Aggregate and return results
            return AgentResponse(
                success=True,
                output={
                    "task": task,
                    "results": results,
                    "summary": plan.get("summary", "Task completed successfully")
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error executing task: {str(e)}", exc_info=True)
            return AgentResponse(
                success=False,
                output=None,
                error=f"Error executing task: {str(e)}",
                metadata={"traceback": str(e.__traceback__)}
            )
    
    async def _plan_task_execution(self, task: str, **kwargs) -> Dict[str, Any]:
        """
        Plan the execution of a high-level task.
        
        Args:
            task: The task to plan
            **kwargs: Additional parameters
            
        Returns:
            A dictionary containing the execution plan
        """
        # TODO: Implement more sophisticated planning with LLM integration
        # For now, return a simple plan
        return {
            "task": task,
            "subtasks": [
                {
                    "description": f"Analyze task: {task}",
                    "agent_type": "data_analysis",
                    "priority": 1,
                    "params": {"query": task, **kwargs}
                }
            ],
            "summary": f"Planned execution for task: {task}"
        }
    
    def create_task(self, 
                   description: str, 
                   agent_type: str,
                   priority: int = 1,
                   dependencies: List[str] = None,
                   params: Dict[str, Any] = None) -> str:
        """
        Create a new task.
        
        Args:
            description: Description of the task
            agent_type: Type of agent to handle the task
            priority: Task priority (higher is more important)
            dependencies: List of task IDs this task depends on
            params: Parameters for the task
            
        Returns:
            The ID of the created task
        """
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        task = Task(
            task_id=task_id,
            description=description,
            agent_type=agent_type,
            priority=priority,
            dependencies=dependencies or [],
            params=params or {}
        )
        
        self.tasks[task_id] = task
        asyncio.create_task(self._enqueue_task(task))
        
        self.logger.info(f"Created task {task_id}: {description}")
        return task_id
    
    async def _enqueue_task(self, task: Task) -> None:
        """Add a task to the execution queue."""
        # Wait for dependencies to complete
        for dep_id in task.dependencies:
            while dep_id not in self.completed_tasks and dep_id not in self.failed_tasks:
                await asyncio.sleep(0.1)
        
        # Add to queue with negative priority for max-heap behavior
        await self.task_queue.put((-task.priority, task.task_id, task))
    
    async def execute_tasks(self) -> Dict[str, Any]:
        """
        Execute all tasks in the queue with dependency resolution.
        
        Returns:
            Dictionary mapping task IDs to their results
        """
        results = {}
        
        while not self.task_queue.empty() or self.running_tasks:
            # Start new tasks if we're under the parallel limit
            while len(self.running_tasks) < self.config["max_parallel_tasks"] and not self.task_queue.empty():
                _, task_id, task = await self.task_queue.get()
                self.logger.info(f"Starting task {task_id}: {task.description}")
                
                # Create task coroutine
                task_coro = self._execute_single_task(task)
                # Create and store the task
                task_handle = asyncio.create_task(task_coro)
                self.running_tasks[task_id] = task_handle
            
            # Wait for at least one task to complete
            if self.running_tasks:
                done, _ = await asyncio.wait(
                    list(self.running_tasks.values()),
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Process completed tasks
                for task_handle in done:
                    task_id = next(
                        tid for tid, t in self.running_tasks.items() 
                        if t == task_handle
                    )
                    
                    # Get the result and update task status
                    try:
                        result = task_handle.result()
                        task = self.tasks[task_id]
                        task.status = "completed"
                        task.result = result
                        self.completed_tasks[task_id] = task
                        results[task_id] = result
                        self.logger.info(f"Completed task {task_id}")
                    except Exception as e:
                        task = self.tasks[task_id]
                        task.status = "failed"
                        task.error = str(e)
                        self.failed_tasks[task_id] = task
                        self.logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)
                    
                    # Clean up
                    del self.running_tasks[task_id]
                    del self.tasks[task_id]
            
            # Small sleep to prevent busy waiting
            await asyncio.sleep(0.1)
        
        return results
    
    async def _execute_single_task(self, task: Task) -> Any:
        """
        Execute a single task using the appropriate agent.
        
        Args:
            task: The task to execute
            
        Returns:
            The result of the task execution
            
        Raises:
            Exception: If the task fails
        """
        agent = await self._get_or_create_agent(task.agent_type, task.task_id)
        
        try:
            # Execute the task with timeout
            task_future = asyncio.wait_for(
                agent.execute(task.description, **task.params),
                timeout=self.config["task_timeout"]
            )
            
            result = await task_future
            return result
            
        except asyncio.TimeoutError:
            raise Exception(f"Task timed out after {self.config['task_timeout']} seconds")
        except Exception as e:
            raise Exception(f"Error executing task: {str(e)}")
    
    async def _get_or_create_agent(self, agent_type: str, task_id: str) -> BaseAgent:
        """
        Get an existing agent or create a new one if it doesn't exist.
        
        Args:
            agent_type: Type of agent to get or create
            task_id: ID of the task that will use this agent
            
        Returns:
            An instance of the requested agent
        """
        agent_id = f"{agent_type}_{task_id}"
        
        if agent_id not in self.agents:
            try:
                agent = AgentFactory.create_agent(
                    agent_type=agent_type,
                    agent_id=agent_id,
                    name=f"{agent_type.capitalize()} Agent",
                    description=f"Agent for {agent_type} tasks"
                )
                self.agents[agent_id] = agent
                self.logger.info(f"Created new {agent_type} agent with ID: {agent_id}")
            except Exception as e:
                self.logger.error(f"Failed to create agent of type {agent_type}: {str(e)}")
                raise Exception(f"Unsupported agent type: {agent_type}")
        
        return self.agents[agent_id]
    
    async def shutdown(self) -> None:
        """Shut down the controller and all agents."""
        self.logger.info("Shutting down controller and all agents")
        
        # Cancel all running tasks
        for task in self.running_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete cancellation
        if self.running_tasks:
            await asyncio.wait(
                list(self.running_tasks.values()),
                timeout=5.0,
                return_when=asyncio.ALL_COMPLETED
            )
        
        # Save state
        await self.save_state()
    
    async def save_state(self) -> bool:
        """
        Save the controller's state to a file.
        
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            state = {
                "agent_id": self.agent_id,
                "name": self.name,
                "description": self.description,
                "config": self.config,
                "tasks": {tid: task.to_dict() for tid, task in self.tasks.items()},
                "completed_tasks": {tid: task.to_dict() for tid, task in self.completed_tasks.items()},
                "failed_tasks": {tid: task.to_dict() for tid, task in self.failed_tasks.items()}
            }
            
            state_file = Path(self.config["state_file"])
            state_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.logger.info(f"Saved controller state to {state_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving controller state: {str(e)}")
            return False
    
    @classmethod
    async def load_state(cls, state_file: str) -> 'ControllerAgent':
        """
        Load a controller's state from a file.
        
        Args:
            state_file: Path to the state file
            
        Returns:
            An instance of ControllerAgent with the loaded state
        """
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            controller = cls(
                agent_id=state["agent_id"],
                name=state["name"],
                description=state["description"],
                config=state["config"]
            )
            
            # Restore tasks
            controller.tasks = {
                tid: Task.from_dict(task_data) 
                for tid, task_data in state.get("tasks", {}).items()
            }
            
            controller.completed_tasks = {
                tid: Task.from_dict(task_data) 
                for tid, task_data in state.get("completed_tasks", {}).items()
            }
            
            controller.failed_tasks = {
                tid: Task.from_dict(task_data) 
                for tid, task_data in state.get("failed_tasks", {}).items()
            }
            
            logger.info(f"Loaded controller state from {state_file}")
            return controller
            
        except Exception as e:
            logger.error(f"Error loading controller state: {str(e)}")
            raise
