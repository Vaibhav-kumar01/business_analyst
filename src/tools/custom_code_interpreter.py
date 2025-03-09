import os
from typing import List, Optional, Dict, Any
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from docker import from_env as docker_from_env
from docker.errors import ImageNotFound, NotFound

class CodeExecutorSchema(BaseModel):
    """Input schema for PersistentCodeExecutor."""
    
    code: str = Field(
        ...,
        description="Python code to execute. Always include print statements for output you want to see.",
    )
    
    libraries_used: List[str] = Field(
        default=[],
        description="List of libraries to use (e.g., pandas, numpy, matplotlib). Only specify libraries not already installed.",
    )

class CustomCodeInterpreterTool(BaseTool):
    """Executes Python code in a persistent Docker container."""
    
    name: str = "Code Executor"
    description: str = "Executes Python code and maintains state between executions."
    args_schema: type[BaseModel] = CodeExecutorSchema
    
    # Configuration
    image_name: str = "data-science-env:latest"
    container_name: str = "persistent-code-executor"
    
    # State tracking
    _container = None
    _installed_libraries: List[str] = []
    
    def __init__(self, 
                 image_name: Optional[str] = None, 
                 dockerfile_path: Optional[str] = None,
                 container_name: Optional[str] = None,
                 verbose: bool = True):
        """
        Initialize the code executor.
        
        Args:
            image_name: Docker image to use (defaults to "data-science-env:latest")
            dockerfile_path: Path to Dockerfile if image needs to be built
            container_name: Name for the persistent container
            verbose: Whether to print detailed logs
        """
        super().__init__()
        if image_name:
            self.image_name = image_name
        if container_name:
            self.container_name = container_name
        self.dockerfile_path = dockerfile_path
        self.verbose = verbose
        
        # Initialize container on startup
        self._ensure_container_running()
    
    def _log(self, message: str) -> None:
        """Print log messages if verbose mode is enabled."""
        if self.verbose:
            print(f"[CodeExecutor] {message}")
    
    def _ensure_image_exists(self) -> None:
        """Make sure the Docker image exists, building it if needed."""
        client = docker_from_env()
        
        try:
            client.images.get(self.image_name)
            self._log(f"Using existing Docker image: {self.image_name}")
        except ImageNotFound:
            if not self.dockerfile_path:
                raise ValueError(f"Docker image {self.image_name} not found and no Dockerfile provided")
            
            if not os.path.exists(self.dockerfile_path):
                raise FileNotFoundError(f"Dockerfile not found at {self.dockerfile_path}")
            
            self._log(f"Building Docker image {self.image_name} from {self.dockerfile_path}")
            client.images.build(
                path=os.path.dirname(self.dockerfile_path),
                dockerfile=os.path.basename(self.dockerfile_path),
                tag=self.image_name,
                rm=True
            )
            self._log(f"Successfully built image: {self.image_name}")
    
    def _ensure_container_running(self) -> None:
        """Ensure a container is running, creating one if needed."""
        if CustomCodeInterpreterTool._container is not None:
            # Container reference exists, check if it's still running
            try:
                client = docker_from_env()
                container = client.containers.get(self.container_name)
                if container.status != "running":
                    self._log(f"Container exists but not running. Starting it...")
                    container.start()
                CustomCodeInterpreterTool._container = container
                self._log(f"Using existing container: {container.short_id}")
                return
            except NotFound:
                self._log(f"Container reference exists but container not found. Creating new one.")
                CustomCodeInterpreterTool._container = None
        
        # Create a new container
        self._ensure_image_exists()
        client = docker_from_env()
        
        # Remove existing container if any
        try:
            old_container = client.containers.get(self.container_name)
            self._log(f"Removing old container: {old_container.short_id}")
            old_container.stop()
            old_container.remove()
        except NotFound:
            pass
        
        # Create and start new container
        self._log(f"Creating new container from image: {self.image_name}")
        CustomCodeInterpreterTool._container = client.containers.run(
            self.image_name,
            detach=True,
            tty=True,
            stdin_open=True,
            working_dir="/workspace",
            name=self.container_name,
            volumes={os.getcwd(): {"bind": "/workspace", "mode": "rw"}},
            remove=False  # Don't auto-remove when stopped
        )
        self._log(f"Created container: {CustomCodeInterpreterTool._container.short_id}")
    
    def _install_libraries(self, libraries: List[str]) -> None:
        """Install libraries that haven't been installed yet."""
        container = CustomCodeInterpreterTool._container
        if not container:
            self._ensure_container_running()
            container = CustomCodeInterpreterTool._container
        
        for library in libraries:
            if library in CustomCodeInterpreterTool._installed_libraries:
                self._log(f"Library already installed, skipping: {library}")
                continue
                
            self._log(f"Installing library: {library}")
            result = container.exec_run(["pip", "install", "--no-cache-dir", library])
            
            if result.exit_code == 0:
                CustomCodeInterpreterTool._installed_libraries.append(library)
                if self.verbose:
                    self._log(f"Successfully installed {library}")
            else:
                self._log(f"Failed to install {library}: {result.output.decode('utf-8')}")
    
    def _run(self, code: str = "", libraries_used: List[str] = []) -> str:
        """Execute code in the persistent container."""
        self._log(f"Executing code with {len(libraries_used)} libraries")
        
        try:
            # Ensure container is running
            self._ensure_container_running()
            
            # Install any requested libraries
            if libraries_used:
                self._install_libraries(libraries_used)
            
            # Execute the code
            self._log("Running code...")
            result = CustomCodeInterpreterTool._container.exec_run(
                ["python3", "-c", code],
                environment={"PYTHONIOENCODING": "utf-8"}
            )
            
            # Process the result
            output = result.output.decode("utf-8")
            if result.exit_code != 0:
                self._log(f"Code execution failed with exit code {result.exit_code}")
                return f"Error executing code:\n{output}"
            
            self._log("Code executed successfully")
            return output
            
        except Exception as e:
            self._log(f"Error: {str(e)}")
            return f"Internal error: {str(e)}"
    
    def cleanup(self) -> None:
        """Stop and remove the container (call at end of session)."""
        if CustomCodeInterpreterTool._container:
            self._log("Cleaning up resources...")
            try:
                CustomCodeInterpreterTool._container.stop()
                CustomCodeInterpreterTool._container.remove()
                CustomCodeInterpreterTool._container = None
                self._log("Container stopped and removed")
            except Exception as e:
                self._log(f"Error during cleanup: {str(e)}")