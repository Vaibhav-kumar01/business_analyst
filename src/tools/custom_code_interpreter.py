from typing import List, Optional
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
from docker import from_env as docker_from_env
from docker.errors import ImageNotFound, NotFound
import os

class CodeExecutorSchema(BaseModel):
    """Input schema for CustomCodeInterpreterTool."""
    
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
    
    # Configuration (these are proper Pydantic fields)
    image_name: str = "data-science-env:latest"
    container_name: str = "persistent-code-executor"
    verbose: bool = True
    
    # Use PrivateAttr for internal state that shouldn't be part of the model schema
    _container = PrivateAttr(default=None)
    _installed_libraries = PrivateAttr(default_factory=list)
    _dockerfile_path = PrivateAttr(default=None)
    
    def __init__(self, **data):
        """Initialize with proper kwargs handling for Pydantic."""
        super().__init__(**data)
        self._installed_libraries = []
        
        # Store dockerfile path if provided
        if "dockerfile_path" in data:
            self._dockerfile_path = data["dockerfile_path"]
        
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
            if not self._dockerfile_path:
                raise ValueError(f"Docker image {self.image_name} not found and no Dockerfile provided")
            
            if not os.path.exists(self._dockerfile_path):
                raise FileNotFoundError(f"Dockerfile not found at {self._dockerfile_path}")
            
            self._log(f"Building Docker image {self.image_name} from {self._dockerfile_path}")
            client.images.build(
                path=os.path.dirname(self._dockerfile_path),
                dockerfile=os.path.basename(self._dockerfile_path),
                tag=self.image_name,
                rm=True
            )
            self._log(f"Successfully built image: {self.image_name}")
    
    def _ensure_container_running(self) -> None:
        """Ensure a container is running, creating one if needed."""
        if self._container is not None:
            # Container reference exists, check if it's still running
            try:
                client = docker_from_env()
                container = client.containers.get(self.container_name)
                if container.status != "running":
                    self._log(f"Container exists but not running. Starting it...")
                    container.start()
                self._container = container
                self._log(f"Using existing container: {container.short_id}")
                return
            except NotFound:
                self._log(f"Container reference exists but container not found. Creating new one.")
                self._container = None
        
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
        self._container = client.containers.run(
            self.image_name,
            detach=True,
            tty=True,
            stdin_open=True,
            working_dir="/workspace",
            name=self.container_name,
            volumes={os.getcwd(): {"bind": "/workspace", "mode": "rw"}},
            remove=False  # Don't auto-remove when stopped
        )
        self._log(f"Created container: {self._container.short_id}")
    
    def _install_libraries(self, libraries: List[str]) -> None:
        """Install libraries that haven't been installed yet."""
        if not self._container:
            self._ensure_container_running()
        
        for library in libraries:
            if library in self._installed_libraries:
                self._log(f"Library already installed, skipping: {library}")
                continue
                
            self._log(f"Installing library: {library}")
            result = self._container.exec_run(["pip", "install", "--no-cache-dir", library])
            
            if result.exit_code == 0:
                self._installed_libraries.append(library)
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
            result = self._container.exec_run(
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
        if self._container:
            self._log("Cleaning up resources...")
            try:
                self._container.stop()
                self._container.remove()
                self._container = None
                self._log("Container stopped and removed")
            except Exception as e:
                self._log(f"Error during cleanup: {str(e)}")