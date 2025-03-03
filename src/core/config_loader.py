"""
Config Loader for handling YAML configuration files.
"""
import os
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ConfigLoader:
    """Loads and processes configuration from YAML files."""
    
    def __init__(self, config_dir="config"):
        """
        Initialize with config directory.
        
        Args:
            config_dir (str): Path to configuration directory
        """
        self.config_dir = config_dir
        self.configs = {}
    
    def load_config(self, config_name):
        """
        Load a configuration file.
        
        Args:
            config_name (str): Name of the configuration file (without .yaml extension)
            
        Returns:
            dict: The loaded configuration
        """
        config_path = os.path.join(self.config_dir, f"{config_name}.yaml")
        
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                
            # Process environment variable substitutions
            config = self._process_env_vars(config)
            
            self.configs[config_name] = config
            return config
        except Exception as e:
            print(f"Error loading configuration {config_name}: {e}")
            return {}
    
    def get_config(self, config_name):
        """
        Get a loaded configuration.
        
        Args:
            config_name (str): Name of the configuration
            
        Returns:
            dict: The configuration if loaded, empty dict otherwise
        """
        if config_name not in self.configs:
            return self.load_config(config_name)
        return self.configs.get(config_name, {})
    
    def _process_env_vars(self, config):
        """
        Process environment variable substitutions in config.
        
        Args:
            config: Configuration object (dict, list, or primitive)
            
        Returns:
            The processed configuration with environment variables substituted
        """
        if isinstance(config, dict):
            return {k: self._process_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._process_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            # Extract environment variable name
            env_var = config[2:-1]
            # Get value from environment, or keep original string if not found
            return os.environ.get(env_var, config)
        else:
            return config