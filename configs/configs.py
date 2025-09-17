# Databricks notebook source
# MAGIC %md
# MAGIC # Configuration Management Module
# MAGIC
# MAGIC This module provides centralized configuration management for the project.
# MAGIC It automatically detects the project structure and loads configuration files.
# MAGIC
# MAGIC ## Usage from Notebooks:
# MAGIC ```python
# MAGIC # Run this config module
# MAGIC %run ../configs/configs
# MAGIC
# MAGIC # Load specific config files and set globals
# MAGIC extract_column_transformation_lists("/config_files/training.yaml")
# MAGIC
# MAGIC # Or load config as dictionary without setting globals
# MAGIC config = load_config_file("/config_files/training.yaml")
# MAGIC ```
# MAGIC
# MAGIC ## Available Config Files:
# MAGIC - /config_files/configs.yaml - Main configuration
# MAGIC - /config_files/data_preprocessing.yaml - Data preprocessing settings
# MAGIC - /config_files/feature_engineering.yaml - Feature engineering settings
# MAGIC - /config_files/training.yaml - Model training configuration
# MAGIC - /config_files/scoring.yaml - Scoring configuration
# MAGIC - /config_files/serving.yaml - Model serving configuration

# COMMAND ----------

import yaml
import os
from pathlib import Path
import re

def load_config(config_path):
    """Loads a YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def functions_path(file_path):
    """
    Helper function to convert notebook paths to Workspace paths for function imports.
    Removes the last two path components and prepends /Workspace.
    
    Example: /path/to/project/notebooks/notebook.py -> /Workspace/path/to/project
    """
    # Pattern to find the last two path components (e.g., /folder/file.yaml)
    minus_pattern = r"(\/)\w+(\/)\w+$" 
    match = re.search(minus_pattern, file_path)
    if match:
        minus_str = match.group(0)
        base_path = file_path[:-len(minus_str)] if file_path.endswith(minus_str) else file_path.replace(minus_str, '')
        function_str = f"/Workspace{base_path}"
    else:
        print(f"Warning: functions_path pattern did not match for {file_path}")
        function_str = f"/Workspace{file_path}"
    return function_str

# Environment detection and configuration
def get_current_environment():
    """
    Detects the current environment based on Databricks workspace URL or ID.
    Returns the environment name (dsexp, modelbuild, modelpromotion, modeldeployment)
    """
    try:
        # Try to get workspace URL first
        workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
        
        # Load the main config to get workspace mappings
        config_path = get_config_file_path("/config_files/configs.yaml")
        config = load_config(config_path)
        
        # Get environment from workspace URL
        env = config.get('workspaces', {}).get(workspace_url)
        if env:
            print(f"Environment detected from workspace URL: {env}")
            return env
            
        # Fallback: try workspace ID
        workspace_id = spark.conf.get("spark.databricks.workspaceId")
        # You can add workspace ID mapping here if needed
        
        # Default to dsexp if can't determine
        print("Warning: Could not determine environment, defaulting to 'dsexp'")
        return 'dsexp'
        
    except Exception as e:
        print(f"Error detecting environment: {e}")
        return 'dsexp'

def get_environment_config(env_name=None):
    """
    Gets the configuration for the specified environment.
    If no environment is specified, detects the current environment.
    """
    if env_name is None:
        env_name = get_current_environment()
    
    # Load the main config
    config_path = get_config_file_path("/config_files/configs.yaml")
    config = load_config(config_path)
    
    environments = config.get('environments', {})
    if env_name not in environments:
        raise ValueError(f"Environment '{env_name}' not found in configuration")
    
    return environments[env_name]

def get_table_path(schema, table, catalog_type='auxiliary'):
    """
    Constructs a fully qualified table path based on the current environment.
    
    Parameters:
    schema: The schema name (e.g., 'single_vehicle_incident_checks')
    table: The table name (e.g., 'claim_referral_log')
    catalog_type: Either 'auxiliary' (for reading), 'mlstore' (for writing), or 'adp' (for ADP certified data)
    
    Returns:
    Fully qualified table path (e.g., 'prod_dsexp_auxiliarydata.single_vehicle_incident_checks.claim_referral_log')
    """
    env_config = get_environment_config()
    
    if catalog_type == 'auxiliary':
        catalog = env_config['auxiliary_catalog']
    elif catalog_type == 'mlstore':
        catalog = env_config['mlstore_catalog']
    elif catalog_type == 'adp':
        catalog = env_config['adp_catalog']
    else:
        raise ValueError(f"Invalid catalog_type: {catalog_type}. Must be 'auxiliary', 'mlstore', or 'adp'")
    
    return f"{catalog}.{schema}.{table}"

# Get the current working directory and project root
try:
    current_notebook_path = Path(os.getcwd())
    
    # Find project root by looking for common project markers
    # This makes the config discovery more robust
    project_root = None
    search_path = current_notebook_path
    
    # Search upward for project root indicators
    while search_path.parent != search_path:
        if (search_path / "configs").exists() and (search_path / "notebooks").exists():
            project_root = search_path
            break
        if (search_path / "bundle.yaml").exists() or (search_path / "requirements.txt").exists():
            project_root = search_path
            break
        search_path = search_path.parent
    
    if project_root is None:
        # Fallback: assume we're in notebooks/ and project root is parent
        if current_notebook_path.name == "notebooks" or "notebooks" in current_notebook_path.parts:
            idx = list(current_notebook_path.parts).index("notebooks") if "notebooks" in current_notebook_path.parts else -1
            if idx >= 0:
                project_root = Path(*current_notebook_path.parts[:idx])
            else:
                project_root = current_notebook_path.parent
        else:
            project_root = current_notebook_path
            
    print(f"Project root detected: {project_root}")
    print(f"Current location: {current_notebook_path}")
    
except Exception as e:
    print(f"Error detecting project structure: {e}")
    current_notebook_path = Path(".")
    project_root = Path(".")

# --- Configuration File Path Setup ---
# This logic assumes a directory structure like:
# project_root/
#  ├── configs/
#  │   └── config_files/
#  │       ├── global_configs.yaml
#  │       ├── configs.yaml (data_prep)
#  │       ├── feature_engineering.yaml
#  │       ├── training.yaml
#  │       ├── scoring.yaml
#  │       ├── serving.yaml
#  │       └── scoring_new_pcd.yaml
#  └── notebooks/  <- (where the notebook running this script is)

def get_config_file_path(relative_config_path):
    """
    Constructs the absolute path to a config file using project root.
    
    Parameters:
    relative_config_path: Path relative to configs directory (e.g., "/config_files/training.yaml")
    
    Returns:
    Absolute path to the config file
    """
    # Use the detected project_root
    config_base_path = project_root / "configs"
    
    # Handle relative path that may start with /
    if relative_config_path.startswith("/"):
        effective_relative_path = relative_config_path[1:]
    else:
        effective_relative_path = relative_config_path
    
    full_path = config_base_path / effective_relative_path
    
    # Check if file exists and provide helpful error if not
    if not full_path.exists():
        print(f"Warning: Config file not found at {full_path}")
        # Try alternative locations
        alt_path = current_notebook_path / effective_relative_path
        if alt_path.exists():
            print(f"Found config at alternative location: {alt_path}")
            return str(alt_path)
    
    return str(full_path)

# Define relative paths for config files
CONFIG_REL_PATH = "/config_files/configs.yaml"
DATA_PREP_CONFIG_REL_PATH = "/config_files/configs.yaml"
FEATURE_CONFIG_REL_PATH = "/config_files/feature_engineering.yaml"
TRAINING_CONFIG_REL_PATH = "/config_files/training.yaml"
SCORING_CONFIG_REL_PATH = "/config_files/scoring.yaml"
SERVING_CONFIG_REL_PATH = "/config_files/serving.yaml"
NEW_SCORING_CONFIG_REL_PATH = "/config_files/scoring_new_pcd.yaml"

# Construct full paths
config_path = get_config_file_path(CONFIG_REL_PATH)
data_prep_config_path = get_config_file_path(DATA_PREP_CONFIG_REL_PATH)
feature_config_path = get_config_file_path(FEATURE_CONFIG_REL_PATH)
training_config_path = get_config_file_path(TRAINING_CONFIG_REL_PATH)
scoring_config_path = get_config_file_path(SCORING_CONFIG_REL_PATH)
serving_config_path = get_config_file_path(SERVING_CONFIG_REL_PATH)
new_scoring_config_path = get_config_file_path(NEW_SCORING_CONFIG_REL_PATH)

# --- Main Logic ---
def set_globals_from_config(config, parent_key=''):
    """
    Dynamically sets global variables for each node in the configuration dictionary.
    The variable names are created by concatenating the keys leading to each value.
    MODIFIED: This version now sets a global variable for the dictionary itself
    before recursing into it.
    """
    for key, value in config.items():
        # Construct the variable name based on the key hierarchy
        # Sanitize key to ensure it's a valid Python variable name if necessary
        # For simplicity, assuming keys are already valid or simple strings.
        # If keys can have spaces or special chars, they would need sanitization.
        safe_key = re.sub(r'\W|^(?=\d)', '_', key) # Basic sanitization: replace non-alphanum with _, prefix numbers with _

        var_name = f"{parent_key}_{safe_key}" if parent_key else safe_key
        
        # Set the global variable for the current key-value pair.
        # If 'value' is a dictionary (e.g., rename_map), 
        # this line will make 'rename_map' (or 'parent_rename_map') a global dict.
        globals()[var_name] = value
        
        # If the value is a dictionary, also recurse to set its children as globals,
        # prefixed with the current var_name.
        if isinstance(value, dict):
            set_globals_from_config(value, var_name)
        # No 'else' needed here because globals()[var_name] = value handles leaf nodes too.

def extract_column_transformation_lists(relative_config_file_path):
    """
    Loads a specific config file and sets global variables from its contents.
    
    Parameters:
    relative_config_file_path: Path relative to configs directory (e.g., "/config_files/training.yaml")
    """
    # Construct the full path to the config file using the improved path resolution
    CONFIG_PATH = get_config_file_path(relative_config_file_path)
    
    print(f"Loading configuration from: {CONFIG_PATH}")
    try:
        config = load_config(CONFIG_PATH)
        set_globals_from_config(config)
        print(f"Successfully loaded and set globals from {CONFIG_PATH}")
        return config  # Return config for potential direct use
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {CONFIG_PATH}")
        return None
    except Exception as e:
        print(f"Error processing configuration file {CONFIG_PATH}: {e}")
        return None

# Add convenience function for loading configs without setting globals
def load_config_file(relative_config_file_path):
    """
    Loads a config file without setting global variables.
    Useful when you want to work with config as a dictionary.
    
    Parameters:
    relative_config_file_path: Path relative to configs directory
    
    Returns:
    Dictionary containing the configuration, or None if error
    """
    CONFIG_PATH = get_config_file_path(relative_config_file_path)
    try:
        return load_config(CONFIG_PATH)
    except Exception as e:
        print(f"Error loading config file {CONFIG_PATH}: {e}")
        return None

# COMMAND ----------

# Additional utility functions for notebook integration

def list_available_configs():
    """Lists all available configuration files in the project."""
    config_dir = project_root / "configs" / "config_files"
    if config_dir.exists():
        yaml_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))
        print("Available configuration files:")
        for f in sorted(yaml_files):
            print(f"  - /config_files/{f.name}")
    else:
        print(f"Config directory not found at {config_dir}")

def get_config_value(config_key, default=None):
    """
    Get a specific configuration value from the global namespace.
    
    Parameters:
    config_key: The key to look up (e.g., 'model_name', 'catalog_prefix')
    default: Default value if key not found
    
    Returns:
    The configuration value or default
    """
    return globals().get(config_key, default)

def reload_all_configs():
    """Reload all standard configuration files."""
    configs_to_load = [
        CONFIG_REL_PATH,
        FEATURE_CONFIG_REL_PATH,
        TRAINING_CONFIG_REL_PATH,
        SCORING_CONFIG_REL_PATH,
        SERVING_CONFIG_REL_PATH
    ]
    
    for config in configs_to_load:
        print(f"\nLoading {config}...")
        extract_column_transformation_lists(config)

# COMMAND ----------

# Additional utility functions for dynamic configuration
def get_mlflow_experiment_name(base_name):
    """
    Gets the MLflow experiment name for the current environment.
    """
    env_config = get_environment_config()
    env_name = get_current_environment()
    return f"/Shared/{env_name}/{base_name}"

def get_model_registry_path(model_name):
    """
    Gets the model registry path for the current environment.
    """
    env_config = get_environment_config()
    catalog = env_config['mlstore_catalog']
    return f"{catalog}.models.{model_name}"

# Display available configs and project structure on import
print("=" * 60)
print("Configuration Module Loaded Successfully")
print("=" * 60)
print(f"\nProject Root: {project_root}")
print(f"Current Location: {current_notebook_path}")
print(f"\nMain Config Path: {config_path}")

# Try to detect current environment
try:
    current_env = get_current_environment()
    env_config = get_environment_config()
    print(f"\nCurrent Environment: {current_env}")
    print(f"MLStore Catalog: {env_config['mlstore_catalog']}")
    print(f"Auxiliary Catalog: {env_config['auxiliary_catalog']}")
    print(f"ADP Catalog: {env_config['adp_catalog']}")
except:
    print("\nNote: Environment detection requires Spark context")

print("\nAvailable functions:")
print("  - get_current_environment() - Detect current environment")
print("  - get_table_path(schema, table, catalog_type) - Get fully qualified table path")
print("  - extract_column_transformation_lists() - Load specific configs")
print("  - load_config_file() - Load config as dictionary")
print("  - list_available_configs() - See all config files")
print("=" * 60)
