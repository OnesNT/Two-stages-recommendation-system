import yaml
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("/home/quang/Two-stages-recommendation-system/configs")

def load_config(file, config_path=DEFAULT_CONFIG_PATH):
    """
    Loads a YAML configuration sfile.

    Parameters:
    - file (str): The filename of the YAML config file.
    - config_path (Path): The directory path to the config files. Defaults to DEFAULT_CONFIG_PATH.

    Returns:
    - dict: Parsed YAML configuration as a dictionary.
    """
    file_path = config_path / file  
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return None
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file '{file_path}'.\n{e}")
        return None
    