import subprocess
import configparser
import shutil
from datetime import datetime


def update_ini(path, updates: dict):
    """
    Update an INI configuration file with the given updates.

    - path: path to the INI file
    - updates: dictionary of sections, each containing keys and values to update

    This reads the INI file, adds/updates sections and keys, then saves it.
    """
    config = configparser.ConfigParser()
    config.read(path)  # Load existing config file

    for section, options in updates.items():
        if not config.has_section(section):
            config.add_section(section)  # Create section if it doesn't exist
        for key, value in options.items():
            config.set(section, key, str(value))  # Set/overwrite key-value pairs

    # Write changes back to the file
    with open(path, 'w') as f:
        config.write(f)


def backup_ini(path):
    """
    Create a timestamped backup copy of the INI file.

    - path: path to the INI file to back up

    This helps keep old versions in case you want to revert.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copyfile(path, f"{path}.{timestamp}.bak")  # Copy with timestamp appended


def run_script_in_env(env_name, script_path):
    """
    Run a Python script inside a specific conda environment.

    - env_name: the conda environment name
    - script_path: path to the Python script to run

    Uses 'conda run' to execute the script inside the environment.
    """
    subprocess.run(["conda", "run", "-n", env_name, "python", script_path])
