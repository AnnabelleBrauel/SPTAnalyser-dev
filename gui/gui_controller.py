import os
import shutil
from tools.execution_utils import update_ini, backup_ini, run_script_in_env

# Define all scripts metadata: envs, paths, inputs
SCRIPT_DEFS = {
    "adapt folder structure": {
        "id": "adapt_folder_structure",
        "env": "SPTAnalyser", # TODO: environments anlegen
        "script_path": "Scripts/adapt_folder_structure.py",
        "ini_path": "Scripts/adapt_folder_structure_config.ini",
        "required_inputs": {
            "dir01": "Path to coverslip directory",
            "remove_str": "Remove string (e.g. _MMStack_Pos0.ome.tif)",
            "output_dir": "Output directory for copied INI and results"
        }
    },
    "write ROI macro": {
        "id": "script2",
        "env": "env_script2",
        "script_path": "scripts/script2.py",
        "ini_path": "configs/config2.ini",
        "required_inputs": {
            "roi_path": "Path to ROI directory"
        }
    },
    "get swift parameters": {
        "id": "script3",
        "env": "env_script3",
        "script_path": "scripts/script3.py",
        "ini_path": "configs/config3.ini",
        "required_inputs": {
            "image_size": "Image size (e.g. 512x512)"
        }
    }
}


class GUIController:

    def __init__(self):
        # Store internal app state here (e.g. user inputs, selected scripts)
        self.state = {}

    def save_user_input(self, data):
        """
        Save the user input data from the GUI.

        - data: dict with keys and user-entered values
        """
        self.state["user_input"] = data

    def get_user_input(self):
        return self.state.get("user_input", {})

    def run_pipeline(self, script_id: str):
        """
        Run a specific script in its respective conda environment, using user input.

        This is a simplified example — you'd want to generalize it.
        """
        # Select paths and envs based on script ID
        if script_id == "script1":
            ini_path = "configs/config1.ini"
            env = "env_script1"
            script_path = "scripts/script1.py"
        elif script_id == "script2":
            ini_path = "configs/config2.ini"
            env = "env_script2"
            script_path = "scripts/script2.py"
        else:
            # Script not recognized, do nothing
            return

        # Update INI file with user input under section "UserInput"
        update_ini(ini_path, {"UserInput": self.get_user_input()})

        # Backup the INI file before running the script
        backup_ini(ini_path)

        # Run the script in the specified conda environment
        run_script_in_env(env, script_path)

    def get_user_input(self):
        """
        Retrieve the saved user input data.
        """
        return self.state.get("user_input", {})
