class GUIController:
    def __init__(self):
        self.state = {}  # placeholder for later addition of logic

    def save_user_input(self, data):
        self.state["user_input"] = data

    def get_user_input(self):
        return self.state.get("user_input", {})
