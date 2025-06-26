from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QComboBox
from widgets.common_buttons import create_back_button, create_confirm_button, create_add_option_button

class OneColorPage(QWidget):
    def __init__(self, switch_callback):
        super().__init__()

        # Function to switch to another page (like "start", "textfields", etc.)
        self.switch_callback = switch_callback

        # Main vertical layout for this page
        self.layout = QVBoxLayout()

        # Sub-layout that holds all the dropdowns
        self.dropdown_container = QVBoxLayout()

        # List to keep references to the dropdowns (QComboBox widgets)
        self.dropdowns = []

        # Add the first dropdown when the page is created
        self.add_dropdown()

        # Create buttons
        btn_add = create_add_option_button(self.add_dropdown)  # Add new dropdown
        btn_next = create_confirm_button(self.on_confirm_clicked)  # Proceed to next page
        btn_back = create_back_button(lambda: self.switch_callback("start"))  # Go back to start page

        # Add all UI elements to the main layout
        self.layout.addLayout(self.dropdown_container)
        self.layout.addWidget(btn_add)
        self.layout.addWidget(btn_next)
        self.layout.addWidget(btn_back)
        self.layout.addStretch()  # Adds flexible space at the bottom

        # Apply the layout to the widget
        self.setLayout(self.layout)

    def add_dropdown(self):
        """Adds a new dropdown (QComboBox) with script options."""
        combo = QComboBox()
        combo.addItems(["adapt folder structure", "write ROI macro", "get swift parameters"])
        self.dropdowns.append(combo)
        self.dropdown_container.addWidget(combo)

    def on_confirm_clicked(self):
        """Called when the 'Next' button is clicked."""
        # Get all selected script names from the dropdowns
        selected_scripts = [dropdown.currentText() for dropdown in self.dropdowns]

        # Store them in the controller (shared logic object)
        self.window().controller.save_selected_scripts(selected_scripts)

        # Go to the next page
        self.switch_callback("textfields")
