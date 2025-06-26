from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton

class TextFieldPage(QWidget):
    def __init__(self, switch_callback):
        super().__init__()

        # Store the page switch function
        self.switch_callback = switch_callback

        # Set up main layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

    def showEvent(self, event):
        """
        Automatically called whenever the page becomes visible.
        We use it to refresh the input fields dynamically.
        """
        super().showEvent(event)
        self.build_inputs()  # rebuild fields every time the page is shown

    def build_inputs(self):
        """
        Dynamically create input fields based on what the selected scripts require.
        """
        # Remove existing widgets from the layout (reset view)
        for i in reversed(range(self.layout.count())):
            widget = self.layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        # Dictionary to store references to the input fields
        self.input_fields = {}

        # Ask the controller which inputs are required
        controller = self.window().controller
        required_inputs = controller.get_required_inputs()

        # Create a label + input box for each required input
        for key, label_text in required_inputs.items():
            label = QLabel(label_text)
            line_edit = QLineEdit()
            self.input_fields[key] = line_edit
            self.layout.addWidget(label)
            self.layout.addWidget(line_edit)

        # Run button: starts the pipeline
        btn_run = QPushButton("▶️ Run pipeline")
        btn_run.clicked.connect(self.on_run_clicked)

        # Back button: returns to 1-color page
        btn_back = QPushButton("⬅️ Back")
        btn_back.clicked.connect(lambda: self.switch_callback("1color"))

        self.layout.addWidget(btn_run)
        self.layout.addWidget(btn_back)
        self.layout.addStretch()

    def on_run_clicked(self):
        """
        Called when 'Run' is clicked. Collects all inputs and tells the controller to start the analysis.
        """
        # Read user input from all fields
        user_input = {key: field.text() for key, field in self.input_fields.items()}

        # Save the user input
        self.window().controller.save_user_input(user_input)

        # Tell the controller to run all the selected scripts
        self.window().controller.run_pipeline_batch()
