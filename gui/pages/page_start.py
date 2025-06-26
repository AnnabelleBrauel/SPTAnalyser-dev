from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton

class StartPage(QWidget):
    def __init__(self, switch_callback):
        super().__init__()

        # Create a vertical layout
        layout = QVBoxLayout()

        # Label at the top
        label = QLabel("Chose mode:")
        layout.addWidget(label)

        # Buttons for choosing a mode
        btn_1color = QPushButton("1 Color")
        btn_2color = QPushButton("2 Color")

        # When a button is clicked, switch to the corresponding page
        btn_1color.clicked.connect(lambda: switch_callback("1color"))
        btn_2color.clicked.connect(lambda: switch_callback("2color"))

        # Add buttons to layout
        layout.addWidget(btn_1color)
        layout.addWidget(btn_2color)
        layout.addStretch()

        # Apply the layout to this widget
        self.setLayout(layout)
