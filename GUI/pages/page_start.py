from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton

class StartPage(QWidget):
    def __init__(self, switch_callback):
        super().__init__()
        layout = QVBoxLayout()

        label = QLabel("Chose mode:")
        layout.addWidget(label)

        btn_1color = QPushButton("1 Color")
        btn_2color = QPushButton("2 Color")

        btn_1color.clicked.connect(lambda: switch_callback("1color"))
        btn_2color.clicked.connect(lambda: switch_callback("2color"))

        layout.addWidget(btn_1color)
        layout.addWidget(btn_2color)
        layout.addStretch()

        self.setLayout(layout)
