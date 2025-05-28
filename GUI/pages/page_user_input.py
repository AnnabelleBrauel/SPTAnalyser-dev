from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton

class TextFieldPage(QWidget):
    def __init__(self, switch_callback):
        super().__init__()
        self.switch_callback = switch_callback
        layout = QVBoxLayout()

        label_d = QLabel("paths to coverslides:")
        self.input_d = QLineEdit()

        label_e = QLabel("image size:")
        self.input_e = QLineEdit()

        btn_back = QPushButton("⬅️ Back")
        btn_back.clicked.connect(lambda: self.switch_callback("1color"))

        layout.addWidget(label_d)
        layout.addWidget(self.input_d)
        layout.addWidget(label_e)
        layout.addWidget(self.input_e)
        layout.addWidget(btn_back)
        layout.addStretch()

        self.setLayout(layout)
