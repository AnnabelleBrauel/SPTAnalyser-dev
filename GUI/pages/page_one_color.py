from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QComboBox

class OneColorPage(QWidget):
    def __init__(self, switch_callback):
        super().__init__()
        self.switch_callback = switch_callback
        self.layout = QVBoxLayout()
        self.dropdown_container = QVBoxLayout()

        self.add_dropdown()

        btn_add = QPushButton("➕ Add option")
        btn_add.clicked.connect(self.add_dropdown)

        btn_next = QPushButton("✅ Next")
        btn_next.clicked.connect(lambda: self.switch_callback("textfields"))

        btn_back = QPushButton("⬅️ Back")
        btn_back.clicked.connect(lambda: self.switch_callback("start"))

        self.layout.addLayout(self.dropdown_container)
        self.layout.addWidget(btn_add)
        self.layout.addWidget(btn_next)
        self.layout.addWidget(btn_back)
        self.layout.addStretch()
        self.setLayout(self.layout)

    def add_dropdown(self):
        combo = QComboBox()
        combo.addItems(["adapt folder structure", "write ROI macro", "get swift parameters"])
        self.dropdown_container.addWidget(combo)
