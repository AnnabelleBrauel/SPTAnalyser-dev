from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QComboBox
from widgets.common_buttons import create_back_button, create_confirm_button, create_add_option_button

class OneColorPage(QWidget):
    def __init__(self, switch_callback):
        super().__init__()
        self.switch_callback = switch_callback
        self.layout = QVBoxLayout()
        self.dropdown_container = QVBoxLayout()

        self.add_dropdown()

        btn_add = create_add_option_button(self.add_dropdown)
        btn_next = create_confirm_button(lambda: self.switch_callback("textfields"))
        btn_back = create_back_button(lambda: self.switch_callback("start"))

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
