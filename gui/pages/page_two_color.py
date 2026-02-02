from PyQt5.QtWidgets import QLabel

class TwoColorPage(QLabel):
    def __init__(self, switch_page_callback):
        super().__init__("2 Color not available yet...")
        self.switch_page = switch_page_callback
