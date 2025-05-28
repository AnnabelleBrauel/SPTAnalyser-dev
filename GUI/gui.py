import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget
from pages.page_start import StartPage
from pages.page_one_color import OneColorPage
from pages.page_user_input import TextFieldPage
from pages.page_two_color import TwoColorPage
from gui_controller import GUIController


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SPTAnalyser GUI")

        self.controller = GUIController()
        self.stack = QStackedWidget()

        self.start_page = StartPage(self.switch_page)
        self.page_1color = OneColorPage(self.switch_page)
        self.page_textfields = TextFieldPage(self.switch_page)
        self.page_2color = TwoColorPage()

        self.stack.addWidget(self.start_page)       # Index 0
        self.stack.addWidget(self.page_1color)      # Index 1
        self.stack.addWidget(self.page_2color)      # Index 2
        self.stack.addWidget(self.page_textfields)  # Index 3

        self.setCentralWidget(self.stack)

    def switch_page(self, target):
        page_map = {
            "start": 0,
            "1color": 1,
            "2color": 2,
            "textfields": 3
        }
        self.stack.setCurrentIndex(page_map.get(target, 0))

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
