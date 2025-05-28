from PyQt5.QtWidgets import QPushButton

def create_back_button(callback):
    btn = QPushButton("⬅️ Back")
    btn.clicked.connect(callback)
    return btn

def create_confirm_button(callback):
    btn = QPushButton("✅ Confirm")
    btn.clicked.connect(callback)
    return btn

def create_add_option_button(callback):
    btn = QPushButton("➕ Add Option")
    btn.clicked.connect(callback)
    return btn
