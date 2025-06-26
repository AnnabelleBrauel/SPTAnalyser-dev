from PyQt5.QtWidgets import QPushButton

def create_back_button(callback):
    """
    Create a QPushButton labeled 'Back' with the given callback on click.
    """
    btn = QPushButton("⬅️ Back")
    btn.clicked.connect(callback)
    return btn

def create_confirm_button(callback):
    """
    Create a QPushButton labeled 'Confirm' with the given callback on click.
    """
    btn = QPushButton("✅ Confirm")
    btn.clicked.connect(callback)
    return btn

def create_add_option_button(callback):
    """
    Create a QPushButton labeled 'Add Option' with the given callback on click.
    """
    btn = QPushButton("➕ Add Option")
    btn.clicked.connect(callback)
    return btn
