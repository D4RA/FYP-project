from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QCursor
from PyQt5.QtCore import Qt

class ClickableLabel(QLabel):
    def __init__(self, text, callback):
        super().__init__(text)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setStyleSheet("color: blue; text-decoration: underline;")
        self.callback = callback

    def mousePressEvent(self, event):
        if callable(self.callback):
            self.callback(self.text())
