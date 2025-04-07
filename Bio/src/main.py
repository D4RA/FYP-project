import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont

from home import HomeScreen
from TSP_app import TSPApp
from about import AboutPage

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 11))

    main_window = TSPApp()

    def launch_main_window():
        home_window.close()
        main_window.show()

    home_window = HomeScreen(on_start_callback=launch_main_window)
    home_window.show()


    def show_about_page(self):
        self.about_window = AboutPage(self.show_main_app, self.geometry(), self.isMaximized())
        self.about_window.show()
        self.hide()


    sys.exit(app.exec())
