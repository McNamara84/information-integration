import argparse
import sys
from PyQt5 import QtCore, QtWidgets

from profiling import profile_dataframe

from load_bibliojobs import load_bibliojobs


class LoadWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(int)
    error = QtCore.pyqtSignal(str)

    def __init__(self, path: str) -> None:
        super().__init__()
        self._path = path

    @QtCore.pyqtSlot()
    def run(self):
        def callback(value: float) -> None:
            self.progress.emit(int(value))

        try:
            dataframe = load_bibliojobs(self._path, progress_callback=callback)
        except FileNotFoundError as exc:  # pragma: no cover
            self.error.emit(str(exc))
            return

        self.finished.emit(dataframe)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.setWindowTitle("Informationsintegration")
        self.resize(800, 600)

        self._status = self.statusBar()
        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        self._status.addPermanentWidget(self._progress)
        self._status.showMessage("CSV-Datei wird eingelesen...")

        self._button = QtWidgets.QPushButton("Data Profiling")
        self._button.setEnabled(False)
        self._button.clicked.connect(self._show_profile)
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.addWidget(self._button)
        layout.addStretch()
        self.setCentralWidget(container)

        self._worker = LoadWorker(path)
        self._thread = QtCore.QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.error.connect(self._on_error)
        self._worker.error.connect(self._thread.quit)
        self._worker.error.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    @QtCore.pyqtSlot(object)
    def _on_finished(self, df) -> None:
        self._status.showMessage("Einlesen abgeschlossen", 5000)
        self._progress.setValue(100)
        self._dataframe = df
        self._button.setEnabled(True)

    @QtCore.pyqtSlot(str)
    def _on_error(self, message: str) -> None:
        self._status.showMessage(message, 5000)
        self._progress.setValue(0)

    def _show_profile(self) -> None:
        stats = profile_dataframe(self._dataframe)
        window = ProfileWindow(stats, self)
        window.show()
        self._profile_window = window  # prevent garbage collection


class ProfileWindow(QtWidgets.QMainWindow):
    def __init__(self, stats, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Data Profiling")
        self.resize(800, 400)
        table = QtWidgets.QTableWidget(self)
        table.setAlternatingRowColors(True)
        table.setRowCount(len(stats))
        table.setColumnCount(len(stats.columns))
        table.setHorizontalHeaderLabels(stats.columns.tolist())
        for row_idx, (_, row) in enumerate(stats.iterrows()):
            for col_idx, value in enumerate(row):
                item = QtWidgets.QTableWidgetItem(str(value))
                table.setItem(row_idx, col_idx, item)
        table.resizeColumnsToContents()
        self.setCentralWidget(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Startet die Informationsintegration-GUI")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="bibliojobs_raw.csv",
        help="Pfad zur einzulesenden CSV-Datei",
    )
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow(args.csv_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
