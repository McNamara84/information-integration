import sys
from PyQt5 import QtCore, QtWidgets

from load_bibliojobs import load_bibliojobs


class LoadWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(int)

    @QtCore.pyqtSlot()
    def run(self):
        def callback(value: float) -> None:
            self.progress.emit(int(value))

        dataframe = load_bibliojobs(progress_callback=callback)
        self.finished.emit(dataframe)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Informationsintegration")
        self.resize(800, 600)

        self._status = self.statusBar()
        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        self._status.addPermanentWidget(self._progress)
        self._status.showMessage("CSV-Datei wird eingelesen...")

        self._worker = LoadWorker()
        self._thread = QtCore.QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    @QtCore.pyqtSlot(object)
    def _on_finished(self, df) -> None:
        self._status.showMessage("Einlesen abgeschlossen", 5000)
        self._progress.setValue(100)
        self.dataframe = df


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
