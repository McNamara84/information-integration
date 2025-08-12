from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from PyQt5 import QtCore, QtWidgets

from profiling import profile_dataframe, get_all_error_types

from load_bibliojobs import load_bibliojobs
from cleaning import clean_dataframe


ERROR_TYPES = [
    "Unzulässige Werte",
    "Verletzte Attributabhängigkeiten",
    "Eindeutigkeitsverletzungen",
    "Verletzte referenzielle Integrität",
    "Fehlende Werte",
    "Schreibfehler",
    "Falsche Werte",
    "Falsche Referenzen",
    "Kryptische Werte",
    "Eingebettete Werte",
    "Falsche Zuordnungen",
    "Widersprüchliche Werte",
    "Transpositionen",
    "Duplikate",
    "Datenkonflikte",
]


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


class CleanWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(int)

    def __init__(self, dataframe) -> None:
        super().__init__()
        self._dataframe = dataframe

    @QtCore.pyqtSlot()
    def run(self):
        def callback(value: float) -> None:
            self.progress.emit(int(value))

        cleaned = clean_dataframe(self._dataframe, progress_callback=callback)
        self.finished.emit(cleaned)


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

        self._clean_button = QtWidgets.QPushButton("Datensätze bereinigen")
        self._clean_button.setEnabled(False)
        self._clean_button.clicked.connect(self._clean_data)

        self._export_cleaned_button = QtWidgets.QPushButton(
            "Ergebnis als Exceltabelle speichern"
        )
        self._export_cleaned_button.hide()
        self._export_cleaned_button.clicked.connect(self._export_cleaned)

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.addWidget(self._button)
        layout.addWidget(self._clean_button)
        layout.addStretch()
        layout.addWidget(self._export_cleaned_button)
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
        self._profile_window: ProfileWindow | None = None

    @QtCore.pyqtSlot(object)
    def _on_finished(self, df) -> None:
        self._status.showMessage("Einlesen abgeschlossen", 5000)
        self._progress.setValue(100)
        self._dataframe = df
        self._button.setEnabled(True)
        self._clean_button.setEnabled(True)

    @QtCore.pyqtSlot(str)
    def _on_error(self, message: str) -> None:
        self._status.showMessage(message, 5000)
        self._progress.setValue(0)

    def _show_profile(self) -> None:
        if self._profile_window is not None:
            self._profile_window.close()
            self._profile_window = None
        stats = profile_dataframe(self._dataframe)
        window = ProfileWindow(stats, self._dataframe, self)
        window.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        window.closed.connect(self._on_profile_window_destroyed)
        window.show()
        self._profile_window = window

    def _on_profile_window_destroyed(self) -> None:
        self._profile_window = None

    def _clean_data(self) -> None:
        self._status.showMessage("Datensätze werden bereinigt...")
        self._progress.setValue(0)
        self._clean_button.setEnabled(False)

        self._clean_worker = CleanWorker(self._dataframe)
        self._clean_thread = QtCore.QThread(self)
        self._clean_worker.moveToThread(self._clean_thread)
        self._clean_thread.started.connect(self._clean_worker.run)
        self._clean_worker.progress.connect(self._progress.setValue)
        self._clean_worker.finished.connect(self._on_cleaned)
        self._clean_worker.finished.connect(self._clean_thread.quit)
        self._clean_worker.finished.connect(self._clean_worker.deleteLater)
        self._clean_thread.finished.connect(self._clean_thread.deleteLater)
        self._clean_thread.start()

    @QtCore.pyqtSlot(object)
    def _on_cleaned(self, df) -> None:
        self._dataframe = df
        self._status.showMessage("Bereinigung abgeschlossen", 5000)
        self._progress.setValue(100)
        self._export_cleaned_button.show()

    def _export_cleaned(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Bereinigte Daten exportieren", filter="Excel Dateien (*.xlsx)"
        )
        if not path:
            return
        self._dataframe.to_excel(path, index=False)
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            QtWidgets.QMessageBox.information(
                self,
                "Export erfolgreich",
                f"Daten wurden erfolgreich exportiert nach:\n{path}",
            )


class ProfileWindow(QtWidgets.QMainWindow):
    closed = QtCore.pyqtSignal()

    def __init__(self, stats, dataframe, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Data Profiling")
        self._stats = stats
        self._dataframe = dataframe

        container = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

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
        layout.addWidget(table)
        self._table = table

        self._export_button = QtWidgets.QPushButton("Bericht exportieren", self)
        self._export_button.clicked.connect(self._export_report)
        layout.addWidget(self._export_button)

        self.setCentralWidget(container)

        total_width = table.verticalHeader().width() + table.frameWidth() * 2
        total_width += table.verticalScrollBar().sizeHint().width()
        for i in range(table.columnCount()):
            total_width += table.columnWidth(i)
        screen = QtWidgets.QApplication.primaryScreen()
        screen_width = screen.availableGeometry().width() if screen else total_width
        self.resize(min(total_width, screen_width), 400)

    def _export_report(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Bericht exportieren", filter="Excel Dateien (*.xlsx)"
        )
        if not path:
            return
        
        # Create the report according to requirements:
        # 1. Spalte: untersuchtes Attribut
        # 2. Spalte: Fehlertyp gemäss der Fehlerklassifikation von Nauman/Leser  
        # 3. Spalte: relative Fehlerquote
        rows = []
        
        # For each column in the dataframe, get ALL error types
        for column in self._dataframe.columns:
            series = self._dataframe[column]
            all_errors = get_all_error_types(series, column)
            
            if all_errors:
                # Add one row for each error type found
                for error_type, error_rate in all_errors:
                    rows.append({
                        "Attribut": column,
                        "Fehlertyp": error_type,
                        "Relative Fehlerquote (%)": round(error_rate, 2),
                    })
            else:
                # If no errors found, add a row indicating this
                rows.append({
                    "Attribut": column,
                    "Fehlertyp": "Keine signifikanten Fehler",
                    "Relative Fehlerquote (%)": 0.0,
                })
        
        # Sort by attribute name, then by error rate (descending)
        rows.sort(key=lambda x: (x["Attribut"], -x["Relative Fehlerquote (%)"]))
        
        report_df = pd.DataFrame(rows)
        report_df.to_excel(path, index=False)
        
        # Show success message only when a display is available
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            QtWidgets.QMessageBox.information(
                self,
                "Export erfolgreich",
                f"Bericht wurde erfolgreich exportiert nach:\n{path}\n\n"
                f"Anzahl Zeilen im Bericht: {len(report_df)}"
            )

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)


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