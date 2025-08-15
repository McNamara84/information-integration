from __future__ import annotations

import argparse
import os
import sys
from typing import cast, TypeVar

import pandas as pd
from PyQt6 import QtCore, QtWidgets, QtGui

ICON_PATH = os.path.join(os.path.dirname(__file__), "assets", "fhp_logo.svg")
# Will be initialized after QApplication creation in main()
APP_ICON: QtGui.QIcon | None = None

from profiling import profile_dataframe, get_all_error_types

from load_bibliojobs import load_bibliojobs
from cleaning import (
    clean_dataframe,
    find_fuzzy_duplicates,
    DEDUPLICATE_COLUMNS,
    prepare_duplicates_export,
)


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


T = TypeVar("T")


def _require(value: T | None, name: str) -> T:
    """Return *value* if it is not ``None`` or raise ``RuntimeError``."""
    if value is None:
        raise RuntimeError(f"{name} is unexpectedly None")
    return value


def apply_modern_style(app: QtWidgets.QApplication) -> None:
    """Apply a Windows 11 inspired style using Qt 6.9 features."""
    app.setStyle("Fusion")
    app.setFont(QtGui.QFont("Segoe UI", 10))

    palette = app.palette()
    accent = palette.color(QtGui.QPalette.ColorRole.Accent)
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, accent)
    palette.setColor(QtGui.QPalette.ColorRole.Button, accent)
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("white"))
    app.setPalette(palette)

    accent_name = accent.name()
    app.setStyleSheet(
        f"""
        QPushButton {{
            background-color: {accent_name};
            color: white;
            border-radius: 6px;
            padding: 6px 12px;
        }}
        QPushButton:disabled {{
            background-color: palette(button);
            color: palette(button-text);
        }}
        QProgressBar {{
            text-align: center;
        }}
        QProgressBar::chunk {{
            background-color: {accent_name};
            border-radius: 3px;
        }}
        """
    )


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
    status = QtCore.pyqtSignal(str)

    def __init__(self, dataframe) -> None:
        super().__init__()
        self._dataframe = dataframe

    @QtCore.pyqtSlot()
    def run(self):
        def callback(value: float) -> None:
            self.progress.emit(int(value))

        def status_callback(message: str) -> None:
            self.status.emit(message)

        cleaned = clean_dataframe(
            self._dataframe,
            progress_callback=callback,
            status_callback=status_callback,
        )
        self.finished.emit(cleaned)


class DedupeWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(int)

    def __init__(self, dataframe) -> None:
        super().__init__()
        self._dataframe = dataframe

    @QtCore.pyqtSlot()
    def run(self):
        def callback(value: float) -> None:
            self.progress.emit(int(value))

        _, duplicates = find_fuzzy_duplicates(
            self._dataframe,
            DEDUPLICATE_COLUMNS,
            progress_callback=callback,
        )
        self.finished.emit(duplicates)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.setWindowTitle("Informationsintegration")
        if APP_ICON is not None:
            self.setWindowIcon(APP_ICON)
        self.resize(800, 600)

        # Create an explicit status bar so that subsequent attribute access is
        # always safe. ``QMainWindow.statusBar`` can technically return ``None``
        # which confuses static type checkers.
        self._status: QtWidgets.QStatusBar = QtWidgets.QStatusBar(self)
        self.setStatusBar(self._status)
        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._status.addPermanentWidget(self._progress)
        self._status.showMessage("CSV-Datei wird eingelesen...")

        self._button = QtWidgets.QPushButton("Data Profiling")
        self._button.setEnabled(False)
        self._button.clicked.connect(self._show_profile)

        self._clean_button = QtWidgets.QPushButton("Datensätze bereinigen")
        self._clean_button.setEnabled(False)
        self._clean_button.clicked.connect(self._clean_data)

        self._dedupe_button = QtWidgets.QPushButton("Dubletten finden")
        self._dedupe_button.setEnabled(False)
        self._dedupe_button.clicked.connect(self._remove_duplicates)

        self._export_cleaned_button = QtWidgets.QPushButton(
            "Ergebnis als Exceltabelle speichern"
        )
        self._export_cleaned_button.hide()
        self._export_cleaned_button.clicked.connect(self._export_cleaned)

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.addWidget(self._button)
        layout.addWidget(self._clean_button)
        layout.addWidget(self._dedupe_button)
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
        self._clean_worker: CleanWorker | None = None
        self._clean_thread: QtCore.QThread | None = None
        self._dedupe_worker: DedupeWorker | None = None
        self._dedupe_thread: QtCore.QThread | None = None

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
        # Use the ``WidgetAttribute`` enum explicitly to satisfy static type
        # checkers that may not know about the ``WA_DeleteOnClose`` attribute on
        # ``Qt``.
        window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
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
        self._clean_worker.status.connect(self._status.showMessage)
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
        self._dedupe_button.setEnabled(True)
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


    def _remove_duplicates(self) -> None:
        if self._dedupe_thread and self._dedupe_thread.isRunning():
            self._status.showMessage("Dublettenprüfung läuft bereits", 5000)
            return

        self._status.showMessage("Suche nach Dubletten...")
        self._progress.setValue(0)
        self._dedupe_button.setEnabled(False)

        self._dedupe_worker = DedupeWorker(self._dataframe)
        self._dedupe_thread = QtCore.QThread(self)
        self._dedupe_worker.moveToThread(self._dedupe_thread)
        self._dedupe_thread.started.connect(self._dedupe_worker.run)
        self._dedupe_worker.progress.connect(self._progress.setValue)
        self._dedupe_worker.finished.connect(self._on_duplicates_found)
        self._dedupe_worker.finished.connect(self._dedupe_thread.quit)
        self._dedupe_worker.finished.connect(self._dedupe_worker.deleteLater)
        self._dedupe_thread.finished.connect(self._dedupe_thread.deleteLater)
        self._dedupe_thread.start()

    @QtCore.pyqtSlot(object)
    def _on_duplicates_found(self, duplicates) -> None:
        if not duplicates.empty:
            window = DuplicatesWindow(duplicates, self)
            window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
            window.remove_requested.connect(self._apply_duplicate_removal)
            window.show()
        else:
            if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
                QtWidgets.QMessageBox.information(
                    self,
                    "Keine Dubletten",
                    "Es wurden keine Dubletten gefunden.",
                )
        self._status.showMessage("Dublettenprüfung abgeschlossen", 5000)
        self._dedupe_button.setEnabled(True)
        self._dedupe_worker = None
        self._dedupe_thread = None

    @QtCore.pyqtSlot(list)
    def _apply_duplicate_removal(self, indices: list[int]) -> None:
        if indices:
            self._dataframe = self._dataframe.drop(index=indices).reset_index(drop=True)
            self._status.showMessage(f"{len(indices)} Dubletten entfernt", 5000)

class ProfileWindow(QtWidgets.QMainWindow):
    closed = QtCore.pyqtSignal()

    def __init__(self, stats, dataframe, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Data Profiling")
        if APP_ICON is not None:
            self.setWindowIcon(APP_ICON)
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
        for row_idx, row in enumerate(stats.itertuples(index=False, name=None)):
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

        # ``verticalHeader`` and ``verticalScrollBar`` are guaranteed to return
        # valid objects at runtime but are typed as optional, so we resolve them
        # through a helper to enforce non-``None`` values.
        header = _require(table.verticalHeader(), "verticalHeader")
        total_width = header.width() + table.frameWidth() * 2
        v_scroll = _require(table.verticalScrollBar(), "verticalScrollBar")
        total_width += v_scroll.sizeHint().width()
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

    def closeEvent(self, a0: QtGui.QCloseEvent | None) -> None:
        # ``pyqtSignal`` instances are descriptors; when accessed through an
        # instance they return ``pyqtBoundSignal`` which provides ``emit``.  Cast
        # accordingly so that type checkers understand the attribute.
        cast(QtCore.pyqtBoundSignal, self.closed).emit()
        super().closeEvent(a0)


class DuplicatesWindow(QtWidgets.QMainWindow):
    remove_requested = QtCore.pyqtSignal(list)

    def __init__(self, dataframe, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Gefundene Dubletten")
        if APP_ICON is not None:
            self.setWindowIcon(APP_ICON)
        self._dataframe = (
            dataframe[dataframe["probability"] == 100]
            .drop(columns=["probability"])
            .reset_index(drop=True)
        )

        container = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        select_all_layout = QtWidgets.QHBoxLayout()
        self._select_all = QtWidgets.QCheckBox("Alle auswählen", self)
        self._select_all.stateChanged.connect(self._on_select_all)
        select_all_layout.addWidget(self._select_all)
        self._selected_count_label = QtWidgets.QLabel("(0)", self)
        select_all_layout.addWidget(self._selected_count_label)
        select_all_layout.addStretch()
        layout.addLayout(select_all_layout)

        display_cols = [
            col
            for col in self._dataframe.columns
            if col not in {"pair_id", "keep", "orig_index"}
        ]
        table = QtWidgets.QTableWidget(self)
        table.setAlternatingRowColors(True)
        table.setRowCount(len(self._dataframe))
        table.setColumnCount(len(display_cols) + 1)
        table.setHorizontalHeaderLabels(["Auswählen"] + display_cols)

        self._checkboxes: list[QtWidgets.QCheckBox] = []
        self._checkbox_map: dict[QtWidgets.QCheckBox, int] = {}
        self._selected_count = 0

        for row_idx, row in enumerate(self._dataframe.itertuples(index=False)):
            for col_idx, col in enumerate(display_cols):
                value = getattr(row, col)
                item = QtWidgets.QTableWidgetItem(str(value))
                table.setItem(row_idx, col_idx + 1, item)
            if not getattr(row, "keep", True):
                checkbox = QtWidgets.QCheckBox()
                checkbox.setChecked(True)
                self._selected_count += 1
                checkbox.stateChanged.connect(self._on_checkbox_state_changed)
                table.setCellWidget(row_idx, 0, checkbox)
                self._checkboxes.append(checkbox)
                try:
                    orig_index = row.orig_index
                except AttributeError as exc:
                    raise RuntimeError("Missing 'orig_index' for duplicate row") from exc
                if orig_index is None:
                    raise RuntimeError("Missing 'orig_index' for duplicate row")
                self._checkbox_map[checkbox] = int(orig_index)
            else:
                item = QtWidgets.QTableWidgetItem("")
                item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
                table.setItem(row_idx, 0, item)

            color = (
                QtGui.QColor(200, 255, 200)
                if getattr(row, "keep", False)
                else QtGui.QColor(255, 200, 200)
            )
            for col_idx in range(table.columnCount()):
                cell_item = table.item(row_idx, col_idx)
                if cell_item is not None:
                    cell_item.setBackground(color)

        table.resizeColumnsToContents()
        screen = QtWidgets.QApplication.primaryScreen()
        if screen:
            max_width = screen.availableGeometry().width() // table.columnCount()
            for i in range(table.columnCount()):
                table.setColumnWidth(i, min(table.columnWidth(i), max_width))
        layout.addWidget(table)
        self._table = table

        button_layout = QtWidgets.QHBoxLayout()
        self._remove_button = QtWidgets.QPushButton("Dubletten entfernen", self)
        self._remove_button.clicked.connect(self._emit_selection)
        button_layout.addWidget(self._remove_button)
        button_layout.addStretch()
        export_button = QtWidgets.QPushButton("Ergebnisse exportieren", self)
        export_button.clicked.connect(self._export_results)
        button_layout.addWidget(export_button)
        layout.addLayout(button_layout)

        self.setCentralWidget(container)

        self._select_all.setChecked(True)
        self._update_button_state()

        header = _require(table.verticalHeader(), "verticalHeader")
        total_width = header.width() + table.frameWidth() * 2
        v_scroll = _require(table.verticalScrollBar(), "verticalScrollBar")
        total_width += v_scroll.sizeHint().width()
        for i in range(table.columnCount()):
            total_width += table.columnWidth(i)
        screen = QtWidgets.QApplication.primaryScreen()
        screen_width = screen.availableGeometry().width() if screen else total_width
        self.resize(min(total_width, screen_width), 400)

    def _on_select_all(self, state: int) -> None:
        checked = state == QtCore.Qt.CheckState.Checked.value
        for cb in self._checkboxes:
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
        self._selected_count = len(self._checkboxes) if checked else 0
        self._update_button_state()

    def _on_checkbox_state_changed(self, state: int) -> None:
        if state == QtCore.Qt.CheckState.Checked.value:
            self._selected_count += 1
        else:
            self._selected_count -= 1
        self._update_button_state()

    def _update_button_state(self) -> None:
        any_checked = self._selected_count > 0
        self._remove_button.setVisible(any_checked)
        all_checked = self._selected_count == len(self._checkboxes)
        self._select_all.blockSignals(True)
        self._select_all.setChecked(all_checked)
        self._select_all.blockSignals(False)
        self._selected_count_label.setText(f"({self._selected_count})")


    def _emit_selection(self) -> None:
        indices = [
            self._checkbox_map[cb] for cb in self._checkboxes if cb.isChecked()
        ]
        self.remove_requested.emit(indices)
        self.close()

    def _export_results(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Dubletten exportieren",
            "dubletten.csv",
            "CSV Dateien (*.csv);;Alle Dateien (*)",
        )
        if not path:
            return
        selected = [
            self._checkbox_map[cb] for cb in self._checkboxes if cb.isChecked()
        ]
        export_df = prepare_duplicates_export(self._dataframe)
        export_df = export_df[
            (~export_df["keep"]) & (export_df["orig_index"].isin(selected))
        ]
        export_df = export_df.drop(columns=["keep", "pair_id", "orig_index"])

        # pandas' to_csv supports only single-character separators. Write the CSV
        # using a placeholder character and replace it with the desired multi-
        # character delimiter afterwards so that the exported file uses ``_§_``
        # like the original data source.
        placeholder = "\x1f"  # unit separator, unlikely to appear in data
        csv_data = export_df.to_csv(
            index=False, sep=placeholder, lineterminator="\n"
        )
        csv_data = csv_data.replace(placeholder, "_§_")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(csv_data)
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            QtWidgets.QMessageBox.information(
                self,
                "Export erfolgreich",
                f"Dublettenergebnisse wurden exportiert nach:\n{path}",
            )

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
    apply_modern_style(app)

    # Initialize the application icon after QApplication is created
    global APP_ICON
    APP_ICON = QtGui.QIcon(ICON_PATH)
    app.setWindowIcon(APP_ICON)

    window = MainWindow(args.csv_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
