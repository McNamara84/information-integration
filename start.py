"""PyQt based user interface for the information integration project."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import webbrowser
import socket
import io
from typing import TYPE_CHECKING, cast, TypeVar

import pandas as pd
from PyQt6 import QtCore, QtWidgets, QtGui
import psycopg2

ICON_PATH = os.path.join(os.path.dirname(__file__), "assets", "icon.png")
# Will be initialized after QApplication creation in main()
APP_ICON: QtGui.QIcon | None = None

from profiling import profile_dataframe, get_all_error_types

from load_bibliojobs import load_bibliojobs
from cleaning import (
    clean_dataframe,
    find_fuzzy_duplicates,
    DEDUPLICATE_COLUMNS,
    prepare_duplicates_export,
    format_export_columns,
)
from data_warehouse import create_data_warehouse, is_data_warehouse_initialized

if TYPE_CHECKING:  # pragma: no cover - typing only
    from werkzeug.serving import BaseWSGIServer


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
    """Ensure that a value is present.

    Parameters
    ----------
    value : T or None
        Value to validate.
    name : str
        Name used in the error message.

    Returns
    -------
    T
        The validated value.

    Raises
    ------
    RuntimeError
        If ``value`` is ``None``.
    """
    if value is None:
        raise RuntimeError(f"{name} is unexpectedly None")
    return value


def _find_free_port() -> int:
    """Return an available port on ``localhost``.

    Returns
    -------
    int
        A free TCP port.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def apply_modern_style(app: QtWidgets.QApplication) -> None:
    """Apply a Windows 11 inspired style to a Qt application.

    Parameters
    ----------
    app : QtWidgets.QApplication
        Application instance whose palette and stylesheet are modified.
    """
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
            background-color: palette(mid);
            color: palette(buttonText);
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


def dataframe_to_custom_csv(dataframe: pd.DataFrame) -> str:
    """Return a CSV representation using ``_§_`` as separator.

    Parameters
    ----------
    dataframe:
        The :class:`pandas.DataFrame` to serialize.

    Returns
    -------
    str
        CSV formatted string with ``_§_`` as field delimiter and ``\n`` as line
        terminator.
    """

    buffer = io.StringIO()
    dataframe.to_csv(buffer, index=False, sep="\t", lineterminator="\n")
    return buffer.getvalue().replace("\t", "_§_")


class LoadWorker(QtCore.QObject):
    """Worker object that reads the raw CSV file in a background thread."""

    finished = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(int)
    error = QtCore.pyqtSignal(str)

    def __init__(self, path: str) -> None:
        super().__init__()
        self._path = path

    @QtCore.pyqtSlot()
    def run(self):
        """Load the CSV file and emit progress and result signals."""

        def callback(value: float) -> None:
            self.progress.emit(int(value))

        try:
            dataframe = load_bibliojobs(self._path, progress_callback=callback)
        except FileNotFoundError as exc:  # pragma: no cover
            self.error.emit(str(exc))
            return

        self.finished.emit(dataframe)


class CleanWorker(QtCore.QObject):
    """Clean the loaded DataFrame in a worker thread."""

    finished = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(int)
    status = QtCore.pyqtSignal(str)

    def __init__(self, dataframe) -> None:
        super().__init__()
        self._dataframe = dataframe

    @QtCore.pyqtSlot()
    def run(self):
        """Execute the cleaning pipeline and emit progress updates."""

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
    """Find duplicate rows in a background thread."""

    finished = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(int)

    def __init__(self, dataframe) -> None:
        super().__init__()
        self._dataframe = dataframe

    @QtCore.pyqtSlot()
    def run(self):
        """Detect duplicates and emit the results."""

        def callback(value: float) -> None:
            self.progress.emit(int(value))

        _, duplicates = find_fuzzy_duplicates(
            self._dataframe,
            DEDUPLICATE_COLUMNS,
            progress_callback=callback,
        )
        self.finished.emit(duplicates)


class DataWarehouseWorker(QtCore.QObject):
    """Populate the PostgreSQL data warehouse in a worker thread."""

    finished = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(int)
    status = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)

    def __init__(self, dataframe, conn_info) -> None:
        super().__init__()
        self._dataframe = dataframe
        self._conn_info = conn_info

    @QtCore.pyqtSlot()
    def run(self):
        """Create the data warehouse and report progress and errors."""

        def progress_cb(value: float) -> None:
            self.progress.emit(int(value))

        def status_cb(message: str) -> None:
            self.status.emit(message)

        try:
            create_data_warehouse(
                self._dataframe,
                self._conn_info,
                progress_callback=progress_cb,
                status_callback=status_cb,
            )
        except psycopg2.OperationalError as exc:  # pragma: no cover - UI only
            self.error.emit(
                "Verbindung zur Datenbank fehlgeschlagen. Bitte prüfen Sie, ob der PostgreSQL-Server läuft und die Zugangsdaten korrekt sind.\n"
                + str(exc)
            )
        except Exception as exc:  # pragma: no cover - UI only
            self.error.emit(str(exc))
        else:
            self.finished.emit()


class MainWindow(QtWidgets.QMainWindow):
    """Main application window orchestrating the data integration workflow."""

    def __init__(self, path: str) -> None:
        super().__init__()
        self.setWindowTitle("Informationsintegration")
        if APP_ICON is not None:
            self.setWindowIcon(APP_ICON)
        self.resize(800, 600)
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

        self._export_csv_button = QtWidgets.QPushButton(
            "Ergebnis als CSV-Datei speichern"
        )
        self._export_csv_button.hide()
        self._export_csv_button.clicked.connect(self._export_csv)

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.addWidget(self._button)
        layout.addWidget(self._clean_button)
        layout.addWidget(self._dedupe_button)
        layout.addStretch()
        layout.addWidget(self._export_cleaned_button)
        layout.addWidget(self._export_csv_button)
        self._init_db_button = QtWidgets.QPushButton("Datenbank initialisieren")
        self._init_db_button.hide()
        self._init_db_button.clicked.connect(self._init_database)
        layout.addWidget(self._init_db_button)
        self._visualize_button = QtWidgets.QPushButton("Datensätze visualisieren")
        self._visualize_button.hide()
        self._visualize_button.clicked.connect(self._visualize_data)
        layout.addWidget(self._visualize_button)
        self._stop_server_button = QtWidgets.QPushButton("Webserver beenden")
        self._stop_server_button.hide()
        self._stop_server_button.clicked.connect(self._stop_server)
        layout.addWidget(self._stop_server_button)
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
        self._server: BaseWSGIServer | None = None
        self._server_thread: threading.Thread | None = None

    @QtCore.pyqtSlot(object)
    def _on_finished(self, df) -> None:
        """Handle completion of the CSV loading step."""

        self._status.showMessage("Einlesen abgeschlossen", 5000)
        self._progress.setValue(100)
        self._dataframe = df
        self._button.setEnabled(True)
        self._clean_button.setEnabled(True)

    @QtCore.pyqtSlot(str)
    def _on_error(self, message: str) -> None:
        """Display an error message from the loading worker."""

        self._status.showMessage(message, 5000)
        self._progress.setValue(0)

    def _show_profile(self) -> None:
        """Show profiling statistics for the currently loaded dataset."""

        if self._profile_window is not None:
            self._profile_window.close()
            self._profile_window = None
        stats = profile_dataframe(self._dataframe)
        window = ProfileWindow(stats, self._dataframe, self)
        window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        window.closed.connect(self._on_profile_window_destroyed)
        window.show()
        self._profile_window = window

    def _on_profile_window_destroyed(self) -> None:
        self._profile_window = None

    def _clean_data(self) -> None:
        """Start the data cleaning process in a worker thread."""

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
        """Receive the cleaned DataFrame from the worker."""

        self._dataframe = df
        self._status.showMessage("Bereinigung abgeschlossen", 5000)
        self._progress.setValue(100)
        self._dedupe_button.setEnabled(True)
        self._export_cleaned_button.show()
        self._export_csv_button.show()

    def _export_cleaned(self) -> None:
        """Export the cleaned dataset to an Excel file."""

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
    
    def _export_csv(self) -> None:
        """Export the cleaned dataset to a CSV file."""

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Bereinigte Daten exportieren", filter="CSV Dateien (*.csv)"
        )
        if not path:
            return
        content = dataframe_to_custom_csv(self._dataframe)
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(content)
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            QtWidgets.QMessageBox.information(
                self,
                "Export erfolgreich",
                f"Daten wurden erfolgreich exportiert nach:\n{path}",
            )


    def _init_database(self) -> None:
        """Open a dialog to create the data warehouse."""

        window = DataWarehouseWindow(self._dataframe, self)
        result = window.exec()
        if result == QtWidgets.QDialog.DialogCode.Accepted:
            self._visualize_button.show()

    def _visualize_data(self) -> None:
        """Launch a local web server to display the dashboard."""

        if self._server is not None:
            self._status.showMessage("Webserver läuft bereits", 5000)
            return
        settings = QtCore.QSettings("fh-potsdam", "information-integration")
        info = {
            "host": settings.value("db/host", "localhost"),
            "port": int(settings.value("db/port", 5432)),
            "user": settings.value("db/user", ""),
            "password": settings.value("db/password", ""),
            "dbname": settings.value("db/dbname", "bibliojobs_dw"),
        }
        port = _find_free_port()

        from dashboard import create_app
        from werkzeug.serving import make_server

        app = create_app(info)
        server = make_server("127.0.0.1", port, app)
        self._server = server
        self._server_thread = threading.Thread(
            target=server.serve_forever, daemon=True
        )
        self._server_thread.start()
        webbrowser.open(f"http://127.0.0.1:{port}/")
        self._stop_server_button.show()
        self._visualize_button.setEnabled(False)

    def _stop_server(self) -> None:
        """Shut down the dashboard web server if it is running."""

        if self._server is None:
            return
        self._server.shutdown()
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=1)
        self._server = None
        self._server_thread = None
        self._stop_server_button.hide()
        self._visualize_button.setEnabled(True)

    def _remove_duplicates(self) -> None:
        """Start duplicate detection in a worker thread."""

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
            self._init_db_button.show()

class ProfileWindow(QtWidgets.QMainWindow):
    """Window displaying profiling statistics for the dataset."""

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
        rows = []
        for column in self._dataframe.columns:
            series = self._dataframe[column]
            all_errors = get_all_error_types(series, column)
            
            if all_errors:
                for error_type, error_rate in all_errors:
                    rows.append({
                        "Attribut": column,
                        "Fehlertyp": error_type,
                        "Relative Fehlerquote (%)": round(error_rate, 2),
                    })
            else:
                rows.append({
                    "Attribut": column,
                    "Fehlertyp": "Keine signifikanten Fehler",
                    "Relative Fehlerquote (%)": 0.0,
                })
        rows.sort(key=lambda x: (x["Attribut"], -x["Relative Fehlerquote (%)"]))
        
        report_df = pd.DataFrame(rows)
        report_df.to_excel(path, index=False)
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
    """Window used to inspect and delete duplicate records."""

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
        export_df = format_export_columns(export_df)

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


class DataWarehouseWindow(QtWidgets.QDialog):
    """Dialog guiding the creation of the data warehouse."""

    def __init__(self, dataframe, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Datenbank initialisieren")
        if APP_ICON is not None:
            self.setWindowIcon(APP_ICON)
        self._dataframe = dataframe

        form = QtWidgets.QFormLayout(self)
        self._host = QtWidgets.QLineEdit("localhost", self)
        self._port = QtWidgets.QLineEdit("5432", self)
        self._user = QtWidgets.QLineEdit(self)
        self._password = QtWidgets.QLineEdit(self)
        self._password.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self._dbname = QtWidgets.QLineEdit("bibliojobs_dw", self)

        self._settings = QtCore.QSettings("fh-potsdam", "information-integration")
        self._load_settings()

        form.addRow("Host", self._host)
        form.addRow("Port", self._port)
        form.addRow("Benutzer", self._user)
        form.addRow("Passwort", self._password)
        form.addRow("Datenbankname", self._dbname)

        self._create_button = QtWidgets.QPushButton("Data Warehouse erstellen", self)
        self._create_button.clicked.connect(self._create)
        form.addRow(self._create_button)
        self._status = QtWidgets.QLabel("", self)
        self._status.hide()
        form.addRow(self._status)
        self._progress = QtWidgets.QProgressBar(self)
        self._progress.setRange(0, 100)
        self._progress.hide()
        form.addRow(self._progress)

        self._worker: DataWarehouseWorker | None = None
        self._thread: QtCore.QThread | None = None

    def _create(self) -> None:
        try:
            info: dict[str, str | int] = {
                "host": self._host.text(),
                "port": int(self._port.text()),
                "user": self._user.text(),
                "password": self._password.text(),
                "dbname": self._dbname.text(),
            }
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "Fehler", "Port muss eine Zahl sein")
            return

        self._save_settings(info)

        try:
            if is_data_warehouse_initialized(self._dataframe, info):
                skip = True
                if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
                    box = QtWidgets.QMessageBox(self)
                    box.setWindowTitle("Datenbank vorhanden")
                    box.setText(
                        "Alle Tabellen sind bereits vorhanden und enthalten die erwartete Anzahl an Datensätzen.\n"
                        "Möchten Sie die Erstellung überspringen?"
                    )
                    box.setStandardButtons(
                        QtWidgets.QMessageBox.StandardButton.Yes
                        | QtWidgets.QMessageBox.StandardButton.No
                    )
                    yes_button = box.button(
                        QtWidgets.QMessageBox.StandardButton.Yes
                    )
                    if yes_button is not None:
                        yes_button.setText("Ja")
                    no_button = box.button(
                        QtWidgets.QMessageBox.StandardButton.No
                    )
                    if no_button is not None:
                        no_button.setText("Nein")
                    result = box.exec()
                    skip = result == QtWidgets.QMessageBox.StandardButton.Yes
                if skip:
                    self.accept()
                    return
        except psycopg2.Error:
            pass

        self._create_button.setEnabled(False)
        self._status.show()
        self._progress.show()
        self._progress.setValue(0)
        self._status.setText("Starte Import ...")

        self._worker = DataWarehouseWorker(self._dataframe, info)
        self._thread = QtCore.QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.status.connect(self._status.setText)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.error.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_finished(self) -> None:
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            QtWidgets.QMessageBox.information(
                self, "Erfolg", "Data Warehouse wurde erstellt",
            )
        self.accept()

    def _on_error(self, message: str) -> None:  # pragma: no cover - UI only
        QtWidgets.QMessageBox.critical(self, "Fehler", message)
        self._create_button.setEnabled(True)

    def _load_settings(self) -> None:
        self._host.setText(self._settings.value("db/host", self._host.text()))
        self._port.setText(str(self._settings.value("db/port", self._port.text())))
        self._user.setText(self._settings.value("db/user", ""))
        self._password.setText(self._settings.value("db/password", ""))
        self._dbname.setText(self._settings.value("db/dbname", self._dbname.text()))

    def _save_settings(self, info: dict[str, str | int]) -> None:
        self._settings.setValue("db/host", info["host"])
        self._settings.setValue("db/port", str(info["port"]))
        self._settings.setValue("db/user", info["user"])
        self._settings.setValue("db/password", info["password"])
        self._settings.setValue("db/dbname", info["dbname"])
        self._settings.sync()

def main() -> None:
    parser = argparse.ArgumentParser(description="Startet die Informationsintegration-GUI")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="bibliojobs_raw.csv",
        help="Pfad zur einzulesenden CSV-Datei",
    )
    args = parser.parse_args()

    if sys.platform.startswith("win"):  # pragma: no cover - Windows only
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "information-integration"
        )

    app = QtWidgets.QApplication(sys.argv)

    if sys.platform == "darwin":  # pragma: no cover - macOS only
        try:
            from AppKit import NSApplication, NSImage  # type: ignore

            NSApplication.sharedApplication().setApplicationIconImage_(
                NSImage.alloc().initWithContentsOfFile_(ICON_PATH)
            )
        except Exception:  # pragma: no cover - optional dependency
            pass
    apply_modern_style(app)
    global APP_ICON
    APP_ICON = QtGui.QIcon(ICON_PATH)
    app.setWindowIcon(APP_ICON)

    window = MainWindow(args.csv_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
