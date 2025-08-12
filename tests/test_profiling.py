import pandas as pd
from profiling import profile_dataframe, top_error, get_all_error_types


def test_profile_dataframe_basic():
    df = pd.DataFrame({
        "a": [1, 2, None, "??"],
        "b": ["x", "y", "x", "na"],
    })
    profile = profile_dataframe(df)

    col_a = profile[profile["Spalte"] == "a"].iloc[0]
    assert col_a["Zeilen"] == 4
    assert col_a["Fehlende Werte"] == 1
    assert col_a["Fehlende Werte %"] == 25.0
    assert col_a["Eindeutige Werte"] == 3

    col_b = profile[profile["Spalte"] == "b"].iloc[0]
    assert col_b["Häufigste Fehlerart"] == "na"
    assert col_b["Fehler Häufigkeit"] == 1
    assert col_b["Fehler %"] == 25.0


def test_none_as_top_error():
    series = pd.Series([None, None, "x"])
    val, count = top_error(series)
    assert val is None
    assert count == 2
    profile = profile_dataframe(pd.DataFrame({"a": series}))
    col = profile[profile["Spalte"] == "a"].iloc[0]
    assert col["Häufigste Fehlerart"] == "None"
    assert col["Fehler Häufigkeit"] == 2


def test_get_all_error_types_basic():
    """Test that get_all_error_types returns multiple error types."""
    series = pd.Series([None, None, "??", "valid"])
    errors = get_all_error_types(series, "test_column")
    
    # Should find missing values
    assert len(errors) >= 1
    assert any(error_type == "Fehlende Werte" for error_type, _ in errors)
    
    # Check that error rates are reasonable
    for error_type, rate in errors:
        assert 0 <= rate <= 100


def test_get_all_error_types_duplicates():
    """Test duplicate detection for unique columns."""
    series = pd.Series([1, 2, 2, 3])  # Has duplicates
    errors = get_all_error_types(series, "jobid")  # jobid should be unique
    
    # Should detect duplicates
    duplicate_errors = [e for e in errors if e[0] == "Eindeutigkeitsverletzungen"]
    assert len(duplicate_errors) == 1
    assert duplicate_errors[0][1] == 25.0  # 1 duplicate out of 4 = 25%


def test_get_all_error_types_spelling_errors():
    """Test detection of spelling errors (HTML entities)."""
    series = pd.Series(["normal", "&#8222;test&#8220;", "another"])
    errors = get_all_error_types(series, "company")
    
    # Should detect spelling errors
    spelling_errors = [e for e in errors if e[0] == "Schreibfehler"]
    assert len(spelling_errors) == 1
    assert spelling_errors[0][1] > 0


def test_get_all_error_types_no_errors():
    """Test with clean data."""
    series = pd.Series(["clean", "data", "here"])
    errors = get_all_error_types(series, "test_column")
    
    # Should return empty list for clean data
    assert len(errors) == 0


def test_profile_window_width_respects_screen():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    from start import ProfileWindow

    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    stats = profile_dataframe(df)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = ProfileWindow(stats, df)  # Pass dataframe as second argument
    win.show()
    app.processEvents()
    table = win.centralWidget().findChild(QtWidgets.QTableWidget)
    expected = table.verticalHeader().width() + table.frameWidth() * 2
    expected += table.verticalScrollBar().sizeHint().width()
    for i in range(table.columnCount()):
        expected += table.columnWidth(i)
    screen_width = app.primaryScreen().availableGeometry().width()
    assert win.width() == min(expected, screen_width)
    win.close()

    # Create a very wide table to ensure screen width is the limiting factor
    df_wide = pd.DataFrame({f"col{i}": [i] for i in range(200)})
    stats_wide = profile_dataframe(df_wide)
    win2 = ProfileWindow(stats_wide, df_wide)  # Pass dataframe as second argument
    win2.show()
    app.processEvents()
    table2 = win2.centralWidget().findChild(QtWidgets.QTableWidget)
    total_w = table2.verticalHeader().width() + table2.frameWidth() * 2
    total_w += table2.verticalScrollBar().sizeHint().width()
    for i in range(table2.columnCount()):
        total_w += table2.columnWidth(i)
    assert total_w > screen_width
    assert win2.width() == screen_width
    win2.close()


def test_profile_window_cleanup():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    from start import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    win = MainWindow.__new__(MainWindow)
    QtWidgets.QMainWindow.__init__(win)
    win._dataframe = pd.DataFrame({"a": [1]})
    win._profile_window = None

    win._show_profile()
    first = win._profile_window
    assert first.isVisible()

    win._show_profile()
    app.processEvents()
    assert not first.isVisible()

    win._profile_window.close()
    app.processEvents()
    assert win._profile_window is None
    win.close()


def test_export_report(tmp_path, monkeypatch):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    from start import ProfileWindow

    # Create test data with known error patterns
    df = pd.DataFrame({
        "a": [1, None, 3],  # Has missing values
        "b": ["&#8222;test&#8220;", "normal", "x"],  # Has spelling errors and possible cryptic values
        "jobid": [1, 2, 2]  # Has duplicates (should be unique)
    })
    stats = profile_dataframe(df)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = ProfileWindow(stats, df)  # Pass dataframe as second argument

    file_path = tmp_path / "report.xlsx"
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: (str(file_path), "")
    )

    win._export_button.click()
    app.processEvents()

    assert file_path.exists()
    report = pd.read_excel(file_path)
    
    # Check that we have multiple rows (since we now export all error types)
    assert len(report) >= 3  # At least one error type per column
    
    # Check that all columns are present
    attributes = report["Attribut"].unique()
    assert "a" in attributes
    assert "b" in attributes
    assert "jobid" in attributes
    
    # Check that we have different error types
    error_types = report["Fehlertyp"].unique()
    assert len(error_types) >= 2  # Should have at least 2 different error types
    
    # Check that missing values are detected for column 'a'
    missing_errors = report[(report["Attribut"] == "a") & (report["Fehlertyp"] == "Fehlende Werte")]
    assert len(missing_errors) >= 1
    if len(missing_errors) > 0:
        # 1 missing value out of 3 = 33.33%
        assert missing_errors.iloc[0]["Relative Fehlerquote (%)"] == pytest.approx(33.33, abs=0.1)
    
    # Check that duplicates are detected for jobid
    duplicate_errors = report[(report["Attribut"] == "jobid") & (report["Fehlertyp"] == "Eindeutigkeitsverletzungen")]
    assert len(duplicate_errors) >= 1
    if len(duplicate_errors) > 0:
        # 1 duplicate out of 3 = 33.33%
        assert duplicate_errors.iloc[0]["Relative Fehlerquote (%)"] == pytest.approx(33.33, abs=0.1)
    
    win.close()


def test_export_report_no_errors(tmp_path, monkeypatch):
    """Test export with clean data that has no errors."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    from start import ProfileWindow

    # Create clean test data
    df = pd.DataFrame({
        "clean_col": ["value1", "value2", "value3"],
    })
    stats = profile_dataframe(df)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = ProfileWindow(stats, df)

    file_path = tmp_path / "report_clean.xlsx"
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: (str(file_path), "")
    )

    win._export_button.click()
    app.processEvents()

    assert file_path.exists()
    report = pd.read_excel(file_path)
    
    # Should have at least one row (for the "no errors" case)
    assert len(report) >= 1
    
    # Check that the clean column is marked as having no significant errors
    clean_rows = report[report["Attribut"] == "clean_col"]
    assert len(clean_rows) >= 1
    # Should either have no rows (no errors detected) or one row with "Keine signifikanten Fehler"
    if len(clean_rows) > 0:
        assert clean_rows.iloc[0]["Fehlertyp"] == "Keine signifikanten Fehler"
        assert clean_rows.iloc[0]["Relative Fehlerquote (%)"] == 0.0
    
    win.close()