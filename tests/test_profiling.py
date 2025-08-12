import pandas as pd
from profiling import profile_dataframe, top_error


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


def test_profile_window_width_respects_screen():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    from start import ProfileWindow

    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    stats = profile_dataframe(df)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = ProfileWindow(stats)
    win.show()
    app.processEvents()
    table = win.centralWidget()
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
    win2 = ProfileWindow(stats_wide)
    win2.show()
    app.processEvents()
    table2 = win2.centralWidget()
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
