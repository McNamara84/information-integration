import os
import sys
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.append(str(Path(__file__).resolve().parents[1]))
QtWidgets = pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)
from start import DataWarehouseWindow


def test_datawarehouse_window_persists_settings(tmp_path) -> None:
    os.environ["XDG_CONFIG_HOME"] = str(tmp_path)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    df = pd.DataFrame()

    window = DataWarehouseWindow(df)
    window._host.setText("example.com")
    window._port.setText("1234")
    window._user.setText("alice")
    window._password.setText("secret")
    window._dbname.setText("mydb")
    info = {
        "host": "example.com",
        "port": 1234,
        "user": "alice",
        "password": "secret",
        "dbname": "mydb",
    }
    window._save_settings(info)
    window.close()

    window2 = DataWarehouseWindow(df)
    assert window2._host.text() == "example.com"
    assert window2._port.text() == "1234"
    assert window2._user.text() == "alice"
    assert window2._password.text() == "secret"
    assert window2._dbname.text() == "mydb"
    window2.close()
    app.quit()
