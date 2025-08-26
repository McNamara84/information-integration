from __future__ import annotations

import psycopg2
from flask import Flask, render_template_string

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Dashboard</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body class="p-4">
    <div class="container">
        <h1 class="mb-4">Dashboard</h1>
        <div class="row g-4">
            <div class="col-md-6">
                <div class="card text-center">
                    <div class="card-body">
                        <h5 class="card-title">Gesamtzahl Stellenanzeigen</h5>
                        <p class="display-4">{{ total }}</p>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">Insttype Verteilung</h5>
                        <canvas id="instChart"></canvas>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script>
        const ctx = document.getElementById('instChart');
        new Chart(ctx, {
            type: 'pie',
            data: {
                labels: {{ labels|tojson }},
                datasets: [{
                    data: {{ counts|tojson }}
                }]
            }
        });
    </script>
</body>
</html>
"""


def create_app(conn_info: dict[str, str | int]) -> Flask:
    app = Flask(__name__)

    def get_conn():
        host = conn_info.get("host", "localhost")
        port = int(conn_info.get("port", 5432))
        user = conn_info.get("user", "postgres")
        password = conn_info.get("password", "")
        dbname = conn_info.get("dbname", "datawarehouse")
        return psycopg2.connect(
            host=host, port=port, user=user, password=password, dbname=dbname
        )

    @app.route('/')
    def index():
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM fact_job")
        row = cur.fetchone()
        total = row[0] if row and row[0] is not None else 0
        cur.execute(
            """SELECT dc.insttype, COUNT(*)
               FROM dim_company dc
               JOIN fact_job fj ON dc.company_id = fj.company_id
               GROUP BY dc.insttype"""
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        labels = []
        counts = []
        for insttype, count in rows:
            labels.append(insttype or "Unbekannt")
            counts.append(count)
        sum_counts = sum(counts) or 1
        labels = [f"{label} ({count / sum_counts * 100:.1f}%)" for label, count in zip(labels, counts)]
        return render_template_string(TEMPLATE, total=total, labels=labels, counts=counts)

    return app
