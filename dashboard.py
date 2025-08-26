from __future__ import annotations

import math
import pgeocode
import psycopg2
from flask import Flask, render_template_string

BUNDESLAENDER = [
    "Baden-Württemberg",
    "Bayern",
    "Berlin",
    "Brandenburg",
    "Bremen",
    "Hamburg",
    "Hessen",
    "Mecklenburg-Vorpommern",
    "Niedersachsen",
    "Nordrhein-Westfalen",
    "Rheinland-Pfalz",
    "Saarland",
    "Sachsen",
    "Sachsen-Anhalt",
    "Schleswig-Holstein",
    "Thüringen",
]

INSTTYPE_NAMES = {
    "informationseinrichtung": "Informationseinrichtung",
    "oeffentliche-bibliothek": "Öffentliche Bibliothek",
    "spezialbibliothek": "Spezialbibliothek",
    "sonstige-Einrichtung": "Sonstige",
    "archiv": "Archiv",
    "wissenschaftliche-Bibliothek": "Wissenschaftliche Bibliothek",
    "bibliothek": "Bibliothek",
}


TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Dashboard</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css"/>
    <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css"/>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
</head>
<body class="p-4">
    <div class="container">
        <h1 class="mb-4">Dashboard</h1>
        <div class="row g-4">
            <div class="col-md-6">
                <div class="card text-center mb-4">
                    <div class="card-body">
                        <h5 class="card-title">Gesamtzahl Stellenanzeigen</h5>
                        <p class="display-4">{{ total }}</p>
                    </div>
                </div>
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">Häufigstes Gehalt nach Bundesland</h5>
                        <ul class="list-group list-group-flush">
                            {% for region, salary in salaries %}
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                {{ region }}
                                <span class="badge bg-secondary">{{ salary }}</span>
                            </li>
                            {% endfor %}
                        </ul>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">Verteilung nach Einrichtungstyp</h5>
                        <canvas id="instChart"></canvas>
                    </div>
                </div>
                <div class="card mt-4">
                    <div class="card-body">
                        <h5 class="card-title">Top 10 der häufigsten Arbeitgeber</h5>
                        <div id="companyCloud" class="d-flex flex-wrap justify-content-center">
                            {% for name, count, size in top_companies %}
                            <span
                                class="mx-2"
                                style="font-size: {{ size }}px;"
                                title="{{ count }} Angebote"
                            >
                                {{ name }}
                            </span>
                            {% endfor %}
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-12">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">Geografische Lage der Stellen</h5>
                        <div id="map" style="height: 500px;"></div>
                    </div>
                </div>
            </div>
            <div class="col-12">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">Stellen nach Bundesland</h5>
                        <ul class="list-group list-group-flush">
                            {% for region, count in regions %}
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                {{ region }}
                                <span class="badge bg-primary rounded-pill">{{ count }}</span>
                            </li>
                            {% endfor %}
                        </ul>
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
        const map = L.map('map').setView([51.3, 10.1], 6);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; OpenStreetMap contributors'
        }).addTo(map);
        const markerData = {{ markers|tojson }};
        const markerCluster = L.markerClusterGroup();
        markerData.forEach(m => {
            const marker = L.marker([m.lat, m.lon]).bindPopup(`<strong>${m.company}</strong><br>${m.jobdescription}`);
            markerCluster.addLayer(marker);
        });
        map.addLayer(markerCluster);
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
        cur.execute(
            """SELECT dl.region, COUNT(*)
               FROM dim_location dl
               JOIN fact_job fj ON dl.location_id = fj.location_id
               GROUP BY dl.region"""
        )
        region_rows = cur.fetchall()
        cur.execute(
            """SELECT region, salary FROM (
                   SELECT dl.region AS region,
                          fj.salary AS salary,
                          COUNT(*) AS cnt,
                          ROW_NUMBER() OVER (
                              PARTITION BY dl.region
                              ORDER BY COUNT(*) DESC
                          ) AS rn
                   FROM dim_location dl
                   JOIN fact_job fj ON dl.location_id = fj.location_id
                   WHERE fj.salary IS NOT NULL AND fj.salary <> ''
                   GROUP BY dl.region, fj.salary
               ) s
               WHERE rn = 1"""
        )
        salary_rows = cur.fetchall()
        cur.execute(
            """SELECT dc.company, COUNT(*) AS cnt
               FROM dim_company dc
               JOIN fact_job fj ON dc.company_id = fj.company_id
               WHERE dc.company IS NOT NULL AND dc.company <> ''
               GROUP BY dc.company
               ORDER BY cnt DESC
               LIMIT 10"""
        )
        top_company_rows = cur.fetchall()
        cur.execute(
            """SELECT dl.geo_lat, dl.geo_lon, dl.plz, dc.company, fj.jobdescription
               FROM dim_location dl
               JOIN fact_job fj ON dl.location_id = fj.location_id
               JOIN dim_company dc ON dc.company_id = fj.company_id"""
        )
        marker_rows = cur.fetchall()
        cur.close()
        conn.close()
        labels = []
        counts = []
        for insttype, count in rows:
            labels.append(INSTTYPE_NAMES.get(insttype, insttype or "Unbekannt"))
            counts.append(count)
        sum_counts = sum(counts) or 1
        labels = [f"{label} ({count / sum_counts * 100:.1f}%)" for label, count in zip(labels, counts)]
        region_counts = {bl: 0 for bl in BUNDESLAENDER}
        unknown = 0
        for region_name, count in region_rows:
            if region_name in region_counts:
                region_counts[region_name] += count
            else:
                unknown += count
        regions = [(bl, region_counts[bl]) for bl in BUNDESLAENDER]
        if unknown:
            regions.append(("Unbekannt", unknown))
        salary_map = {bl: "Unbekannt" for bl in BUNDESLAENDER}
        for region_name, salary in salary_rows:
            if region_name in salary_map and salary:
                salary_map[region_name] = salary
        salaries = [(bl, salary_map[bl]) for bl in BUNDESLAENDER]
        min_font, max_font = 14, 48
        top_companies = []
        if top_company_rows:
            counts_only = [cnt for _, cnt in top_company_rows]
            min_count = min(counts_only)
            max_count = max(counts_only)
            denom = max_count - min_count or 1
            for name, cnt in top_company_rows:
                size = min_font + (cnt - min_count) / denom * (max_font - min_font)
                top_companies.append((name, cnt, round(size, 2)))
        geocoder = pgeocode.Nominatim('de')
        plz_cache: dict[str, tuple[float, float] | None] = {}
        markers = []
        for lat, lon, plz, company, description in marker_rows:
            marker_info = {"company": company or "", "jobdescription": description or ""}
            if lat is not None and lon is not None:
                markers.append({"lat": float(lat), "lon": float(lon), **marker_info})
            elif plz:
                coords = plz_cache.get(plz)
                if coords is None and plz not in plz_cache:
                    result = geocoder.query_postal_code(plz)
                    if not math.isnan(result.latitude) and not math.isnan(result.longitude):
                        coords = (float(result.latitude), float(result.longitude))
                    plz_cache[plz] = coords
                coords = plz_cache.get(plz)
                if coords:
                    markers.append({"lat": coords[0], "lon": coords[1], **marker_info})
        return render_template_string(
            TEMPLATE,
            total=total,
            labels=labels,
            counts=counts,
            regions=regions,
            salaries=salaries,
            markers=markers,
            top_companies=top_companies,
        )

    return app
