from typing import Dict

import pandas as pd
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


def create_data_warehouse(df: pd.DataFrame, conn_info: Dict[str, str | int]) -> None:
    """Create a simple star-schema data warehouse from *df*.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleansed dataset without duplicates.
    conn_info : dict
        Dictionary containing connection parameters ``host``, ``port``,
        ``user``, ``password`` and ``dbname``.
    """
    host = conn_info.get("host", "localhost")
    port = int(conn_info.get("port", 5432))
    user = conn_info.get("user", "postgres")
    password = conn_info.get("password", "")
    dbname = conn_info.get("dbname", "datawarehouse")

    # Connect to default database to create the warehouse database
    admin_conn = psycopg2.connect(host=host, port=port, user=user, password=password, dbname="postgres")
    admin_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    admin_cur = admin_conn.cursor()
    admin_cur.execute(f"DROP DATABASE IF EXISTS {dbname}")
    admin_cur.execute(f"CREATE DATABASE {dbname}")
    admin_cur.close()
    admin_conn.close()

    # Connect to the newly created database
    conn = psycopg2.connect(host=host, port=port, user=user, password=password, dbname=dbname)
    cur = conn.cursor()

    # Create dimension tables
    cur.execute(
        """
        CREATE TABLE dim_company (
            company_id SERIAL PRIMARY KEY,
            company TEXT,
            insttype TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE dim_location (
            location_id SERIAL PRIMARY KEY,
            location TEXT,
            country TEXT,
            geo_lat DOUBLE PRECISION,
            geo_lon DOUBLE PRECISION,
            plz TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE dim_jobtype (
            jobtype_id SERIAL PRIMARY KEY,
            jobtype TEXT
        )
        """
    )

    # Create fact table
    cur.execute(
        """
        CREATE TABLE fact_job (
            job_id SERIAL PRIMARY KEY,
            jobdescription TEXT,
            company_id INTEGER REFERENCES dim_company(company_id),
            jobtype_id INTEGER REFERENCES dim_jobtype(jobtype_id),
            location_id INTEGER REFERENCES dim_location(location_id),
            fixedterm TEXT,
            workinghours TEXT,
            salary TEXT
        )
        """
    )

    # Populate dimension tables
    company_df = df[["company", "insttype"]].drop_duplicates().reset_index(drop=True)
    for _, row in company_df.iterrows():
        cur.execute(
            "INSERT INTO dim_company (company, insttype) VALUES (%s, %s)",
            (row.get("company"), row.get("insttype")),
        )
    conn.commit()
    cur.execute("SELECT company_id, company, insttype FROM dim_company")
    company_map = {(r[1], r[2]): r[0] for r in cur.fetchall()}

    location_df = df[["location", "country", "geo_lat", "geo_lon", "plz"]].drop_duplicates().reset_index(drop=True)
    for _, row in location_df.iterrows():
        cur.execute(
            """INSERT INTO dim_location (location, country, geo_lat, geo_lon, plz)
            VALUES (%s, %s, %s, %s, %s)""",
            (
                row.get("location"),
                row.get("country"),
                row.get("geo_lat"),
                row.get("geo_lon"),
                row.get("plz"),
            ),
        )
    conn.commit()
    cur.execute(
        "SELECT location_id, location, country, geo_lat, geo_lon, plz FROM dim_location"
    )
    location_map = {(r[1], r[2], r[3], r[4], r[5]): r[0] for r in cur.fetchall()}

    jobtype_df = df[["jobtype"]].drop_duplicates().reset_index(drop=True)
    for _, row in jobtype_df.iterrows():
        cur.execute(
            "INSERT INTO dim_jobtype (jobtype) VALUES (%s)",
            (row.get("jobtype"),),
        )
    conn.commit()
    cur.execute("SELECT jobtype_id, jobtype FROM dim_jobtype")
    jobtype_map = {r[1]: r[0] for r in cur.fetchall()}

    # Populate fact table
    for _, row in df.iterrows():
        company_id = company_map.get((row.get("company"), row.get("insttype")))
        location_id = location_map.get(
            (
                row.get("location"),
                row.get("country"),
                row.get("geo_lat"),
                row.get("geo_lon"),
                row.get("plz"),
            )
        )
        jobtype_id = jobtype_map.get(row.get("jobtype"))
        cur.execute(
            """INSERT INTO fact_job (
            jobdescription, company_id, jobtype_id, location_id, fixedterm, workinghours, salary
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (
                row.get("jobdescription"),
                company_id,
                jobtype_id,
                location_id,
                row.get("fixedterm"),
                row.get("workinghours"),
                row.get("salary"),
            ),
        )
    conn.commit()
    cur.close()
    conn.close()
