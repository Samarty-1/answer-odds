"""Tests run on the seeded made-up site (odds.fixture), in a database they may wipe."""
import os
import sys
from pathlib import Path

import psycopg
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TEST_DSN = os.environ.get("ODDS_TEST_DSN", "postgresql://postgres@localhost:5544/odds_test")


def _ensure_database(dsn):
    try:
        psycopg.connect(dsn).close()
        return
    except psycopg.OperationalError:
        pass
    name = psycopg.conninfo.conninfo_to_dict(dsn)["dbname"]
    with psycopg.connect(psycopg.conninfo.make_conninfo(dsn, dbname="postgres"), autocommit=True) as c:
        c.execute(psycopg.sql.SQL("CREATE DATABASE {}").format(psycopg.sql.Identifier(name)))


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    assert "test" in psycopg.conninfo.conninfo_to_dict(TEST_DSN)["dbname"]
    _ensure_database(TEST_DSN)
    os.environ["ODDS_DSN"] = TEST_DSN
    from odds import build, fixture, load

    xml = tmp_path_factory.mktemp("xml")
    fixture.write(xml)
    load.load(xml, verbose=False)
    return build.build(verbose=False)


@pytest.fixture()
def conn(built):
    with psycopg.connect(TEST_DSN, autocommit=True) as c:
        yield c
