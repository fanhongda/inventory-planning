"""
The HTTP surface — that it is a wrapper, and that a correction survives it.

The two properties worth holding. A correction made over HTTP has to land as an ordinary
declaration, in the ordinary place, so a headless run reproduces a run driven from a
browser. And re-resolving has to read the landed rows rather than the uploaded file,
because that is the whole reason the landing layer exists: fixing a mapping should cost
neither a re-upload nor a hand-authored adapter.
"""

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi", reason="the HTTP surface is an optional extra")
from fastapi.testclient import TestClient  # noqa: E402

from inventory_planning.api import create_app  # noqa: E402
from inventory_planning.store.declarations import (  # noqa: E402
    DECLARATIONS_DIRNAME, Declarations,
)

SAMPLE = Path("sample_data/inventory.csv")


@pytest.fixture
def workspace(tmp_path):
    """A config directory and a store of its own — the store path is the isolation."""
    config = tmp_path / "config"
    config.mkdir()
    return config, tmp_path / "store"


@pytest.fixture
def client(workspace):
    config, store = workspace
    return TestClient(create_app(config_dir=config, store_root=store))


def _upload(client, path=SAMPLE, name=None):
    with open(path, "rb") as fh:
        return client.post("/uploads",
                           files={"file": (name or path.name, fh.read(), "text/csv")})


class TestWhatThePipelineCanRead:

    def test_health_names_the_store_it_is_pointed_at(self, client, workspace):
        body = client.get("/health").json()
        assert body["ok"] is True
        assert str(workspace[1]) in body["store_root"]
        assert body["contracts"] == 9

    def test_contracts_are_listed_with_their_fingerprint(self, client):
        body = client.get("/contracts").json()
        by_type = {c["doc_type"]: c for c in body}
        assert "inventory" in by_type
        assert by_type["inventory"]["required"] >= 1
        assert len(by_type["inventory"]["fingerprint"]) == 12

    def test_one_contract_carries_its_fields_and_aliases(self, client):
        body = client.get("/contracts/inventory").json()
        sku = [f for f in body["fields"] if f["field"] == "sku"][0]
        assert sku["required"] is True
        assert sku["aliases"]

    def test_an_unknown_contract_is_404(self, client):
        assert client.get("/contracts/warehouse_layout").status_code == 404

    def test_a_template_downloads_as_a_workbook(self, client):
        response = client.get("/contracts/substitution/template")
        assert response.status_code == 200
        assert response.content[:2] == b"PK"     # a zip, which is what xlsx is
        assert "template_substitution.xlsx" in response.headers["content-disposition"]


class TestGettingAFileIn:

    def test_an_upload_lands_and_says_what_was_made_of_it(self, client):
        body = _upload(client).json()
        assert len(body["landed"]) == 1
        landed = body["landed"][0]
        assert landed["doc_type"] == "inventory"
        assert landed["rows"] == 10

        resolution = landed["resolution"]
        assert resolution["missing_required"] == []
        mapped = {f["field"]: f for f in resolution["fields"]}
        assert mapped["sku"]["source"] == "mapped"
        assert mapped["sku"]["column"] == "Item Code"
        assert mapped["sku"]["fill_rate"] == 1.0

    def test_the_batch_is_listed_as_landed_not_promoted(self, client):
        batch_id = _upload(client).json()["landed"][0]["batch_id"]
        listed = {b["batch_id"]: b for b in client.get("/batches").json()}
        assert listed[batch_id]["status"] == "landed"
        assert listed[batch_id]["doc_type"] == "inventory"

    def test_the_landed_rows_keep_the_headers_the_file_used(self, client):
        batch_id = _upload(client).json()["landed"][0]["batch_id"]
        body = client.get(f"/batches/{batch_id}/rows", params={"limit": 3}).json()
        assert body["total"] == 10
        assert len(body["rows"]) == 3
        assert "Item Code" in body["columns"]

    def test_an_unreadable_upload_is_400_not_500(self, client, tmp_path):
        junk = tmp_path / "notes.csv"
        junk.write_text("", encoding="utf-8")
        assert _upload(client, junk).status_code == 400

    def test_a_missing_batch_is_404(self, client):
        assert client.get("/batches/nope/resolution").status_code == 404


class TestResolvingAgainWithoutTheFile:

    def test_the_resolution_is_rebuilt_from_the_landed_rows(self, client):
        """No path to the upload is kept. If this works, landing is the source."""
        batch_id = _upload(client).json()["landed"][0]["batch_id"]
        body = client.get(f"/batches/{batch_id}/resolution").json()
        assert body["doc_type"] == "inventory"
        assert body["rows"] == 10
        assert body["tests_passed"] is True


class TestACorrectionIsADeclaration:

    def test_it_is_refused_without_a_reason_or_a_name(self, client):
        batch_id = _upload(client).json()["landed"][0]["batch_id"]
        response = client.post(f"/batches/{batch_id}/declarations",
                               json={"scope": "value", "field": "currency",
                                     "value": "CNY", "by": "jfanhon"})
        assert response.status_code == 400
        assert "reason" in response.json()["detail"]

    def test_a_declared_value_lands_in_the_config_directory_as_yaml(
            self, client, workspace):
        config, _ = workspace
        batch_id = _upload(client).json()["landed"][0]["batch_id"]
        response = client.post(f"/batches/{batch_id}/declarations",
                               json={"scope": "value", "field": "currency",
                                     "value": "CNY", "by": "jfanhon",
                                     "reason": "stock is carried at standard cost in CNY"})
        assert response.status_code == 200

        written = list((config / DECLARATIONS_DIRNAME).glob("*.yaml"))
        assert len(written) == 1
        # And a plain, headless load sees it — the property the whole design rests on.
        loaded = Declarations.load(config)
        assert loaded.values_for("value", doc_type="inventory") == {"currency": "CNY"}

    def test_the_response_says_what_the_declaration_changed(self, client):
        batch_id = _upload(client).json()["landed"][0]["batch_id"]
        body = client.post(f"/batches/{batch_id}/declarations",
                           json={"scope": "value", "field": "currency", "value": "CNY",
                                 "by": "jfanhon", "reason": "plant books in CNY"}).json()
        changed = {c["field"]: c for c in body["changed"]}
        assert "currency" in changed
        assert changed["currency"]["from"]["source"] == "absent"
        assert changed["currency"]["to"]["source"] == "default"

    def test_a_mapping_declaration_moves_the_field_to_the_declared_column(self, client):
        batch_id = _upload(client).json()["landed"][0]["batch_id"]
        body = client.post(f"/batches/{batch_id}/declarations",
                           json={"scope": "mapping", "field": "sku",
                                 "value": "Base Unit", "by": "jfanhon",
                                 "reason": "worked example, not a real correction"}).json()
        sku = {f["field"]: f for f in body["resolution"]["fields"]}["sku"]
        assert sku["source"] == "declared"
        assert sku["column"] == "Base Unit"
        changed = {c["field"] for c in body["changed"]}
        assert "sku" in changed

    def test_a_declaration_naming_a_column_the_file_lacks_is_not_reported_as_applied(
            self, client):
        """
        Believing a mapping is corrected when it is not is worse than not having tried,
        because the looking stops. This labelled the routed column `declared` and said
        nothing — found by a test that used a column name the sample does not have.
        """
        batch_id = _upload(client).json()["landed"][0]["batch_id"]
        body = client.post(f"/batches/{batch_id}/declarations",
                           json={"scope": "mapping", "field": "sku",
                                 "value": "Description", "by": "jfanhon",
                                 "reason": "names a column this export does not have"}).json()
        resolution = body["resolution"]
        sku = {f["field"]: f for f in resolution["fields"]}["sku"]
        assert sku["source"] == "mapped"
        assert sku["column"] == "Item Code"
        assert resolution["ignored_declarations"] == [
            {"field": "sku", "column": "Description",
             "reason": "the file has no such column"}]
        assert body["changed"] == []

    def test_a_scope_that_does_not_belong_to_one_batch_is_refused(self, client):
        batch_id = _upload(client).json()["landed"][0]["batch_id"]
        response = client.post(f"/batches/{batch_id}/declarations",
                               json={"scope": "parameter", "field": "min_order_qty",
                                     "value": 500, "by": "j", "reason": "x"})
        assert response.status_code == 400


class TestWithdrawingABatch:

    def test_it_needs_a_reason_and_a_name(self, client):
        batch_id = _upload(client).json()["landed"][0]["batch_id"]
        assert client.post(f"/batches/{batch_id}/void", json={}).status_code == 400

    def test_a_voided_batch_says_so_in_the_listing(self, client):
        batch_id = _upload(client).json()["landed"][0]["batch_id"]
        response = client.post(f"/batches/{batch_id}/void",
                               json={"reason": "wrong date range", "by": "jfanhon"})
        assert response.status_code == 200
        listed = {b["batch_id"]: b for b in client.get("/batches").json()}
        assert listed[batch_id]["status"] == "void"


class TestWhatIsNotBuiltSaysSo:

    @pytest.mark.parametrize("path", ["/facts/inventory", "/runs"])
    def test_it_is_501_naming_what_is_missing_not_an_empty_list(self, client, path):
        """
        An empty list would read as "no data", which is a different and worse claim
        than "there is no query engine in front of this yet".
        """
        response = client.get(path)
        assert response.status_code == 501
        assert "as-of" in response.json()["detail"]


class TestABlankTemplateSaysWhatIsWrongWithIt:

    def test_it_names_the_template_and_the_next_action(self, client, tmp_path):
        from inventory_planning.ingest.templates import emit

        blank = emit("substitution", tmp_path)
        with open(blank, "rb") as fh:
            response = client.post("/uploads", files={"file": (blank.name, fh.read())})
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "substitution template with no rows" in detail
        assert "`data` sheet" in detail
