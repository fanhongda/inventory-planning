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

    def test_the_run_registry_is_501_naming_what_is_missing_not_an_empty_list(
            self, client):
        """
        An empty list would read as "no runs", which is a different and worse claim
        than "nothing reads the registry back across runs yet".
        """
        response = client.get("/runs")
        assert response.status_code == 501
        assert "run registry" in response.json()["detail"]


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


class TestTheReviewScreenIsServedWithoutShadowingTheApi:

    def test_the_page_and_its_module_load(self, client):
        assert "Import review" in client.get("/").text
        assert client.get("/app.js").status_code == 200

    def test_routes_registered_before_the_mount_still_win(self, client):
        """
        Starlette matches in registration order and a mount at "/" swallows everything
        after it. The mount is added last for that reason, and this is what says so.
        """
        assert client.get("/health").json()["ok"] is True
        assert client.get("/contracts").status_code == 200
        assert client.get("/runs").status_code == 501


class TestTheSummaryIsTheQuestionAReaderCanAnswer:

    def test_it_carries_the_totals_and_what_they_rest_on(self, client):
        batch_id = _upload(client).json()["landed"][0]["batch_id"]
        body = client.get(f"/batches/{batch_id}/summary").json()
        doc = body["documents"][0]
        assert doc["doc_type"] == "inventory"
        assert doc["rows"] == 10 and doc["skus"] == 10
        totals = {r["column"]: r["total"] for r in doc["readings"] if r["kind"] == "qty"}
        assert totals["qty_on_hand"] == 2125.0
        assert body["resting_on"]["resting_on"][0]["field"] == "currency"

    def test_a_declaration_is_attributed_in_the_exposure_list(self, client):
        """
        `values_for` reduces an override to its value and drops who asserted it, which
        rendered on screen as "someone declared".
        """
        batch_id = _upload(client).json()["landed"][0]["batch_id"]
        client.post(f"/batches/{batch_id}/declarations",
                    json={"scope": "value", "field": "currency", "value": "CNY",
                          "by": "jfanhon", "reason": "plant books in CNY"})
        resting = client.get(f"/batches/{batch_id}/summary").json()["resting_on"]
        item = resting["resting_on"][0]
        assert item["basis"] == "declared"
        assert item["by"] == "jfanhon"
        assert item["reason"] == "plant books in CNY"

    def test_re_reading_a_batch_classifies_it_again_rather_than_trusting_the_first_call(
            self, client):
        """
        Handing the landed doc type back as a hint pins the first decision: the
        classifier never runs again and every re-read reports the type as stated, at a
        flat 100%, where it should report what it measured.
        """
        batch_id = _upload(client).json()["landed"][0]["batch_id"]
        body = client.get(f"/batches/{batch_id}/resolution").json()
        assert body["confidence_basis"] == "classification"
        assert body["stated"] is False
        assert body["confidence"] < 1.0


class TestTheChecklistOfWhatIsStillMissing:

    def test_an_empty_store_lists_everything_as_outstanding(self, client):
        body = client.get("/requirements").json()
        assert body["can_run"] is False
        assert {"demand_signal", "position_signal"} <= set(body["missing_required"])
        required = [d for d in body["documents"] if d["required"]]
        assert required, "a run has required documents"
        for doc in required:
            assert doc["landed"] is None
            assert all(f["present"] is False for f in doc["fields"])
            assert doc["fields"], f"{doc['doc_type']} should name its core fields"

    def test_an_upload_ticks_its_document_and_its_fields(self, client):
        _upload(client)
        body = client.get("/requirements").json()
        assert "position_signal" not in body["missing_required"]
        inventory = [d for d in body["documents"] if d["doc_type"] == "inventory"][0]
        assert inventory["landed"]["rows"] == 10
        assert {f["field"] for f in inventory["fields"]} == {"sku", "qty_on_hand"}
        assert all(f["present"] for f in inventory["fields"])

    def test_alternatives_are_reported_as_alternatives_not_as_two_gaps(self, client):
        """
        `item_dimension` takes either master. Listing both as missing would ask for a
        file the run does not need.
        """
        with open("sample_data/item_master.csv", "rb") as fh:
            client.post("/uploads",
                        files={"file": ("item_master.csv", fh.read(), "text/csv")})
        body = client.get("/requirements").json()
        assert "item_dimension" not in body["missing_required"]
        item_dimension = [c for c in body["capabilities"]
                          if c["name"] == "item_dimension"][0]
        assert item_dimension["satisfied"] is True
        assert len(item_dimension["suppliers"]) > 1

    def test_it_is_recomputed_from_the_store_not_accumulated(self, client):
        """
        A checklist that remembers what it was told drifts from the store the moment a
        batch is voided — and being a checklist, a person trusts it over the store.
        """
        batch_id = _upload(client).json()["landed"][0]["batch_id"]
        assert "position_signal" not in client.get("/requirements").json()["missing_required"]

        client.post(f"/batches/{batch_id}/void",
                    json={"reason": "wrong month", "by": "jfanhon"})
        body = client.get("/requirements").json()
        assert "position_signal" in body["missing_required"]
        assert [d for d in body["documents"]
                if d["doc_type"] == "inventory"][0]["landed"] is None

    def test_a_document_that_cannot_back_what_it_declares_says_which(self, client,
                                                                     tmp_path):
        """
        A purchase history with no goods-receipt date is a real record of ordering
        behaviour and no record of lead time at all. The checklist has to show the
        capability as still outstanding, or the item-master fallback looks unnecessary.
        """
        path = tmp_path / "po_history.csv"
        path.write_text(
            "PO Number,Line,Item Code,PO Qty,PO Date,Unit Price\n"
            "PO-1,10,SKU-001,100,2024-01-05,12.5\n"
            "PO-2,10,SKU-002,50,2024-02-05,8.0\n", encoding="utf-8")
        with open(path, "rb") as fh:
            client.post("/uploads", files={"file": (path.name, fh.read(), "text/csv")})

        lead_time = [c for c in client.get("/requirements").json()["capabilities"]
                     if c["name"] == "lead_time_signal"][0]
        assert lead_time["satisfied"] is False
        assert "po_history" in lead_time["withheld_by"]
        assert lead_time["fallback"]


class TestPromotingAndReadingFacts:

    def _storable(self, client):
        """
        Upload, then declare the key part the sample export does not carry.

        The sample has no plant column, and `inventory` is keyed on sku + location_id.
        That is not incidental to these tests: it is the ordinary shape of a
        single-plant export, and the declaration is the intended remedy.
        """
        batch_id = _upload(client).json()["landed"][0]["batch_id"]
        client.post(f"/batches/{batch_id}/declarations",
                    json={"scope": "value", "field": "location_id", "value": "DC-01",
                          "by": "jfanhon",
                          "reason": "single-plant export; the DC is DC-01"})
        return batch_id

    def _promote(self, client, batch_id, valid_time="2024-07-01"):
        return client.post(f"/batches/{batch_id}/promote",
                           json={"valid_time": valid_time, "by": "jfanhon"})

    def test_a_promotion_needs_the_date_the_data_describes(self, client):
        batch_id = self._storable(client)
        response = client.post(f"/batches/{batch_id}/promote", json={"by": "jfanhon"})
        assert response.status_code == 400
        assert "valid_time" in response.json()["detail"]

    def test_a_batch_that_cannot_be_keyed_is_refused_with_the_remedy(self, client):
        """
        `KeyStatus.storable` was defined as the gate a fact store applies and nothing
        applied it. Storing an unkeyable batch is not a smaller version of storing a
        keyable one — a later correction lands as a second row, both are current, and
        every as-of read of the document then fails or double-counts.
        """
        batch_id = _upload(client).json()["landed"][0]["batch_id"]
        response = self._promote(client, batch_id)
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "location_id" in detail and "scope `value`" in detail

    def test_the_fact_batch_keeps_the_landing_batch_id(self, client):
        """`(batch_id, row_no)` on a fact is supposed to resolve to the verbatim row."""
        batch_id = self._storable(client)
        body = self._promote(client, batch_id).json()
        assert body["batch_id"] == batch_id
        assert body["rows"] == 10
        assert client.get(f"/batches/{batch_id}").json()["status"] == "promoted"

    def test_a_batch_already_promoted_is_not_promoted_again(self, client):
        batch_id = self._storable(client)
        self._promote(client, batch_id)
        again = self._promote(client, batch_id)
        assert again.status_code == 409
        assert "not landed" in again.json()["detail"]

    def test_the_same_file_uploaded_twice_is_stored_once(self, client):
        """Same bytes, same parameters, same as-of date: a second batch would be a
        second copy of one observation, and every quantity would double."""
        first = self._storable(client)
        self._promote(client, first)
        second = _upload(client).json()["landed"][0]["batch_id"]
        again = self._promote(client, second)
        assert again.status_code == 409
        assert "already in the store" in again.json()["detail"]

    def test_two_different_files_do_not_collide_on_the_same_as_of_date(self, client,
                                                                       tmp_path):
        """
        The content key is (source bytes, parameters, as-of). Landing was not recording
        the source hash, so it was (nothing, parameters, as-of) and a second, genuinely
        different export promoted at the same date was dropped as a duplicate.
        """
        self._promote(client, self._storable(client))

        other = tmp_path / "inventory_dc2.csv"
        other.write_text("Item Code,On Hand,In Transit,Base Unit,Report Date\n"
                         "SKU-900,7,0,EA,2024-07-01\n", encoding="utf-8")
        batch_id = _upload(client, other).json()["landed"][0]["batch_id"]
        client.post(f"/batches/{batch_id}/declarations",
                    json={"scope": "value", "field": "location_id", "value": "DC-02",
                          "by": "jfanhon", "reason": "second plant, same report"})
        assert self._promote(client, batch_id).status_code == 200
        assert len(client.get("/facts").json()) == 1
        assert client.get("/facts/inventory").json()["batches"].__len__() == 2

    def test_facts_come_back_as_of_a_moment(self, client):
        batch_id = self._storable(client)
        self._promote(client, batch_id)

        body = client.get("/facts/inventory").json()
        assert body["mode"] == "current"
        assert len(body["rows"]) == 10
        assert body["batches"][0]["valid_time"] == "2024-07-01"
        assert body["carried_forward"]["carried"] == 0

        earlier = client.get("/facts/inventory", params={"as_of": "2024-01-01"}).json()
        assert earlier["rows"] == []
        assert "no batch matches" in earlier["selection"]

    def test_the_three_readings_are_named_and_history_is_not_a_position(self, client):
        batch_id = self._storable(client)
        self._promote(client, batch_id)
        for mode in ("current", "latest", "history"):
            body = client.get("/facts/inventory", params={"mode": mode}).json()
            assert body["mode"] == mode and len(body["rows"]) == 10
        assert client.get("/facts/inventory",
                          params={"mode": "guess"}).status_code == 422

    def test_a_filter_narrows_it(self, client):
        batch_id = self._storable(client)
        self._promote(client, batch_id)
        body = client.get("/facts/inventory", params={"sku": "SKU-003"}).json()
        assert len(body["rows"]) == 1
        assert body["rows"][0]["sku"] == "SKU-003"

    def test_the_index_lists_what_the_store_holds(self, client):
        assert client.get("/facts").json() == []
        batch_id = self._storable(client)
        self._promote(client, batch_id)
        listed = client.get("/facts").json()
        assert listed[0]["doc_type"] == "inventory" and listed[0]["batches"] == 1

    def test_a_voided_batch_cannot_be_promoted(self, client):
        batch_id = self._storable(client)
        client.post(f"/batches/{batch_id}/void",
                    json={"reason": "wrong month", "by": "jfanhon"})
        assert self._promote(client, batch_id).status_code == 409
