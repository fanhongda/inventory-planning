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
        assert client.get("/runs").json()["runs"] == []


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


class TestOneBatchBeforeAndAfterTheAdapter:

    def test_the_canonical_rows_are_served_beside_the_verbatim_ones(self, client):
        """
        Reading the two together is how a person without the vocabulary to adjudicate a
        mapping still finds a mis-mapped column: the value is visibly the wrong kind of
        thing under a name that expects another.
        """
        batch_id = _upload(client).json()["landed"][0]["batch_id"]
        raw = client.get(f"/batches/{batch_id}/rows", params={"limit": 3}).json()
        canonical = client.get(f"/batches/{batch_id}/canonical",
                               params={"limit": 3}).json()

        assert canonical["doc_type"] == "inventory"
        assert canonical["total"] == raw["total"] == 10
        assert len(canonical["rows"]) == 3
        # The point of the pairing: the same row under two vocabularies.
        assert "Item Code" in raw["columns"] and "sku" in canonical["columns"]
        assert canonical["rows"][0]["sku"] == raw["rows"][0]["Item Code"]

    def test_it_reflects_a_declaration_without_a_re_upload(self, client):
        batch_id = _upload(client).json()["landed"][0]["batch_id"]
        assert "location_id" not in client.get(
            f"/batches/{batch_id}/canonical").json()["columns"]

        client.post(f"/batches/{batch_id}/declarations",
                    json={"scope": "value", "field": "location_id", "value": "DC-01",
                          "by": "jfanhon", "reason": "single-plant export"})
        body = client.get(f"/batches/{batch_id}/canonical").json()
        assert body["rows"][0]["location_id"] == "DC-01"

    def test_a_missing_batch_is_404(self, client):
        assert client.get("/batches/nope/canonical").status_code == 404


class TestTheClientIsServedAsModules:

    def test_every_module_the_shell_imports_is_there(self, client):
        """
        No build step means no bundler to notice a missing file: a bad import path is a
        blank page at runtime and nothing at test time.
        """
        shell = client.get("/app.js")
        assert shell.status_code == 200
        for module in ("/ui.js", "/review.js", "/browse.js"):
            assert module in shell.text
            assert client.get(module).status_code == 200

    def test_the_page_hosts_both_screens(self, client):
        page = client.get("/").text
        assert 'id="screen-review"' in page and 'id="screen-browse"' in page


class TestPolicyIsShownAndNotEdited:

    @pytest.fixture
    def client(self, workspace):
        """
        The shipped config, copied in. These assertions are about the real parameter
        file — the rules it declares and the conventions it fixes — so testing against
        an empty directory would test the 404 and nothing else.
        """
        import shutil

        config, store = workspace
        for name in ("planning_parameters.md", "node_config.json", "fx_rates.json"):
            shutil.copy(Path("config") / name, config / name)
        return TestClient(create_app(config_dir=config, store_root=store))

    def test_the_rules_come_back_with_the_reason_each_one_exists(self, client):
        body = client.get("/policy").json()
        assert body["source"].endswith("planning_parameters.md")
        assert body["rules"], "the shipped parameter file declares rules"
        for rule in body["rules"]:
            assert rule["rule_id"] and rule["scope"] and rule["sets"]
            assert rule["rationale"], "a rule without a rationale cannot be judged later"

    def test_the_conventions_that_change_every_figure_are_shown(self, client):
        body = client.get("/policy").json()
        assert "safety_stock_exposure" in body["conventions"]
        assert "days_per_year" in body["conventions"]

    def test_with_no_run_behind_them_it_says_so_rather_than_showing_zeros(self, client):
        """
        A rule's reach is a question about a frame of SKUs and there is no frame here,
        so it takes a run to answer. An empty count would read as "this rule matched
        nothing", which is a finding rather than a silence.
        """
        body = client.get("/policy").json()
        assert body["hits"] is None
        assert "takes a run" in body["note"]

    def test_no_endpoint_writes_policy(self, client):
        for method in ("post", "put", "patch", "delete"):
            assert getattr(client, method)("/policy").status_code in (404, 405)

    def test_macro_reads_currencies_through_the_fx_table(self, client):
        """
        Parsing `fx_rates.json` here reported its top-level keys — `rates`,
        `reporting_currency` — as currency codes. A second implementation of a read the
        package already does is the mistake this interface is meant to make impossible.
        """
        body = client.get("/policy/macro").json()
        settings = {s["name"]: s for s in body["settings"]}
        assert settings["reporting_currency"]["value"] == "USD"
        assert "USD" in settings["fx_currencies"]["value"]
        assert "rates" not in settings["fx_currencies"]["value"]

    def test_macro_lists_only_settings_the_engine_reads(self, client):
        """No working-day switch: the distinction does not exist in the pipeline."""
        names = {s["name"] for s in client.get("/policy/macro").json()["settings"]}
        assert "days_per_year" in names
        assert not {n for n in names if "working" in n or "growth" in n}


class TestTheQualityGate:
    """
    The checkpoint that catches this pipeline's actual failure: not a crash, but a
    complete report built on a join that matched nothing. It is visible at intake and
    invisible in the report it would go on to write, which is why it belongs in front of
    the person who just uploaded the file rather than in a run log an hour later.
    """

    @pytest.fixture
    def client(self, workspace, tmp_path):
        import shutil

        config, store = workspace
        for name in ("quality_gates.json", "node_config.json", "fx_rates.json"):
            shutil.copy(Path("config") / name, config / name)
        return TestClient(create_app(config_dir=config, store_root=store,
                                     output_dir=tmp_path / "out"))

    @pytest.fixture
    def disagreeing(self, tmp_path):
        """An inventory export whose item numbers meet nothing else — the real case."""
        import pandas as pd

        frame = pd.read_csv(SAMPLE)
        frame["Item Code"] = [f"ZZ-{9000 + i}" for i in range(len(frame))]
        path = tmp_path / "inventory.csv"
        frame.to_csv(path, index=False)
        return path

    def test_with_nothing_landed_it_says_so_rather_than_passing(self, client):
        """
        A clean result on an empty store would read as "your data is fine". The gate
        compares documents against each other, and one document cannot disagree with
        itself.
        """
        body = client.get("/gates").json()
        assert body["ran"] is False
        assert body["passed"] is None
        assert "nothing to check" in body["note"]

    def test_a_clean_set_passes_with_no_findings(self, client):
        for name in ("inventory", "sales_history", "item_master"):
            _upload(client, Path(f"sample_data/{name}.csv"))
        body = client.get("/gates").json()
        assert body["ran"] is True and body["passed"] is True
        assert body["findings"] == []

    def test_an_item_number_that_meets_nothing_blocks(self, client, disagreeing):
        _upload(client, disagreeing)
        _upload(client, Path("sample_data/sales_history.csv"))
        _upload(client, Path("sample_data/item_master.csv"))

        body = client.get("/gates").json()
        assert body["passed"] is False
        assert body["counts"]["block"] == 1
        finding, = body["findings"]
        assert finding["check"] == "sku_agreement"
        assert finding["severity"] == "block"
        assert finding["doc_type"] == "inventory"
        # what / why / fix, all three: a finding that cannot say what to do about it is
        # a finding that should not stop a run.
        assert finding["what"] and finding["why"] and finding["fix"]

    def test_a_clean_answer_never_claims_the_run_will_pass(self, client):
        """
        Three of the four gates need a time series, a forecast and a position, none of
        which exist before the run. "Nothing at intake stops this" is the claim that can
        be made here; the page has the other three by name so it is not left to inference.
        """
        for name in ("inventory", "sales_history", "item_master"):
            _upload(client, Path(f"sample_data/{name}.csv"))
        body = client.get("/gates").json()
        assert {g["stage"] for g in body["later_stages"]} == {"demand", "forecast", "plan"}
        assert all(g["needs"] and g["checks"] for g in body["later_stages"])


class TestWaivingOneCheck:

    @pytest.fixture
    def client(self, workspace, tmp_path):
        import shutil

        config, store = workspace
        for name in ("quality_gates.json", "node_config.json", "fx_rates.json"):
            shutil.copy(Path("config") / name, config / name)
        return TestClient(create_app(config_dir=config, store_root=store,
                                     output_dir=tmp_path / "out"))

    @pytest.fixture
    def blocked(self, client, tmp_path):
        import pandas as pd

        frame = pd.read_csv(SAMPLE)
        frame["Item Code"] = [f"ZZ-{9000 + i}" for i in range(len(frame))]
        path = tmp_path / "inventory.csv"
        frame.to_csv(path, index=False)
        _upload(client, path)
        _upload(client, Path("sample_data/sales_history.csv"))
        _upload(client, Path("sample_data/item_master.csv"))
        return client

    def _waive(self, client, **body):
        return client.post("/gates/sku_agreement/waivers",
                           json={"doc_type": "inventory", **body})

    def test_a_waiver_downgrades_the_finding_and_leaves_it_visible(self, blocked):
        assert self._waive(blocked, expires="2027-12-31", by="jfanhon",
                           reason="this DC stocks spares nothing else sells"
                           ).status_code == 200

        body = blocked.get("/gates").json()
        assert body["passed"] is True
        finding, = body["findings"]
        assert finding["severity"] == "warn"
        assert finding["waived"] is True
        assert finding["waived_by"] == "jfanhon"
        assert finding["waived_until"] == "2027-12-31"

    def test_it_lands_as_an_ordinary_declaration_a_headless_run_reads(
            self, blocked, workspace):
        """
        The governing rule. A waiver made by clicking has to be the same statement, in
        the same place, as one typed into the file — otherwise the run and the screen
        disagree about what has been declared.
        """
        from inventory_planning.quality.gates import BLOCK, Finding, GateReport

        self._waive(blocked, expires="2027-12-31", by="jfanhon", reason="disjoint")
        loaded = Declarations.load(workspace[0])
        report = loaded.waive(GateReport("intake", [Finding(
            stage="intake", check="sku_agreement", severity=BLOCK,
            what="w", why="y", fix="f", evidence={"doc_type": "inventory"})]))
        assert not report.blocking

    def test_a_waiver_must_expire(self, blocked):
        response = self._waive(blocked, by="jfanhon", reason="disjoint")
        assert response.status_code == 400
        assert "permanently disabled check" in response.json()["detail"]

    def test_an_expiry_already_past_is_refused_rather_than_written(self, blocked):
        """It would write cleanly, apply to nothing, and read as a waiver in force."""
        response = self._waive(blocked, expires="2020-01-01", by="jfanhon",
                               reason="disjoint")
        assert response.status_code == 400
        assert "in the past" in response.json()["detail"]
        assert blocked.get("/gates").json()["passed"] is False

    def test_it_must_say_why_and_who(self, blocked):
        assert self._waive(blocked, expires="2027-12-31", by="jfanhon"
                           ).status_code == 400
        assert self._waive(blocked, expires="2027-12-31", reason="disjoint"
                           ).status_code == 400

    def test_a_waiver_on_one_document_leaves_the_others_checked(self, blocked,
                                                                workspace):
        """
        The whole reason this is not `allow_degraded`: waiving the agreement finding on
        a stock snapshot must not wave through an open PO mapped to a money column. The
        scope has to be in the file, because the file is what the run reads.
        """
        from inventory_planning.quality.gates import BLOCK, Finding, GateReport

        self._waive(blocked, expires="2027-12-31", by="jfanhon", reason="disjoint")
        waiver, = Declarations.load(workspace[0]).waivers
        assert waiver.doc_type == "inventory"

        elsewhere = GateReport("intake", [Finding(
            stage="intake", check="sku_agreement", severity=BLOCK,
            what="w", why="y", fix="f", evidence={"doc_type": "open_po"})])
        assert Declarations.load(workspace[0]).waive(elsewhere).blocking

    def test_a_malformed_expiry_is_a_400_not_a_500(self, blocked):
        response = self._waive(blocked, expires="next tuesday", by="jfanhon",
                               reason="disjoint")
        assert response.status_code == 400


class TestEditingAScalar:
    """
    Two requests, not a stored proposal: the first asks what a change would do, the
    second approves the diff it was shown. Nothing is kept on the server between them —
    INTERFACE.md rules out interface-only state — so what ties them together is the
    digest of the file the diff was made against.
    """

    @pytest.fixture
    def client(self, workspace, tmp_path):
        import shutil

        config, store = workspace
        for name in ("planning_parameters.md", "node_config.json", "fx_rates.json"):
            shutil.copy(Path("config") / name, config / name)
        return TestClient(create_app(config_dir=config, store_root=store,
                                     output_dir=tmp_path / "out"))

    def _put(self, client, **body):
        return client.put("/policy/macro", json=body)

    def test_a_proposal_returns_the_diff_and_writes_nothing(self, client, workspace):
        before = (workspace[0] / "planning_parameters.md").read_text(encoding="utf-8")
        body = self._put(client, name="days_per_year", value=250).json()

        assert body["applied"] is False
        assert body["from"] == 365 and body["to"] == 250
        assert "-days_per_year: 365" in body["diff"]
        assert "+days_per_year: 250" in body["diff"]
        assert (workspace[0] / "planning_parameters.md").read_text(
            encoding="utf-8") == before

    def test_the_proposal_says_what_the_change_would_move(self, client):
        """
        These settings cost wildly different amounts. `location_name` is a label;
        `cycle_stock_basis` halves or doubles the cycle stock in every figure, and a
        form that presented them identically would be hiding that.
        """
        body = self._put(client, name="cycle_stock_basis", value="average").json()
        assert "cycle stock" in body["impact"]
        label = self._put(client, name="location_name", value="DC North").json()
        assert "no figure moves" in label["impact"]

    def test_approving_the_diff_applies_it(self, client, workspace):
        proposal = self._put(client, name="days_per_year", value=250).json()
        body = self._put(client, name="days_per_year", value=250, apply=True,
                         reason="finance counts working days", by="jfanhon",
                         basis=proposal["basis"]).json()

        assert body["applied"] is True
        assert "days_per_year: 250" in (
            workspace[0] / "planning_parameters.md").read_text(encoding="utf-8")
        assert proposal["basis"] != body["digest"]

    def test_a_file_that_moved_since_the_diff_is_a_refusal(self, client, workspace):
        proposal = self._put(client, name="days_per_year", value=250).json()
        path = workspace[0] / "planning_parameters.md"
        path.write_text(path.read_text(encoding="utf-8").replace(
            "transit_share_of_lt: 0.45", "transit_share_of_lt: 0.5"), encoding="utf-8")

        response = self._put(client, name="days_per_year", value=250, apply=True,
                             reason="r", by="jfanhon", basis=proposal["basis"])
        assert response.status_code == 400
        assert "has changed since that diff" in response.json()["detail"]
        assert "days_per_year: 365" in path.read_text(encoding="utf-8")

    def test_an_apply_without_a_reason_or_a_name_is_refused(self, client):
        proposal = self._put(client, name="days_per_year", value=250).json()
        for missing in ({"by": "jfanhon"}, {"reason": "because"}):
            response = self._put(client, name="days_per_year", value=250, apply=True,
                                 basis=proposal["basis"], **missing)
            assert response.status_code == 400

    def test_a_value_the_engine_would_ignore_is_refused_with_the_alternatives(
            self, client):
        response = self._put(client, name="pipeline_basis", value="incoterm_awre")
        assert response.status_code == 400
        assert "incoterm_aware" in response.json()["detail"]

    def test_a_setting_the_pipeline_does_not_read_is_not_editable(self, client):
        assert self._put(client, name="echelon_level", value=2).status_code == 400

    def test_the_listing_says_which_settings_a_form_may_write(self, client):
        settings = {s["name"]: s for s in client.get("/policy/macro").json()["settings"]}
        assert settings["days_per_year"]["editable"] is True
        assert settings["cycle_stock_basis"]["choices"] == ["peak", "average"]
        # Derived readings are not settings: there is nothing in a file to write back.
        assert settings["fx_currencies"]["editable"] is False

    def test_the_change_is_listed_afterwards_with_its_reason(self, client):
        proposal = self._put(client, name="days_per_year", value=250).json()
        self._put(client, name="days_per_year", value=250, apply=True,
                  reason="finance counts working days", by="jfanhon",
                  basis=proposal["basis"])

        body = client.get("/policy/macro").json()
        entry, = body["changes"]
        assert entry["by"] == "jfanhon" and entry["to"] == 250
        assert entry["reason"] == "finance counts working days"
        assert {s["name"]: s["value"] for s in body["settings"]}["days_per_year"] == 250


class TestWhatEachRuleReached:
    """
    The counts come from a run, are labelled with the run they came from, and are shown
    only against the rules that produced them. A count carried over from a different
    rule set would be attached to rules that never produced it, which is worse than the
    blank it replaced.
    """

    @pytest.fixture
    def rules_file(self, workspace):
        import shutil

        config, _ = workspace
        for name in ("planning_parameters.md", "node_config.json", "fx_rates.json"):
            shutil.copy(Path("config") / name, config / name)
        return config / "planning_parameters.md"

    def _run_under(self, output_dir, rules_file, hits):
        from inventory_planning.provenance import RunManifest, RunRegistry

        manifest = RunManifest.begin(output_dir=output_dir, policy_file=rules_file)
        manifest.record_rules([h["rule_id"] for h in hits])
        manifest.record_rule_hits([_FakeHit(h) for h in hits])
        RunRegistry(output_dir).save(manifest)
        return manifest

    def _client(self, workspace, output_dir):
        config, store = workspace
        return TestClient(create_app(config_dir=config, store_root=store,
                                     output_dir=output_dir))

    def test_the_counts_come_back_keyed_by_rule_and_named_with_their_run(
            self, workspace, rules_file, tmp_path):
        out = tmp_path / "out"
        run = self._run_under(out, rules_file, [
            {"rule_id": "R-001", "matched": 12, "effective": 9},
            {"rule_id": "R-002", "matched": 3, "effective": 3},
        ])
        body = self._client(workspace, out).get("/policy").json()

        assert body["hits"]["run_id"] == run.run_id
        assert body["hits"]["rules"]["R-001"]["matched"] == 12
        assert body["hits"]["rules"]["R-001"]["effective"] == 9
        assert run.run_id in body["note"]

    def test_a_rule_edited_since_the_last_run_shows_no_counts_at_all(
            self, workspace, rules_file, tmp_path):
        """
        Not stale ones, and not zeros. The digest of the file is what the run is found
        by, so an edit of any kind — including one that leaves every rule id in place —
        detaches the counts from the rules on screen.
        """
        out = tmp_path / "out"
        self._run_under(out, rules_file, [{"rule_id": "R-001", "matched": 12,
                                           "effective": 9}])
        rules_file.write_text(rules_file.read_text(encoding="utf-8")
                              + "\n<!-- a comment -->\n", encoding="utf-8")
        body = self._client(workspace, out).get("/policy").json()

        assert body["hits"] is None
        assert "edited since the last one" in body["note"]

    def test_a_rule_the_run_never_reported_is_absent_rather_than_zero(
            self, workspace, rules_file, tmp_path):
        """
        The join is by rule id and the caller holds the rules, so a rule with nothing
        recorded against it is visibly blank instead of a row that quietly reports
        somebody else's number.
        """
        out = tmp_path / "out"
        self._run_under(out, rules_file, [{"rule_id": "R-001", "matched": 12,
                                           "effective": 9}])
        body = self._client(workspace, out).get("/policy").json()

        ids = {r["rule_id"] for r in body["rules"]}
        assert "R-002" in ids and "R-002" not in body["hits"]["rules"]

    def test_a_run_that_kept_no_reach_is_not_reported_as_an_edit(
            self, workspace, rules_file, tmp_path):
        """
        Every run from before the reach was retained lands here — on the first look at
        an existing output directory, all of them. Saying the file has been edited
        would send someone hunting for a change to a file nobody has touched.
        """
        from inventory_planning.provenance import RunManifest, RunRegistry

        out = tmp_path / "out"
        manifest = RunManifest.begin(output_dir=out, policy_file=rules_file)
        manifest.record_rules(["R-001"])
        RunRegistry(out).save(manifest)
        body = self._client(workspace, out).get("/policy").json()

        assert body["hits"] is None
        assert manifest.run_id in body["note"]
        assert "recorded no per-rule reach" in body["note"]
        assert "edited" not in body["note"]

    def test_a_miss_inside_a_bounded_scan_does_not_claim_the_file_changed(
            self, workspace, rules_file, tmp_path, monkeypatch):
        """
        The search reads back a bounded number of manifests. A matching run older than
        that bound is not found, and "not found" is not "does not exist" — the claim
        the message may make is only the one the search established.
        """
        from inventory_planning.api import app as app_module

        monkeypatch.setattr(app_module, "_REACH_SCAN", 2)
        out = tmp_path / "out"
        self._run_under(out, rules_file, [{"rule_id": "R-001", "matched": 12,
                                           "effective": 9}])
        other = tmp_path / "other_rules.md"
        other.write_text(rules_file.read_text(encoding="utf-8") + "\n<!-- x -->\n",
                         encoding="utf-8")
        for _ in range(3):
            self._run_under(out, other, [{"rule_id": "R-001", "matched": 1,
                                          "effective": 1}])
        body = self._client(workspace, out).get("/policy").json()

        assert body["hits"] is None
        assert "2 most recent runs" in body["note"]
        assert "edited" not in body["note"]

    def test_with_every_run_searched_and_none_matching_it_does_say_edited(
            self, workspace, rules_file, tmp_path):
        """The one case where the claim about history is one the search can make."""
        out = tmp_path / "out"
        self._run_under(out, rules_file, [{"rule_id": "R-001", "matched": 12,
                                           "effective": 9}])
        rules_file.write_text(rules_file.read_text(encoding="utf-8")
                              + "\n<!-- a comment -->\n", encoding="utf-8")
        body = self._client(workspace, out).get("/policy").json()

        assert "Every recorded run was searched" in body["note"]
        assert "edited since the last one" in body["note"]


class _FakeHit:
    """`policy.parameters.RuleHit` as `record_rule_hits` reads it — duck-typed."""

    class _Rule:
        def __init__(self, rule_id):
            self.rule_id, self.name, self.scope = rule_id, "", "x == 1"
            self.overrides = {"review_period_days": 7}

    def __init__(self, spec):
        self.rule = self._Rule(spec["rule_id"])
        self.matched = spec["matched"]
        self.effective = spec["effective"]
        self.sample_skus = ["A-1"]
        self.unavailable_columns = []
        self.overrides_earlier = {}


class TestRunsAndTheirDifferences:

    def _registry(self, tmp_path):
        from inventory_planning.provenance import RunRegistry

        registry = RunRegistry(tmp_path)
        registry.dir.mkdir(parents=True, exist_ok=True)
        return registry

    def _write(self, registry, run_id, *, inputs="I", config="C", policy="P", code="G"):
        import json

        entry = {"run_id": run_id, "run_at": f"2026-09-05T10:00:0{run_id[-1]}",
                 "input_fingerprint": inputs, "config_fingerprint": config,
                 "policy_fingerprint": policy, "git_sha": code}
        with open(registry.index_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        (registry.dir / f"{run_id}.json").write_text(json.dumps(entry), encoding="utf-8")

    @pytest.fixture
    def runs_client(self, workspace, tmp_path):
        config, store = workspace
        registry = self._registry(tmp_path / "out")
        self._write(registry, "run-1")
        self._write(registry, "run-2", policy="P2")     # parameters only
        self._write(registry, "run-3", inputs="I2", code="G2")
        return TestClient(create_app(config_dir=config, store_root=store,
                                     output_dir=tmp_path / "out"))

    def test_the_registry_is_listed_newest_first(self, runs_client):
        runs = runs_client.get("/runs").json()["runs"]
        assert [r["run_id"] for r in runs] == ["run-3", "run-2", "run-1"]

    def test_one_run_comes_back_whole(self, runs_client):
        assert runs_client.get("/runs/run-1").json()["run_id"] == "run-1"
        assert runs_client.get("/runs/nope").status_code == 404

    def test_a_parameter_only_difference_is_attributable(self, runs_client):
        body = runs_client.get("/runs/run-1/diff/run-2").json()
        assert body["basis"] == "scenario"
        assert body["moved"] == ["parameters"]
        assert "attributable to the policy change" in body["describe"]

    def test_two_axes_moving_attributes_nothing(self, runs_client):
        """
        The point of recording the basis. A comparison where facts and code both moved
        cannot be read as a policy result, and saying so is more use than a number.
        """
        body = runs_client.get("/runs/run-2/diff/run-3").json()
        assert body["basis"] == "mixed"
        assert set(body["moved"]) == {"facts", "parameters", "code"}
        assert "nothing here is attributable" in body["describe"]

    def test_an_unknown_run_is_404(self, runs_client):
        assert runs_client.get("/runs/run-1/diff/nope").status_code == 404


class TestFiltersAndTemporaryFiles:

    def test_filtering_on_a_column_the_document_lacks_is_422_not_500(self, client,
                                                                     tmp_path):
        """
        The API offers `location_id` on every doc_type and only some contracts carry
        it. This reached DuckDB, whose binder error no caller catches, and surfaced as
        Internal Server Error on ordinary input.
        """
        from inventory_planning.store.fact_store import FactStore
        import pandas as pd

        store = FactStore(client.app.state.service.store_root)
        store.write_batch(
            doc_type="sales_history",
            frame=pd.DataFrame([{"sku": "A", "ship_date": "2024-01-01", "qty": 1,
                                 "so_number": "S1", "so_line_number": "10"}]),
            valid_time="2024-07-01", source_name="s.csv", source_sha="x",
            written_by="tests")

        assert client.get("/facts/sales_history").status_code == 200
        response = client.get("/facts/sales_history", params={"location_id": "DC-01"})
        assert response.status_code == 422
        assert "location_id" in response.json()["detail"]

    def test_an_upload_leaves_no_copy_of_the_file_behind(self, client, tmp_path,
                                                         monkeypatch):
        """
        Every upload wrote the file into a fresh mkdtemp and never removed it, so a
        copy of every file ever uploaded accumulated — un-anonymised extracts among
        them, outside both the repository and the store's retention.
        """
        holding = tmp_path / "temp"
        holding.mkdir()
        monkeypatch.setenv("TMPDIR", str(holding))

        assert _upload(client).status_code == 200
        leftover = [p for p in holding.rglob("*") if p.is_file()]
        assert leftover == [], f"left behind: {leftover}"

    def test_a_template_download_cleans_up_after_itself(self, client, tmp_path,
                                                        monkeypatch):
        """
        Deleted after the response is sent rather than in a `finally`: FileResponse
        streams the file once the handler has returned.
        """
        holding = tmp_path / "temp"
        holding.mkdir()
        monkeypatch.setenv("TMPDIR", str(holding))

        response = client.get("/contracts/substitution/template")
        assert response.status_code == 200 and response.content[:2] == b"PK"
        assert [p for p in holding.rglob("*") if p.is_file()] == []

    def test_an_unnamed_upload_does_not_write_over_its_own_directory(self, client):
        response = client.post("/uploads",
                               files={"file": ("", b"a,b\n1,2\n", "text/csv")})
        assert response.status_code in (400, 422)


class TestReadingOneLayerByName:

    def _two_layers(self, client):
        from inventory_planning.store.fact_store import FactStore
        from inventory_planning.store.ledger import LAYER_PREPARED
        import pandas as pd

        store = FactStore(client.app.state.service.store_root)
        store.write_batch(doc_type="inventory",
                          frame=pd.DataFrame([{"sku": "A", "location_id": "DC-01",
                                               "qty_on_hand": 10}]),
                          valid_time="2024-06-01", source_name="old.csv",
                          source_sha="old", written_by="tests",
                          frame_layer=LAYER_PREPARED)
        store.write_batch(doc_type="inventory",
                          frame=pd.DataFrame([{"sku": "A", "location_id": "DC-01",
                                               "qty_on_hand": 15}]),
                          valid_time="2024-07-01", source_name="new.csv",
                          source_sha="new", written_by="tests")

    def test_a_mixed_read_is_refused_and_names_the_layers(self, client):
        self._two_layers(client)
        response = client.get("/facts/inventory")
        assert response.status_code == 422
        assert "`layer=canonical`" in response.json()["detail"]

    def test_naming_a_layer_reads_it(self, client):
        self._two_layers(client)
        for layer, qty in (("prepared", 10), ("canonical", 15)):
            body = client.get("/facts/inventory", params={"layer": layer}).json()
            assert body["layer"] == layer
            assert body["layers"] == [layer]
            assert body["rows"][0]["qty_on_hand"] == qty

    def test_an_unknown_layer_name_is_rejected_by_the_signature(self, client):
        assert client.get("/facts/inventory",
                          params={"layer": "whatever"}).status_code == 422
