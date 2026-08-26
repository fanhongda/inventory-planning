"""
Run identity.

The point of a manifest is not that it records things — it is that two runs can be
compared and the comparison *means* something. So most of what is pinned here is the
fingerprints: that they are stable when nothing moved, that they move when the thing
they cover moves, and that `compare` refuses to attribute a difference when more than
one axis changed.

The other half is that provenance can never fail the run it documents. A missing config
directory, a path that is not a file, an input handed over as a frame: each is recorded
as unknown, and nothing raises.
"""

import json

import pytest

from inventory_planning.provenance import (
    RunComparison, RunManifest, RunRegistry,
)


@pytest.fixture
def files(tmp_path):
    a = tmp_path / "open_po.csv"
    a.write_text("po_number,sku,open_qty\nP1,A,10\n", encoding="utf-8")
    b = tmp_path / "inventory.csv"
    b.write_text("sku,qty_on_hand\nA,5\n", encoding="utf-8")
    return a, b


def _manifest(config_dir=None, output_dir=None):
    return RunManifest.begin(config_dir=config_dir, output_dir=output_dir)


class TestRunId:
    def test_two_runs_started_together_are_still_distinct(self):
        assert _manifest().run_id != _manifest().run_id

    def test_it_sorts_chronologically(self):
        """
        To the second. Within one second the uniqueness suffix breaks the tie in no
        particular order, which is why `run_at` and not the id is what a comparison
        orders on.
        """
        first, second = _manifest(), _manifest()
        assert first.run_id.split("-")[0] <= second.run_id.split("-")[0]
        assert first.run_at <= second.run_at


class TestInputFingerprint:
    def test_the_same_files_fingerprint_the_same(self, files):
        a, b = files
        one, two = _manifest(), _manifest()
        for m in (one, two):
            m.record_input(a, doc_type="open_po")
            m.record_input(b, doc_type="inventory")
        assert one.input_fingerprint == two.input_fingerprint

    def test_the_order_they_were_read_in_does_not_matter(self, files):
        a, b = files
        one, two = _manifest(), _manifest()
        one.record_input(a, doc_type="open_po"); one.record_input(b, doc_type="inventory")
        two.record_input(b, doc_type="inventory"); two.record_input(a, doc_type="open_po")
        assert one.input_fingerprint == two.input_fingerprint

    def test_changing_a_byte_changes_it(self, files):
        a, b = files
        before = _manifest(); before.record_input(a, doc_type="open_po")
        a.write_text("po_number,sku,open_qty\nP1,A,11\n", encoding="utf-8")
        after = _manifest(); after.record_input(a, doc_type="open_po")
        assert before.input_fingerprint != after.input_fingerprint

    def test_a_file_re_read_through_a_second_entry_point_is_one_input(self, files):
        """`load_*` records the path; `load_all` then records the contract's view."""
        a, _ = files
        m = _manifest()
        m.record_input(a, doc_type="open_po")
        m.record_input(a, doc_type="open_po", rows=60, key_verdict="degraded", storable=False)
        assert len(m.inputs) == 1
        assert m.inputs[0].rows == 60
        assert m.inputs[0].sha256 is not None

    def test_an_input_with_no_file_behind_it_is_recorded_not_dropped(self):
        m = _manifest()
        m.record_input(None, doc_type="open_po")
        assert len(m.inputs) == 1
        assert m.inputs[0].sha256 is None
        assert m.input_fingerprint


class TestRecordIntake:
    """
    `IntakeResult.documents` is a dict keyed by doc_type. Iterating it yields the keys,
    so a version of this that took the documents for a sequence recorded five inputs
    named after nothing and silently lost every row count and key verdict.
    """

    @staticmethod
    def _intake(path):
        class _Frame(list):
            pass

        class _Status:
            verdict, storable = "degraded", False

        class _Report:
            key_status = _Status()

        class _Adapter:
            slug = "default__open_po/open_po.v1"

        class _Route:
            adapter = _Adapter()

        class _Doc:
            doc_type = "open_po"
            source_path = path
            frame = _Frame([1, 2, 3])
            test_report = _Report()
            route = _Route()

        class _Intake:
            documents = {"open_po": _Doc()}

        return _Intake()

    def test_it_reads_the_documents_not_their_keys(self, files):
        a, _ = files
        m = _manifest()
        m.record_intake(self._intake(a))
        assert [i.doc_type for i in m.inputs] == ["open_po"]
        assert m.inputs[0].rows == 3
        assert m.inputs[0].key_verdict == "degraded"
        assert m.inputs[0].adapter == "default__open_po/open_po.v1"

    def test_a_plain_sequence_of_documents_also_works(self, files):
        a, _ = files
        m = _manifest()
        m.record_intake(type("R", (), {"documents": [self._intake(a).documents["open_po"]]})())
        assert m.inputs[0].rows == 3


class TestConfigFingerprint:
    def test_editing_a_parameter_moves_it(self, tmp_path):
        cfg = tmp_path / "config"; cfg.mkdir()
        (cfg / "stocking_policy.json").write_text('{"a": 1}', encoding="utf-8")
        before = _manifest(config_dir=cfg)
        (cfg / "stocking_policy.json").write_text('{"a": 2}', encoding="utf-8")
        after = _manifest(config_dir=cfg)
        assert before.config_fingerprint != after.config_fingerprint

    def test_recording_rules_does_not_move_it(self, tmp_path):
        """
        It used to. The rules are parsed well after the shadow write has stamped
        batches with this value, so a fingerprint that included them was a different
        number before and after — and every batch carried one that appeared nowhere in
        the manifest.
        """
        cfg = tmp_path / "config"; cfg.mkdir()
        (cfg / "stocking_policy.json").write_text('{"a": 1}', encoding="utf-8")
        m = _manifest(config_dir=cfg)
        before = m.config_fingerprint
        m.record_rules(["R-001", "R-002"])
        assert m.config_fingerprint == before
        assert m.rule_ids == ["R-001", "R-002"]

    def test_editing_the_rules_file_still_moves_it(self, tmp_path):
        """Which is why dropping rule_ids from the hash covers nothing less."""
        cfg = tmp_path / "config"; cfg.mkdir()
        (cfg / "planning_parameters.md").write_text("rule_id: R-001\n", encoding="utf-8")
        before = _manifest(config_dir=cfg).config_fingerprint
        (cfg / "planning_parameters.md").write_text("rule_id: R-001\nrule_id: R-002\n",
                                                    encoding="utf-8")
        assert _manifest(config_dir=cfg).config_fingerprint != before

    def test_a_missing_config_directory_is_a_note_not_a_failure(self, tmp_path):
        m = _manifest(config_dir=tmp_path / "nope")
        assert m.config_files == {}
        assert any("config" in n for n in m.notes)
        assert m.config_fingerprint


class TestPolicyFingerprint:
    """
    `run_policy_analysis(parameters_file=...)` resolves the rule engine against a path
    anywhere on disk. Hashing the config directory alone made a scenario run
    indistinguishable from its baseline, and `compare` then called the pair `identical`
    — telling the reader that a real policy result was non-determinism.
    """

    def test_a_different_rules_file_moves_it(self, tmp_path):
        base = tmp_path / "planning_parameters.md"
        base.write_text("service_level: 0.95\n", encoding="utf-8")
        alt = tmp_path / "scenario_rules.md"
        alt.write_text("service_level: 0.99\n", encoding="utf-8")

        one, two = _manifest(), _manifest()
        one.record_policy_file(base)
        two.record_policy_file(alt)
        assert one.policy_fingerprint != two.policy_fingerprint

    def test_the_same_rules_file_does_not(self, tmp_path):
        rules = tmp_path / "planning_parameters.md"
        rules.write_text("service_level: 0.95\n", encoding="utf-8")
        one, two = _manifest(), _manifest()
        one.record_policy_file(rules)
        two.record_policy_file(rules)
        assert one.policy_fingerprint == two.policy_fingerprint

    def test_it_does_not_disturb_the_config_fingerprint(self, tmp_path):
        """Which is stamped on batches at intake and must not move afterwards."""
        cfg = tmp_path / "config"; cfg.mkdir()
        (cfg / "fx_rates.json").write_text('{"a": 1}', encoding="utf-8")
        rules = tmp_path / "rules.md"; rules.write_text("x\n", encoding="utf-8")
        m = _manifest(config_dir=cfg)
        before = m.config_fingerprint
        m.record_policy_file(rules)
        m.record_rules(["R-001"])
        assert m.config_fingerprint == before

    def test_an_unreadable_rules_file_is_a_note_not_a_failure(self, tmp_path):
        m = _manifest()
        m.record_policy_file(tmp_path / "gone.md")
        assert any("unfingerprinted" in n for n in m.notes)
        assert m.policy_fingerprint


class TestComparisonNamesWhatMoved:
    @staticmethod
    def _entry(inputs="i1", config="c1", sha="s1", policy="p1"):
        return {"run_id": "r", "input_fingerprint": inputs,
                "config_fingerprint": config, "policy_fingerprint": policy,
                "git_sha": sha}

    def test_nothing_moved(self):
        c = RunComparison(a=self._entry(), b=self._entry())
        assert c.basis == "identical"

    def test_only_the_parameters_moved_is_a_scenario(self):
        c = RunComparison(a=self._entry(), b=self._entry(config="c2"))
        assert c.basis == "scenario"
        assert "attributable to the policy change" in c.describe()

    def test_only_the_facts_moved_is_new_data(self):
        c = RunComparison(a=self._entry(), b=self._entry(inputs="i2"))
        assert c.basis == "new_data"
        assert "the world moving" in c.describe()

    def test_two_axes_moving_attributes_nothing(self):
        """The case a policy UI must refuse to draw a conclusion from."""
        c = RunComparison(a=self._entry(), b=self._entry(inputs="i2", config="c2"))
        assert c.basis == "mixed"
        assert "nothing here is attributable" in c.describe().lower()

    def test_a_code_change_alone_is_not_a_scenario(self):
        c = RunComparison(a=self._entry(), b=self._entry(sha="s2"))
        assert c.basis == "mixed"

    def test_a_rule_set_change_alone_is_a_scenario(self):
        """The case that read as `identical` while only the config dir was hashed."""
        c = RunComparison(a=self._entry(), b=self._entry(policy="p2"))
        assert c.basis == "scenario"
        assert not c.same_config


class TestRegistry:
    def test_a_saved_run_comes_back(self, tmp_path, files):
        a, _ = files
        m = _manifest(output_dir=tmp_path)
        m.record_input(a, doc_type="open_po")
        registry = RunRegistry(tmp_path)
        path = registry.save(m)

        assert path.exists()
        assert registry.get(m.run_id)["input_fingerprint"] == m.input_fingerprint

    def test_the_index_appends_rather_than_replaces(self, tmp_path):
        registry = RunRegistry(tmp_path)
        ids = []
        for _ in range(3):
            m = _manifest(output_dir=tmp_path)
            registry.save(m)
            ids.append(m.run_id)
        assert [e["run_id"] for e in registry.index()] == ids

    def test_a_run_saved_twice_appears_once_in_its_original_position(self, tmp_path):
        """
        The planning stage records what a run read; the policy stage records what was
        in force, afterwards, and may have resolved a different rule file. Both write.
        The log keeps every line; a read of it keeps the newest per run.
        """
        registry = RunRegistry(tmp_path)
        first, second = _manifest(output_dir=tmp_path), _manifest(output_dir=tmp_path)
        registry.save(first)
        registry.save(second)
        first.record_rules(["R-001"])
        registry.save(first)

        index = registry.index()
        assert [e["run_id"] for e in index] == [first.run_id, second.run_id]
        assert index[0]["policy_fingerprint"] == first.policy_fingerprint
        assert len(registry.index_path.read_text(encoding="utf-8").splitlines()) == 3

    def test_a_truncated_index_line_does_not_lose_the_rest(self, tmp_path):
        registry = RunRegistry(tmp_path)
        m = _manifest(output_dir=tmp_path)
        registry.save(m)
        with open(registry.index_path, "a", encoding="utf-8") as fh:
            fh.write('{"run_id": "half\n')
        assert [e["run_id"] for e in registry.index()] == [m.run_id]

    def test_comparing_an_unknown_run_returns_nothing(self, tmp_path):
        registry = RunRegistry(tmp_path)
        m = _manifest(output_dir=tmp_path)
        registry.save(m)
        assert registry.compare(m.run_id, "not-a-run") is None

    def test_two_saved_runs_compare(self, tmp_path, files):
        a, _ = files
        registry = RunRegistry(tmp_path)
        first = _manifest(output_dir=tmp_path); first.record_input(a, doc_type="open_po")
        registry.save(first)
        a.write_text("po_number,sku,open_qty\nP1,A,99\n", encoding="utf-8")
        second = _manifest(output_dir=tmp_path); second.record_input(a, doc_type="open_po")
        registry.save(second)

        assert registry.compare(first.run_id, second.run_id).basis == "new_data"


class TestItReportsWhatItDoesNotKnow:
    def test_an_unstorable_input_is_counted_in_the_summary(self, files):
        a, _ = files
        m = _manifest()
        m.record_input(a, doc_type="open_po", key_verdict="degraded", storable=False)
        assert len(m.unstorable_inputs) == 1
        assert "not yet storable" in m.summary()

    def test_a_complete_key_is_not_flagged(self, files):
        a, _ = files
        m = _manifest()
        m.record_input(a, doc_type="open_po", key_verdict="complete", storable=True)
        assert m.unstorable_inputs == []
        assert "not yet storable" not in m.summary()

    def test_the_manifest_serialises(self, tmp_path, files):
        a, _ = files
        m = _manifest(config_dir=tmp_path, output_dir=tmp_path)
        m.record_input(a, doc_type="open_po")
        m.record_output(a, rows=1)
        round_tripped = json.loads(json.dumps(m.to_dict()))
        assert round_tripped["inputs"][0]["doc_type"] == "open_po"
        assert round_tripped["outputs"][0]["rows"] == 1
