import gzip
import json

from debate.records import read_records


def test_reads_plain_and_gzipped_logs(tmp_path):
    rows = [{"problem_id": "p0"}, {"problem_id": "p1"}]
    content = "\n".join(json.dumps(r) for r in rows) + "\n"

    plain = tmp_path / "a.jsonl"
    plain.write_text(content)
    assert read_records(plain) == rows

    with gzip.open(tmp_path / "b.jsonl.gz", "wt") as f:
        f.write(content)
    assert read_records(tmp_path / "b.jsonl.gz") == rows
    assert read_records(tmp_path / "b.jsonl") == rows  # trouve la version compressée
    assert read_records(tmp_path / "absent.jsonl") == []


def test_repair_removes_truncated_last_line(tmp_path):
    from debate.records import repair_jsonl

    path = tmp_path / "r.jsonl"
    path.write_text('{"problem_id": "p0"}\n{"problem_id": "p1"}\n{"problem_id": "p2", "cal')
    assert repair_jsonl(path) > 0
    assert read_records(path) == [{"problem_id": "p0"}, {"problem_id": "p1"}]
    assert path.read_text().endswith("\n")
    assert repair_jsonl(path) == 0  # fichier sain : inchangé
