"""Tests for the generated topic README block."""

from rubinwork.studies import index


def _study(root, topic, name, status="active", answer="An answer, 1.0 um."):
    d = root / topic / name
    d.mkdir(parents=True)
    (d / "study.md").write_text(
        f"---\nstudy: {name}\ntopic: {topic}\nstatus: {status}\n"
        f"question: \"Why?\"\nanswer: \"{answer}\"\nuses: {{}}\nstarted: 2026-10-01\n---\n\n"
        f"# `{name}`\n\n## Question\n\nWhy?\n\n## Current answer\n\n{answer}\n\n"
        "## Log\n\n### 2026-10-02 — first\n\nDone.\n")


def test_block_lists_the_topic_studies(tmp_path):
    _study(tmp_path, "aos", "alpha")
    _study(tmp_path, "aos", "beta", status="parked")
    _study(tmp_path, "camera", "gamma")
    block = index.topic_block("aos", tmp_path)
    assert block.startswith(index.BEGIN) and block.rstrip().endswith(index.END)
    assert "[alpha](alpha/study.md) | active | 2026-10-02" in block
    assert "[beta](beta/study.md) | parked" in block
    assert "gamma" not in block


def test_hand_written_text_is_kept(tmp_path):
    _study(tmp_path, "aos", "alpha")
    readme = tmp_path / "aos/README.md"
    readme.write_text("# AOS\n\nIntro.\n\n## Processing\n\nHand-written.\n")
    assert index.write_topic_readme("aos", tmp_path)
    text = readme.read_text()
    assert text.startswith("# AOS\n\nIntro.\n\n" + index.BEGIN)
    assert text.endswith("## Processing\n\nHand-written.\n")
    # A rerun with no study change leaves the file alone, and a new study only
    # rewrites the block.
    assert not index.write_topic_readme("aos", tmp_path)
    _study(tmp_path, "aos", "beta")
    assert index.write_topic_readme("aos", tmp_path)
    text = readme.read_text()
    assert "[beta](beta/study.md)" in text
    assert text.count(index.BEGIN) == 1
    assert text.endswith("## Processing\n\nHand-written.\n")


def test_missing_readme_is_created(tmp_path):
    _study(tmp_path, "astrometry", "alpha")
    assert index.write_topic_readmes(tmp_path) == ["astrometry"]
    text = (tmp_path / "astrometry/README.md").read_text()
    assert text.startswith("# astrometry\n\n" + index.BEGIN)
