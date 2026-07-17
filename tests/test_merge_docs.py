"""Compile and execute the public merge guide examples."""

from __future__ import annotations
import datetime as dt

import pathlib
import re
import warnings
import types

import polars as pl
import pyarrow as pa

from fsspeckit.core.incremental import MergeResult

from pydala import ParquetDataset
from pydala.filesystem import FileSystem


_ROOT = pathlib.Path(__file__).parents[1]
_DOCS = _ROOT / "docs"
_GUIDE = _DOCS / "user-guide" / "merge.md"
_PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _public_pages() -> list[pathlib.Path]:
    excluded = {"plans", "agents", "adr"}
    return [
        path
        for path in _DOCS.rglob("*.md")
        if not excluded.intersection(path.relative_to(_DOCS).parts)
    ]


def _examples(page: pathlib.Path = _GUIDE) -> list[str]:
    return _PYTHON_FENCE.findall(page.read_text())


def _compile_examples(page: pathlib.Path, *, merge_only: bool = False) -> int:
    compiled = 0
    for index, example in enumerate(_examples(page)):
        if merge_only and ".merge(" not in example:
            continue
        compile(example, f"{page}::python-block-{index}", "exec")
        compiled += 1
    return compiled


def test_merge_guide_python_examples_compile() -> None:
    assert _compile_examples(_GUIDE) >= 5


def test_merge_guide_python_examples_run(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for index, example in enumerate(_examples()):
            code = compile(example, f"{_GUIDE}::python-block-{index}", "exec")
            exec(code, namespace)

    for name in (
        "initial_result",
        "insert_result",
        "update_result",
        "upsert_result",
        "legacy_result",
        "replacement_result",
    ):
        assert isinstance(namespace[name], MergeResult)

    assert namespace["insert_result"].inserted == 1
    assert namespace["update_result"].updated == 1
    assert namespace["upsert_result"].inserted == 1
    assert namespace["upsert_result"].updated == 1


def test_all_public_merge_snippets_compile() -> None:
    compiled = sum(_compile_examples(page, merge_only=True) for page in _public_pages())
    assert compiled >= 10


def test_contextual_public_merge_snippets_run(tmp_path: pathlib.Path) -> None:
    pages = (
        _DOCS / "index.md",
        _DOCS / "quick-start.md",
        _DOCS / "user-guide" / "basic-usage.md",
        _DOCS / "user-guide" / "data-operations.md",
        _DOCS / "user-guide" / "catalog-management.md",
    )

    for index, page in enumerate(pages):
        dataset = ParquetDataset(
            path=f"contextual-{index}",
            filesystem=FileSystem(bucket=str(tmp_path), cached=False),
        )
        namespace: dict[str, object] = {
            "catalog": types.SimpleNamespace(load=lambda *args, **kwargs: dataset),
            "dataset": dataset,
            "datetime": dt.datetime,
            "incoming_orders": pa.table(
                {"order_id": [1], "year": [2024], "month": [1]}
            ),
            "new_sales_data": pa.table(
                {
                    "sale_id": [1],
                    "date": [dt.date(2024, 1, 1)],
                    "region": ["EU"],
                    "timestamp": [dt.datetime(2024, 1, 1)],
                }
            ),
            "pa": pa,
            "pl": pl,
            "updated_customers": pa.table({"customer_id": [1], "name": ["Ada"]}),
        }

        for block_index, example in enumerate(_examples(page)):
            if ".merge(" not in example:
                continue
            code = compile(
                example,
                f"{page}::python-block-{block_index}",
                "exec",
            )
            exec(code, namespace)

        assert isinstance(namespace["result"], MergeResult)


def test_legacy_delta_examples_only_appear_in_migration_guide() -> None:
    legacy_call = re.compile(r"mode\s*=\s*[\"']delta")
    for page in _public_pages():
        if page == _GUIDE:
            continue
        text = page.read_text()
        assert legacy_call.search(text) is None, page
        assert "### Delta Updates" not in text, page


def test_planning_references_are_only_in_the_migration_guide() -> None:
    planning_markers = (
        "fsspeckit-merge-replaces-delta-prd",
        "github.com/legout/pydala2/issues/24",
        "github.com/legout/pydala2/issues/25",
        "github.com/legout/pydala2/issues/26",
        "github.com/legout/pydala2/issues/27",
        "github.com/legout/pydala2/issues/28",
        "github.com/legout/pydala2/issues/29",
        "github.com/legout/pydala2/issues/30",
    )
    for page in _public_pages():
        if page == _GUIDE:
            continue
        text = page.read_text()
        assert not any(marker in text for marker in planning_markers), page

    assert "plans/" not in (_ROOT / "mkdocs.yml").read_text()
