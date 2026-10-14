"""Guard the project rule: notebooks call src/, they don't define logic or use the removed Keras stack."""

import json
import re
from pathlib import Path

import pytest

NOTEBOOKS = sorted(Path(__file__).resolve().parent.parent.joinpath("notebooks").glob("*.ipynb"))


def test_there_is_at_least_one_notebook():
    assert NOTEBOOKS


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_no_function_or_class_definitions_and_no_keras(path):
    cells = json.loads(path.read_text())["cells"]
    for i, cell in enumerate(c for c in cells if c["cell_type"] == "code"):
        src = "".join(cell["source"])
        assert not re.search(r"^\s*(def|class)\s", src, re.M), f"{path.name} code cell {i} defines logic; move it to src/"
        assert not re.search(r"\b(tensorflow|keras|keras_tuner)\b", src), f"{path.name} code cell {i} imports the old Keras stack"
