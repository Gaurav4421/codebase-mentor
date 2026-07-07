from ingestion.file_loader import extract_file_imports, iter_source_files


def test_iter_source_files_finds_all_supported_files(tiny_repo):
    files = list(iter_source_files(str(tiny_repo)))
    paths = {f.path for f in files}

    assert len(files) == 2
    assert any(p.endswith("utils.py") for p in paths)
    assert any(p.endswith("main.py") for p in paths)
    assert all(f.language == "python" for f in files)


def test_iter_source_files_skips_ignored_dirs(tiny_repo):
    ignored = tiny_repo / "node_modules"
    ignored.mkdir()
    (ignored / "vendored.py").write_text("x = 1\n")

    files = list(iter_source_files(str(tiny_repo)))
    assert not any("node_modules" in f.path for f in files)


def test_iter_source_files_skips_empty_files(tiny_repo):
    (tiny_repo / "empty.py").write_text("   \n\n")

    files = list(iter_source_files(str(tiny_repo)))
    assert not any(f.path.endswith("empty.py") for f in files)


def test_extract_file_imports_picks_up_import_lines():
    content = '"""Docstring."""\nimport os\nfrom pathlib import Path\n\ndef f():\n    pass\n'
    imports = extract_file_imports(content)
    assert "import os" in imports
    assert "from pathlib import Path" in imports
    assert not any("def f" in line for line in imports)
