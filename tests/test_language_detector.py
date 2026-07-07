from ingestion.language_detector import detect_language, is_supported


def test_detects_common_languages():
    assert detect_language("main.py") == "python"
    assert detect_language("app.tsx") == "typescript"
    assert detect_language("index.js") == "javascript"
    assert detect_language("Main.java") == "java"
    assert detect_language("lib.rs") == "rust"


def test_case_insensitive_extension():
    assert detect_language("README.MD") == "markdown"


def test_unsupported_extension_returns_none():
    assert detect_language("binary.exe") is None
    assert not is_supported("image.png")


def test_supported_extension():
    assert is_supported("server.go")
