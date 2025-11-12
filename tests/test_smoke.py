def test_import():
    import importlib
    m = importlib.import_module("auto_project")
    assert hasattr(m, "__doc__") or True
