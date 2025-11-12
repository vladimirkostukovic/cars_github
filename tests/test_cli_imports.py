import importlib

def test_main_entry_resolves():
    pkg = importlib.import_module("auto_project")
    assert pkg is not None
    sub = importlib.import_module("auto_project.__main__")
    assert hasattr(sub, "__package__")
