import json, pathlib

def test_config_example_json_valid():
    cfg = pathlib.Path("config/config.example.json")
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert "db" in data and "runtime" in data
