import config

import json
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

def test_presets_have_required_keys():
    for name, preset in config.PRESETS.items():
        assert "model" in preset
        assert "data" in preset
        assert "train" in preset

def test_preset_seq_len_matches():
    for name, preset in config.PRESETS.items():
        m_seq = preset["model"]["seq_len"]
        d_seq = preset["data"]["seq_len"]
        assert m_seq == d_seq, f"preset '{name}' seq_len mismatch"

def test_model_heads_divide_embed_dim():
    for name, preset in config.PRESETS.items():
        m = preset["model"]
        assert m["embed_dim"] % m["n_heads"] == 0
def test_from_json_overrides_and_defaults(tmp_path):
    """from_json: JSON values win, missing keys use dataclass defaults, unknown keys dropped."""
    f = tmp_path / "model.json"
    f.write_text(json.dumps({
        "note": "not a ModelConfig field - must be dropped",
        "embed_dim": 640,
        "n_heads": 10,
        "n_layers": 12,
        "seq_len": 1024,
    }))
    cfg = config.from_json(config.ModelConfig, f)
    # overrides from the JSON
    assert cfg.embed_dim == 640
    assert cfg.n_heads == 10
    assert cfg.n_layers == 12
    assert cfg.seq_len == 1024
    # keys missing from the JSON fall back to dataclass defaults
    assert cfg.device == "cuda"
    assert cfg.vocab_size == 50257
    assert cfg.dropout == 0.1

def test_sample_configs_load_without_error():
    """Sample JSONs must still load into the right config type.
    Values are user-editable data, so we assert type only - never values."""
    expected = [
        (config.ModelConfig, "model.json"),
        (config.TrainConfig, "train.json"),
        (config.DataConfig, "data.json"),
        (config.GeneratorConfig, "gen.json"),
    ]
    for cls, name in expected:
        cfg = config.from_json(cls, CONFIGS_DIR / name)
        assert isinstance(cfg, cls)

def test_to_json_roundtrip(tmp_path):
    cfg = config.ModelConfig(embed_dim=640, n_heads=10, n_layers=12, seq_len=1024)
    out = tmp_path / "model.json"
    config.to_json(cfg, out)
    loaded = config.from_json(config.ModelConfig, out)
    assert loaded == cfg
