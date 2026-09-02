import config

import json
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

# configs/*.json are now the single source of truth (PRESETS removed).
CONFIG_FILES = [
    ("model.json", config.ModelConfig),
    ("train.json", config.TrainConfig),
    ("data.json", config.DataConfig),
    ("gen.json", config.GeneratorConfig),
]

def test_config_jsons_exist():
    """Every expected config JSON must be present on disk."""
    for name, _cls in CONFIG_FILES:
        assert (CONFIGS_DIR / name).is_file(), f"missing {name}"


def test_config_jsons_load_into_right_type():
    """Each config JSON must load into its matching dataclass type."""
    for name, cls in CONFIG_FILES:
        cfg = config.from_json(cls, CONFIGS_DIR / name)
        assert isinstance(cfg, cls), f"{name} did not load into {cls.__name__}"


def test_json_data_window_fits_model_context():
    """A training window (data seq_len) must fit inside the model's context."""
    model_cfg = config.from_json(config.ModelConfig, CONFIGS_DIR / "model.json")
    data_cfg = config.from_json(config.DataConfig, CONFIGS_DIR / "data.json")
    assert data_cfg.seq_len <= model_cfg.seq_len


def test_json_model_heads_divide_embed_dim():
    """Model config must have embed_dim divisible by n_heads."""
    model_cfg = config.from_json(config.ModelConfig, CONFIGS_DIR / "model.json")
    assert model_cfg.embed_dim % model_cfg.n_heads == 0


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
    for cls, name in [(cls, name) for name, cls in CONFIG_FILES]:
        cfg = config.from_json(cls, CONFIGS_DIR / name)
        assert isinstance(cfg, cls)

def test_to_json_roundtrip(tmp_path):
    cfg = config.ModelConfig(embed_dim=640, n_heads=10, n_layers=12, seq_len=1024)
    out = tmp_path / "model.json"
    config.to_json(cfg, out)
    loaded = config.from_json(config.ModelConfig, out)
    assert loaded == cfg
