import config

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
