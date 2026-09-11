import pytest

from src.core.config import load_config


def test_default_config():
    config = load_config()
    assert config["camera"]["camera_id"] == "CAM_001"
    assert config["detector"]["confidence_threshold"] == 0.4
    assert "person" in config["detector"]["detection_classes"]
    assert "person" not in config["violation"]["target_classes"]
    assert config["violation"]["min_inside_frames"] >= 1
    assert config["zones"]["bbox_overlap_threshold"] == 0.2
    assert "stationary" not in config


def test_custom_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("camera:\n  camera_id: TEST\n", encoding="utf-8")
    assert load_config(path) == {"camera": {"camera_id": "TEST"}}


def test_missing_config(tmp_path):
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(tmp_path / "missing.yaml")


def test_invalid_yaml(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("camera: [unterminated", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid YAML"):
        load_config(path)


@pytest.mark.parametrize("content", ["", "- item", "42"])
def test_non_mapping_config(tmp_path, content):
    path = tmp_path / "bad.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(path)
