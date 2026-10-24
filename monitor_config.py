"""Validated YAML configuration, without importing the inference stack."""

from dataclasses import dataclass, field
import math
from pathlib import Path


class ConfigError(ValueError):
    pass


@dataclass
class Watch:
    path: str = ""
    scan_seconds: float = 10
    stable_seconds: float = 30
    recursive: bool = False


@dataclass
class Mount:
    remount_command: list = field(default_factory=list)
    interval_seconds: float = 60
    timeout_seconds: float = 30


@dataclass
class Analysis:
    model: str = "runs/detect/train-8/weights/best.pt"
    data: str = ""
    confidence: float = 0.45
    min_object_hits: int = 3
    max_bad_frames: int = 25
    threads: int = 1
    iou: float = 0.7
    timeout_seconds: float = 1800
    max_attempts: int = 3


@dataclass
class Signal:
    command: list = field(default_factory=list)
    group_id: str = ""
    timeout_seconds: float = 120


@dataclass
class Storage:
    path: str = "state"
    max_spool_mb: int = 256


@dataclass
class Retry:
    initial_seconds: float = 30
    maximum_seconds: float = 3600


@dataclass
class Config:
    watch: Watch
    mount: Mount
    analysis: Analysis
    signal: Signal
    storage: Storage
    retry: Retry


def load_config(filename):
    try:
        import yaml
    except ImportError as exc:
        raise ConfigError("PyYAML is required to read configuration") from exc
    source = Path(filename).resolve()
    try:
        raw = yaml.safe_load(source.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot read YAML configuration: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("Configuration must be a YAML mapping")
    sections = dict(watch=Watch, mount=Mount, analysis=Analysis,
                    signal=Signal, storage=Storage, retry=Retry)
    if set(raw) - set(sections):
        raise ConfigError(f"Unknown sections: {sorted(map(str, set(raw) - set(sections)))}")
    values = {}
    for name, cls in sections.items():
        supplied = raw.get(name, {})
        if not isinstance(supplied, dict):
            raise ConfigError(f"{name} must be a mapping")
        if set(supplied) - set(cls.__dataclass_fields__):
            raise ConfigError(f"Unknown setting in {name}")
        values[name] = cls(**supplied)
    config = Config(**values)
    for name in ("watch", "mount", "analysis", "signal", "storage", "retry"):
        obj = getattr(config, name)
        defaults = sections[name]()
        for key in obj.__dataclass_fields__:
            value, default = getattr(obj, key), getattr(defaults, key)
            label = f"{name}.{key}"
            if isinstance(default, bool):
                valid = type(value) is bool
            elif isinstance(default, (float, int)):
                valid = (type(value) in (int, float) and math.isfinite(value)
                         and value > 0)
                if isinstance(default, int):
                    valid = valid and type(value) is int
            elif isinstance(default, list):
                valid = (isinstance(value, list) and bool(value)
                         and all(isinstance(v, str) and v.strip() for v in value))
            else:
                valid = isinstance(value, str)
            if not valid:
                raise ConfigError(f"Invalid value for {label}")
    if not config.watch.path.strip():
        raise ConfigError("watch.path is required")
    if not config.analysis.model.strip():
        raise ConfigError("analysis.model is required")
    if not config.storage.path.strip():
        raise ConfigError("storage.path is required")
    for key in ("confidence", "iou"):
        if getattr(config.analysis, key) > 1:
            raise ConfigError(f"analysis.{key} must be at most 1")
    if config.retry.maximum_seconds < config.retry.initial_seconds:
        raise ConfigError("retry.maximum_seconds must be >= initial_seconds")
    if not config.signal.group_id.strip():
        raise ConfigError("signal.group_id is required")
    for placeholder in ("{image}", "{message}", "{group_id}"):
        if placeholder not in config.signal.command:
            raise ConfigError(f"signal.command must contain a separate {placeholder} argument")
    # All relative paths are relative to the YAML file, regardless of working directory.
    for obj, key in ((config.watch, "path"), (config.storage, "path"),
                     (config.analysis, "model"), (config.analysis, "data")):
        value = getattr(obj, key)
        if value:
            setattr(obj, key, str((source.parent / value).resolve()))
    watch, state = Path(config.watch.path), Path(config.storage.path)
    if state == watch or watch in state.parents:
        raise ConfigError("storage.path must be outside watch.path")
    return config
