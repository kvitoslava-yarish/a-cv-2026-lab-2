import argparse
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


def merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def set_path(cfg, dotted, value):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_config(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("overrides", nargs="*", help="key=value, dotted keys allowed")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(BASE.read_text())
    if args.config:
        cfg = merge(cfg, yaml.safe_load(Path(args.config).read_text()) or {})
    for item in args.overrides:
        key, value = item.split("=", 1)
        set_path(cfg, key, yaml.safe_load(value))
    return cfg
