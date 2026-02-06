from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClusterConfig:
    host: str
    username: str
    key_path: Path
