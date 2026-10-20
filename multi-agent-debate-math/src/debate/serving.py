"""Lancement des serveurs vLLM d'une étape d'expériences (sur GPU).

Un serveur par modèle, chacun sur un port et un GPU donnés (configs/stages.yaml). Le T4
(architecture Turing) ne gère pas le bfloat16 : les modèles sont chargés en float16.
"""

import logging
import os
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServerSpec:
    model: str
    port: int
    gpu: int
    memory: float  # part de la mémoire du GPU réservée (poids + cache KV)
    max_model_len: int = 16384  # débats à 5 agents : 4 réponses d'autres agents dans le prompt

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}/v1"

    def command(self) -> list[str]:
        return [
            "vllm",
            "serve",
            self.model,
            "--port",
            str(self.port),
            "--dtype",
            "half",
            "--gpu-memory-utilization",
            str(self.memory),
            "--max-model-len",
            str(self.max_model_len),
            "--seed",
            "0",
        ]


@dataclass(frozen=True)
class Stage:
    name: str
    description: str
    servers: list[ServerSpec]
    experiments: list[Path]

    @property
    def endpoints(self) -> dict[str, str]:
        return {s.model: s.url for s in self.servers}


def load_stage(name: str, path: Path = Path("configs/stages.yaml")) -> Stage:
    stages = yaml.safe_load(path.read_text())
    if name not in stages:
        raise SystemExit(f"Étape inconnue : {name} (disponibles : {', '.join(stages)})")
    spec = stages[name]
    return Stage(
        name=name,
        description=spec["description"],
        servers=[ServerSpec(**s) for s in spec["servers"]],
        experiments=[Path(p) for p in spec["experiments"]],
    )


def start_server(spec: ServerSpec, log_dir: Path) -> subprocess.Popen:
    log_dir.mkdir(parents=True, exist_ok=True)
    log = (log_dir / f"vllm_{spec.port}.log").open("w")
    env = os.environ | {"CUDA_VISIBLE_DEVICES": str(spec.gpu)}
    logger.info("Lancement de %s sur le GPU %d (port %d)", spec.model, spec.gpu, spec.port)
    return subprocess.Popen(spec.command(), stdout=log, stderr=subprocess.STDOUT, env=env)


def wait_until_ready(spec: ServerSpec, process: subprocess.Popen, timeout: float = 1800) -> None:
    """Attend que le serveur réponde (téléchargement + chargement du modèle)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Le serveur vLLM de {spec.model} s'est arrêté (voir son log)")
        try:
            with urllib.request.urlopen(f"http://localhost:{spec.port}/health", timeout=5):
                logger.info("%s prêt", spec.model)
                return
        except OSError:
            time.sleep(10)
    raise TimeoutError(f"{spec.model} n'a pas démarré en {timeout:.0f} s")
