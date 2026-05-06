from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


INRIA_DATASET = "inria_building"
DEVENTER_DATASET = "deventer_512_valtest_as_val"


@dataclass
class CommandStep:
    name: str
    argv: list[str]
    cwd: Path
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class RunContext:
    release_root: Path
    data_processed_root: Path
    output_root: Path
    method: str
    dataset: str
    task: str
    mode: str
    run_name: str
    gpu: str
    seed: int
    smoke: bool
    run_val: bool
    no_conda: bool
    env_name: str | None
    bbox_json: Path | None
    params: dict[str, str]

    @property
    def is_inria(self) -> bool:
        return self.dataset == INRIA_DATASET

    @property
    def dataset_root(self) -> Path:
        return self.data_processed_root / self.dataset

    def model_root(self, method: str | None = None) -> Path:
        return self.release_root / "models" / (method or self.method)

    def source_root(self, method: str | None = None) -> Path:
        return self.model_root(method) / "source"

    def output_dir(self, method: str | None = None) -> Path:
        return self.output_root / (method or self.method) / self.dataset / self.task / self.run_name

    def data_dir(self, baseline_dir: str, *, fallback: str | None = None) -> Path:
        candidates = []
        if self.is_inria:
            candidates.append(self.dataset_root / baseline_dir)
            if fallback:
                candidates.append(self.dataset_root / fallback)
        else:
            candidates.append(self.dataset_root / baseline_dir / self.task)
            if fallback:
                candidates.append(self.dataset_root / fallback / self.task)
        for candidate in candidates:
            if (candidate / "train").exists() or (candidate / "val").exists():
                return candidate
        return candidates[0]

    def hisup_gt_json(self) -> Path:
        return self.data_dir("hisup") / "val" / "annotation.json"

    def param(self, name: str, default: str | int | float | bool) -> str:
        if name in self.params:
            return self.params[name]
        if name in os.environ:
            return os.environ[name]
        if isinstance(default, bool):
            return "1" if default else "0"
        return str(default)

    def env(self, **items: str | Path | int | float | None) -> dict[str, str]:
        result = {"CUDA_VISIBLE_DEVICES": self.gpu}
        for key, value in items.items():
            if value is not None:
                result[key] = str(value)
        return result

    def path(self, path: Path, cwd: Path) -> str:
        path = Path(path)
        if not path.is_absolute():
            return str(path)
        return os.path.relpath(path, cwd)

    def conda(self, env_name: str, argv: list[str]) -> list[str]:
        if self.no_conda:
            return argv
        return ["conda", "run", "--no-capture-output", "-n", env_name, *argv]

    def python(self, env_name: str, script: Path, args: list[str], cwd: Path) -> list[str]:
        return self.conda(env_name, ["python", self.path(script, cwd), *args])

    def evaluate_step(
        self,
        *,
        name: str,
        pred: Path,
        gt: Path | None = None,
        env_name: str,
        cwd: Path | None = None,
        pred_type: str = "hisup",
        gt_type: str = "hisup",
    ) -> CommandStep:
        cwd = cwd or self.release_root
        output = pred.parent / "metrics.json"
        args = [
            "--pred", self.path(pred, cwd),
            "--gt", self.path(gt or self.hisup_gt_json(), cwd),
            "--gt-type", gt_type,
            "--pred-type", pred_type,
            "--min-hole-area", self.param("MIN_HOLE_AREA", 16),
            "--num-workers", self.param("EVAL_NUM_WORKERS", 16),
            "--output", self.path(output, cwd),
        ]
        return CommandStep(
            name=name,
            cwd=cwd,
            argv=self.python(env_name, self.release_root / "utilis" / "evaluate_vector_polygons.py", args, cwd),
        )


def parse_key_values(items: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError(f"--set has an empty key: {item}")
        parsed[key] = value
    return parsed


def find_data_processed_root(release_root: Path, override: str | None) -> Path:
    if override:
        path = Path(override)
        return path if path.is_absolute() else release_root / path
    candidates = [
        release_root / "dataset" / "data_processed",
        release_root / "data_processed",
        release_root.parent / "data_processed",
    ]
    for candidate in candidates:
        if (candidate / INRIA_DATASET).exists() or (candidate / DEVENTER_DATASET).exists():
            return candidate
    return candidates[0]


def default_task(dataset: str, task: str | None) -> str:
    if task:
        return task
    return "building" if dataset == INRIA_DATASET else "road"


def default_run_name(method: str, task: str, mode: str, smoke: bool) -> str:
    suffix = "smoke" if smoke else mode
    return f"{method}_{task}_{suffix}"


def run_steps(steps: list[CommandStep], *, dry_run: bool) -> None:
    for step in steps:
        step.cwd.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(step.env)
        rendered = " ".join(f"{key}={shlex.quote(value)}" for key, value in step.env.items())
        rendered = f"{rendered} " if rendered else ""
        try:
            cwd_display = os.path.relpath(step.cwd, Path.cwd())
        except ValueError:
            cwd_display = str(step.cwd)
        print(f"[{step.name}] cwd={cwd_display}")
        print(f"{rendered}{shlex.join(step.argv)}", flush=True)
        if dry_run:
            continue
        subprocess.run(step.argv, cwd=step.cwd, env=env, check=True)


def require_paths(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required release input(s):\n" + "\n".join(missing))
