from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    meaning: str
    default: float | int | str
    source: str
    minimum: float | int | None = None
    maximum: float | int | None = None
    kind: type = float


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    epochs: int
    episodes_per_epoch: int
    alpha: float
    epsilon: float
    seed: int


PARAMETERS = [
    ParameterSpec(
        name="epochs",
        meaning="outer training loops",
        default=2,
        source="conservative-default smoke value",
        minimum=1,
        maximum=100000,
        kind=int,
    ),
    ParameterSpec(
        name="episodes_per_epoch",
        meaning="inner episodes per epoch",
        default=2,
        source="conservative-default smoke value",
        minimum=1,
        maximum=100000,
        kind=int,
    ),
    ParameterSpec(
        name="alpha",
        meaning="learning rate",
        default=0.001,
        source="global default; replace with accepted-local-best when available",
        minimum=0.0,
        maximum=1.0,
        kind=float,
    ),
    ParameterSpec(
        name="epsilon",
        meaning="fixed exploration rate",
        default=0.1,
        source="global default; replace with accepted-local-best when available",
        minimum=0.0,
        maximum=1.0,
        kind=float,
    ),
    ParameterSpec(
        name="seed",
        meaning="random seed for reproducible non-robustness runs",
        default=42,
        source="global-default-single-seed",
        minimum=0,
        maximum=9999999999,
        kind=int,
    ),
]


def _format_range(spec: ParameterSpec) -> str:
    if spec.minimum is None and spec.maximum is None:
        return "unbounded"
    return f"[{spec.minimum}, {spec.maximum}]"


def _ask(spec: ParameterSpec) -> float | int | str:
    while True:
        raw = input(
            f"{spec.name} ({spec.meaning}, range={_format_range(spec)}, "
            f"default={spec.default}, source={spec.source}): "
        ).strip()
        if not raw:
            return spec.default
        try:
            value = spec.kind(raw)
        except ValueError:
            print(f"Invalid value. Expected {spec.kind.__name__}.")
            continue
        if spec.minimum is not None and value < spec.minimum:
            print(f"Value must be >= {spec.minimum}.")
            continue
        if spec.maximum is not None and value > spec.maximum:
            print(f"Value must be <= {spec.maximum}.")
            continue
        return value


def interactive_config() -> RunConfig:
    print("Student-friendly research run. Press Enter to use defaults.")
    values = {spec.name: _ask(spec) for spec in PARAMETERS}
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_student_run")
    return RunConfig(run_id=run_id, **values)
