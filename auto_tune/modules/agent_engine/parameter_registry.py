"""Single source of truth for LLM-tunable YOLO training parameters."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParameterSpec:
    kind: str
    cli_name: str
    minimum: float | None = None
    maximum: float | None = None
    choices: frozenset[str] = field(default_factory=frozenset)
    group: str = "hyperparameter"


def _float(cli_name: str, minimum: float, maximum: float) -> ParameterSpec:
    return ParameterSpec("float", cli_name, minimum, maximum)


def _int(cli_name: str, minimum: int, maximum: int, group: str = "training") -> ParameterSpec:
    return ParameterSpec("int", cli_name, minimum, maximum, group=group)


PARAMETER_REGISTRY: dict[str, ParameterSpec] = {
    "lr0": _float("lr0", 1e-5, 0.1),
    "lrf": _float("lrf", 1e-5, 1.0),
    "momentum": _float("momentum", 0.6, 0.99),
    "weight_decay": _float("weight_decay", 0.0, 0.1),
    "warmup_epochs": _float("warmup_epochs", 0.0, 10.0),
    "warmup_momentum": _float("warmup_momentum", 0.0, 0.99),
    "warmup_bias_lr": _float("warmup_bias_lr", 0.0, 0.2),
    "box": _float("box", 1.0, 20.0),
    "cls": _float("cls", 0.1, 5.0),
    "dfl": _float("dfl", 0.5, 5.0),
    "degrees": _float("degrees", 0.0, 45.0),
    "translate": _float("translate", 0.0, 1.0),
    "scale": _float("scale", 0.0, 1.0),
    "shear": _float("shear", 0.0, 45.0),
    "perspective": _float("perspective", 0.0, 0.001),
    "flipud": _float("flipud", 0.0, 1.0),
    "fliplr": _float("fliplr", 0.0, 1.0),
    "mosaic": _float("mosaic", 0.0, 1.0),
    "mixup": _float("mixup", 0.0, 1.0),
    "copy_paste": _float("copy_paste", 0.0, 1.0),
    "hsv_h": _float("hsv_h", 0.0, 0.1),
    "hsv_s": _float("hsv_s", 0.0, 1.0),
    "hsv_v": _float("hsv_v", 0.0, 1.0),
    "dropout": _float("dropout", 0.0, 0.5),
    "batch": _int("batch", 1, 256),
    "epochs": _int("epochs", 1, 1000),
    "patience": _int("patience", 0, 200),
    "imgsz": _int("imgsz", 32, 4096),
    "close_mosaic": _int("close_mosaic", 0, 1000),
    "optimizer": ParameterSpec(
        "choice",
        "optimizer",
        choices=frozenset({"SGD", "AdamW", "Adam", "auto", "Adamax", "NAdam", "RAdam"}),
        group="training",
    ),
    "cos_lr": ParameterSpec("bool", "cos_lr"),
    "model": ParameterSpec("string", "model", group="training"),
}


def get_tunable_parameter_names() -> frozenset[str]:
    return frozenset(PARAMETER_REGISTRY)

