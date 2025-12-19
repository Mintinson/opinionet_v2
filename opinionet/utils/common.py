from dataclasses import dataclass, field


@dataclass
class Metrics:
    loss: list[float] = field(default_factory=list)
    f1: list[float] = field(default_factory=list)
    precision: list[float] = field(default_factory=list)
    recall: list[float] = field(default_factory=list)



__all__ = ["Metrics"]

