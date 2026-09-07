"""Synthetic stochastic-process datasets for Neural SDE experiments."""

from dataclasses import dataclass
from typing import Any

from .brownian_bridge import (
    BrownianBridgeParameters,
    BrownianBridgeProcess,
    sample_brownian_bridge,
)
from .cir import CIRParameters, CIRProcess, sample_cir
from .ou import OUParameters, OUProcess, sample_ou
from ._validation import validate_time_grid


@dataclass(frozen=True)
class ProcessDefinition:
    """Parameter and implementation types for one named process."""

    parameter_type: type
    process_type: type

    def build(self, parameters: Any) -> Any:
        return self.process_type(parameters)


PROCESS_TYPES = {
    "ou": ProcessDefinition(OUParameters, OUProcess),
    "cir": ProcessDefinition(CIRParameters, CIRProcess),
    "brownian_bridge": ProcessDefinition(BrownianBridgeParameters, BrownianBridgeProcess),
}

__all__ = [
    "BrownianBridgeParameters",
    "BrownianBridgeProcess",
    "CIRParameters",
    "CIRProcess",
    "OUParameters",
    "OUProcess",
    "PROCESS_TYPES",
    "ProcessDefinition",
    "sample_brownian_bridge",
    "sample_cir",
    "sample_ou",
    "validate_time_grid",
]
