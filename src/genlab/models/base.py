from __future__ import annotations

from abc import ABC, abstractmethod


class Base3DGenModel(ABC):
    """Base interface for 3D generation model adapters."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def setup(self) -> None:
        """Prepare environment and dependencies."""

    @abstractmethod
    def generate(
        self,
        input_image: str | None = None,
        input_prompt: str | None = None,
        output_dir: str | None = None,
    ) -> str:
        """Run generation and return path to generated mesh file."""
