"""Generate a reusable synthetic trajectory dataset."""

import argparse
import json
import os
import random
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

from neuralsde.latent_sde.config import DatasetConfig
from neuralsde.latent_sde.data import build_dataset


DEFAULT_CONFIG = Path(__file__).parents[1] / "configs/mixture.dataset.yaml"
SECTION_HEADER = re.compile(
    r"^(?P<indent>\s*)(?P<name>samplers|params)\s*:\s*(?:#.*)?$"
)
NUMBER_SETTING = re.compile(
    r"^(?P<prefix>\s*[^#:\n]+:\s*)"
    r"(?P<number>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?P<suffix>\s*(?:#.*)?)$"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="use this config directly; omit to edit a randomized temporary config",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def prepare_interactive_config(text: str, rng: random.Random) -> str:
    """Randomize process parameters and comment every sampler in one pass."""
    prepared: list[str] = []
    samplers_indent: int | None = None
    params_indent: int | None = None

    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        line_ending = line[len(content) :]
        stripped = content.lstrip()
        indentation = len(content) - len(stripped)

        active_line = bool(stripped and not stripped.startswith("#"))
        if active_line:
            if samplers_indent is not None and indentation <= samplers_indent:
                samplers_indent = None
            if params_indent is not None and indentation <= params_indent:
                params_indent = None

        header = SECTION_HEADER.fullmatch(content)
        if header:
            section_indent = len(header.group("indent"))
            if header.group("name") == "samplers":
                samplers_indent = section_indent
            else:
                params_indent = section_indent
        elif params_indent is not None and (
            setting := NUMBER_SETTING.fullmatch(content)
        ):
            value = float(setting.group("number"))
            bounds = (
                (0.1, 1.0)
                if value == 0
                else sorted((value / 2, value * 1.5))
            )
            randomized = rng.uniform(*bounds)
            content = (
                f"{setting.group('prefix')}{randomized:.2f}{setting.group('suffix')}"
            )

        is_samplers_header = header and header.group("name") == "samplers"
        if samplers_indent is not None and active_line and not is_samplers_header:
            content = f"#{content}"
        prepared.append(content + line_ending)

    return "".join(prepared)


def edit_randomized_config(source: str | Path) -> DatasetConfig:
    """Open a randomized temporary copy of a dataset config in ``$EDITOR``."""
    editor = shlex.split(os.environ.get("EDITOR", ""))
    if not editor:
        raise RuntimeError("$EDITOR is not set")

    DatasetConfig.from_yaml(source)
    source_text = Path(source).read_text()
    editable_config = prepare_interactive_config(source_text, random.SystemRandom())

    with tempfile.TemporaryDirectory(prefix="neuralsde-dataset-") as directory:
        temporary_config = Path(directory) / "dataset.yaml"
        temporary_config.write_text(editable_config)

        subprocess.run([*editor, str(temporary_config)], check=True)
        try:
            return DatasetConfig.from_yaml(temporary_config)
        except ValueError as error:
            if str(error) == "samplers must be a non-empty list":
                raise ValueError(
                    "No sampler is enabled. Uncomment a complete sampler block "
                    "under 'samplers:' before closing $EDITOR."
                ) from error
            raise


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = (
            edit_randomized_config(DEFAULT_CONFIG)
            if args.config is None
            else DatasetConfig.from_yaml(args.config)
        )
    except ValueError as error:
        parser.error(str(error))
    dataset = build_dataset(config)
    dataset.save(args.output)
    print(
        json.dumps(
            {
                "dataset": str(args.output),
                "paths": len(dataset),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
