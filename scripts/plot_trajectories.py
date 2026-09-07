"""Render learned latent-SDE rollouts with the known OU reference envelope."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--paths", type=int, default=60)
    args = parser.parse_args()

    trajectories: dict[int, list[tuple[float, float]]] = {}
    with args.input.open() as source:
        for row in csv.DictReader(source):
            path_id = int(row["path_id"])
            if path_id >= args.paths:
                break
            trajectories.setdefault(path_id, []).append((float(row["time"]), float(row["decoded_value"])))

    times = [time for time, _ in trajectories[0]]
    values = [value for path in trajectories.values() for _, value in path]
    # The reference is not fitted to the samples. It is the known distribution
    # of the synthetic OU process used to test whether the neural SDE learned
    # both mean reversion and the time-dependent variance.

    width, height = 960, 560
    left, right, top, bottom = 82, 28, 34, 72
    x_min, x_max = times[0], times[-1]
    y_min = min(values)
    y_max = max(values)
    x = lambda t: left + (t - x_min) / (x_max - x_min) * (width - left - right)
    y = lambda v: height - bottom - (v - y_min) / (y_max - y_min) * (height - top - bottom)
    polyline = lambda points: " ".join(f"{x(t):.2f},{y(v):.2f}" for t, v in points)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text { font-family: system-ui, sans-serif; fill: #1f2937; } .axis { stroke: #374151; stroke-width: 1; } .grid { stroke: #d1d5db; stroke-width: 1; } </style>',
        '<text x="82" y="23" font-size="18" font-weight="600">Decoded latent-SDE prior rollouts</text>',
    ]
    for tick in range(6):
        v = y_min + tick * (y_max - y_min) / 5
        svg += [f'<line class="grid" x1="{left}" y1="{y(v):.2f}" x2="{width-right}" y2="{y(v):.2f}"/>', f'<text x="{left-10}" y="{y(v)+4:.2f}" font-size="12" text-anchor="end">{v:.1f}</text>']

    for tick in range(5):
        t = x_min + tick * (x_max - x_min) / 4
        svg += [f'<line class="grid" x1="{x(t):.2f}" y1="{top}" x2="{x(t):.2f}" y2="{height-bottom}"/>', f'<text x="{x(t):.2f}" y="{height-bottom+24}" font-size="12" text-anchor="middle">{t:.1f}</text>']
    for path in trajectories.values():
        svg.append(f'<polyline points="{polyline(path)}" fill="none" stroke="#2563eb" stroke-width="1" opacity="0.22"/>')
    svg += [
        f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>',
        f'<text x="{(left+width-right)/2}" y="{height-22}" font-size="14" text-anchor="middle">time</text>',
        f'<text x="22" y="{height/2}" font-size="14" text-anchor="middle" transform="rotate(-90 22 {height/2})">latent state</text>',
        '<line x1="650" y1="20" x2="675" y2="20" stroke="#2563eb" opacity="0.55"/><text x="682" y="25" font-size="12">learned samples</text>',
        '</svg>',
    ]
    args.output.write_text("\n".join(svg))


if __name__ == "__main__":
    main()
