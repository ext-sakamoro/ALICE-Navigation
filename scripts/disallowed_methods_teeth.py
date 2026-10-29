#!/usr/bin/env python3
"""Every entry of clippy.toml's disallowed-methods must stop a build.

For each `<type>::<method>` entry, a throwaway module calling it in both forms (method call and
path call) is added to the library, and `cargo clippy --lib -- -D warnings` must fail with a
`disallowed_methods` error for that path, twice. The library is restored after each entry.

usage: python3 scripts/disallowed_methods_teeth.py      (from the repository root)
Prints "ran N of M: K red via disallowed_methods, J not red" and exits non-zero unless every entry
is red, N == M and M > 0.
"""
import pathlib
import re
import subprocess
import sys

if len(sys.argv) > 1:
    sys.exit(f"unknown arguments: {sys.argv[1:]}")

root = pathlib.Path(__file__).resolve().parent.parent
paths = re.findall(r'path = "((?:f32|f64)::[a-z_0-9]+)"', (root / "clippy.toml").read_text(encoding="utf-8"))
lib = root / "src" / "lib.rs"
probe = root / "src" / "disallowed_probe.rs"
original = lib.read_text(encoding="utf-8")

ARGS = {"atan2": 2, "hypot": 2, "log": 2, "powf": 2, "powi": 2, "max": 2, "min": 2, "clamp": 3}


def call(ty: str, method: str) -> tuple[str, str]:
    one = "1.5" + ty
    extra = {"powi": "2", "log": "2.0" + ty}.get(method, "0.5" + ty)
    if ARGS.get(method) == 3:
        lo, hi = "0.5" + ty, "2.5" + ty
        return f"{one}.{method}({lo}, {hi})", f"{ty}::{method}({one}, {lo}, {hi})"
    if ARGS.get(method) == 2:
        return f"{one}.{method}({extra})", f"{ty}::{method}({one}, {extra})"
    return f"{one}.{method}()", f"{ty}::{method}({one})"


red, not_red = [], []
try:
    for path in paths:
        ty, method = path.split("::")
        a, b = call(ty, method)
        # no typed return: a method returning a tuple (sin_cos) must still reach the lint, not stop
        # at a type error
        probe.write_text(
            "#[allow(dead_code)]\n"
            "pub fn probe() {\n"
            f"    let _ = {a};\n"
            f"    let _ = {b};\n"
            "}\n",
            encoding="utf-8",
        )
        lib.write_text(original + "\nmod disallowed_probe;\n", encoding="utf-8")
        out = subprocess.run(
            ["cargo", "clippy", "--lib", "--", "-D", "warnings"],
            cwd=root,
            capture_output=True,
            text=True,
        )
        hits = len(re.findall(rf"use of a disallowed method `[^`]*{re.escape(path)}`", out.stderr))
        (red if out.returncode != 0 and hits >= 2 else not_red).append((path, hits))
finally:
    lib.write_text(original, encoding="utf-8")
    probe.unlink(missing_ok=True)

print(f"ran {len(red) + len(not_red)} of {len(paths)}: {len(red)} red via disallowed_methods, {len(not_red)} not red")
for path, hits in not_red:
    print(f"  not red: {path} ({hits} hits)")
if not paths or not_red or len(red) != len(paths):
    sys.exit(1)
