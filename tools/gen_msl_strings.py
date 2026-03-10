#!/usr/bin/env python3
"""Generate src/metal/msl_strings.h from the canonical .metal kernel files.

The thirteen MSL libraries compiled at runtime by primitives*.mm are sourced from
.metal files in src/metal/kernels/.  This script reads those files and emits
a single C++ header (msl_strings.h) containing the thirteen constexpr string
constants that primitives.mm includes.

Usage (from repository root):
    # Regenerate the header after editing a .metal file:
    python3 tools/gen_msl_strings.py

    # Verify the header is in sync (non-zero exit if not — suitable for CI):
    python3 tools/gen_msl_strings.py --check
"""

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).parent.parent
KERNELS_DIR = REPO_ROOT / "src" / "metal" / "kernels"
OUTPUT_FILE = REPO_ROOT / "src" / "metal" / "msl_strings.h"

# Local .metalh headers that may be #included by .metal files.
# These are inlined by the generator since MTLDevice newLibraryWithSource:
# does not have filesystem access to resolve #include directives.
LOCAL_HEADERS = {
    "metal_math.metalh": KERNELS_DIR / "metal_math.metalh",
}

# Ordered list of (filename_stem, constant_name).
# Order must match the order the constants are used in primitives.mm.
KERNELS = [
    ("elementwise",    "kElementwiseMSL"),
    ("activation",     "kActivationMSL"),
    ("broadcast",      "kBroadcastMSL"),
    ("beam_search",    "kBeamSearchMSL"),
    ("transpose",      "kTransposeMSL"),
    ("reduction",      "kReductionMSL"),
    ("normalization",  "kNormalizationMSL"),
    ("gather",         "kGatherMSL"),
    ("sdpa",           "kSdpaMSL"),
    ("conv1d",         "kConv1dMSL"),
    ("quantize",       "kQuantizeMSL"),
    ("topk",           "kTopKMSL"),
    ("fused_norm_gemm", "kFusedNormGemmMSL"),
    ("indexed_fill",    "kIndexedFillMSL"),
]

# Raw-string delimiter used to wrap the MSL source.
# Must not appear inside any .metal file (checked below).
RAW_DELIM = "msl"

FILE_HEADER = """\
// AUTO-GENERATED — DO NOT EDIT.
//
// Source:    src/metal/kernels/*.metal  (canonical MSL source files)
// Generator: tools/gen_msl_strings.py
//
// To regenerate after editing a .metal file:
//     python3 tools/gen_msl_strings.py
//
// To verify that the header is in sync (run in CI):
//     python3 tools/gen_msl_strings.py --check

"""


def _resolve_local_includes(content: str) -> str:
    """Inline local .metalh #includes that MTL runtime cannot resolve."""
    import re
    _header_cache: dict[str, str] = {}

    def _replace(m: re.Match) -> str:
        name = m.group(1)
        if name not in LOCAL_HEADERS:
            return m.group(0)  # leave unknown includes untouched
        if name not in _header_cache:
            path = LOCAL_HEADERS[name]
            if not path.exists():
                sys.exit(f"ERROR: local header not found: {path}")
            _header_cache[name] = path.read_text()
        return _header_cache[name]

    return re.sub(r'#include\s+"([^"]+\.metalh)"', _replace, content)


def build_content() -> str:
    """Return the full text of the generated header."""
    parts = [FILE_HEADER]
    close_marker = f'){RAW_DELIM}"'
    for stem, const_name in KERNELS:
        metal_file = KERNELS_DIR / f"{stem}.metal"
        if not metal_file.exists():
            sys.exit(f"ERROR: source file not found: {metal_file}")
        content = metal_file.read_text()
        # Inline local .metalh headers before embedding.
        content = _resolve_local_includes(content)
        # Ensure the raw-string close marker does not appear inside the content.
        if close_marker in content:
            sys.exit(
                f"ERROR: {metal_file} contains the raw-string terminator "
                f"'{close_marker}'.  Choose a different RAW_DELIM."
            )
        # Ensure content ends with a newline so the close marker is on its own line.
        if not content.endswith("\n"):
            content += "\n"
        parts.append(f"// Source: src/metal/kernels/{stem}.metal\n")
        parts.append(f'static constexpr const char* {const_name} = R"{RAW_DELIM}(\n')
        parts.append(content)
        parts.append(f'){RAW_DELIM}";\n\n')
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="Verify header matches .metal sources; exit 1 if not")
    args = parser.parse_args()

    new_content = build_content()

    if args.check:
        if not OUTPUT_FILE.exists():
            sys.exit(
                f"ERROR: {OUTPUT_FILE} does not exist.\n"
                f"Run:  python3 tools/gen_msl_strings.py"
            )
        existing = OUTPUT_FILE.read_text()
        if existing == new_content:
            print(f"OK: {OUTPUT_FILE.relative_to(REPO_ROOT)} is in sync.")
        else:
            sys.exit(
                f"ERROR: {OUTPUT_FILE.relative_to(REPO_ROOT)} is out of sync "
                f"with the .metal source files.\n"
                f"Run:  python3 tools/gen_msl_strings.py"
            )
    else:
        OUTPUT_FILE.write_text(new_content)
        print(f"Generated {OUTPUT_FILE.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
