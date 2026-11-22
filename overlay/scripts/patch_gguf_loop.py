# SPDX-License-Identifier: Apache-2.0
"""Patch the loop pass count (and is_varloop) inside a GGUF header in place.

All GGUF metadata values are fixed-width, so rewriting scalars does not move
the tensor-info table or the tensor data. Usage:

    python patch_gguf_loop.py model.gguf --count 1 --varying 1
"""

from __future__ import annotations

import argparse
import shutil
import struct
import sys
from pathlib import Path

# GGUF value type sizes (excluding string/array, handled separately)
_SCALAR = {
    0: 1,  # uint8
    1: 1,  # int8
    2: 2,  # uint16
    3: 2,  # int16
    4: 4,  # uint32
    5: 4,  # int32
    6: 4,  # float32
    7: 1,  # bool
    10: 8,  # uint64
    11: 8,  # int64
    12: 8,  # float64
}


def read_kv_table(path: Path) -> list[tuple[str, int, int]]:
    """Returns [(key, value_type, value_offset)] for all metadata KVs."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise SystemExit(f"not a GGUF file: {magic!r}")
        version = struct.unpack("<I", f.read(4))[0]
        if version != 3:
            raise SystemExit(f"unsupported GGUF version {version}")
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        kv_count = struct.unpack("<Q", f.read(8))[0]
        entries = []
        for _ in range(kv_count):
            (key_len,) = struct.unpack("<Q", f.read(8))
            key = f.read(key_len).decode()
            (vtype,) = struct.unpack("<I", f.read(4))
            vpos = f.tell()
            if vtype in _SCALAR:
                f.seek(_SCALAR[vtype], 1)
            elif vtype == 8:  # string
                (slen,) = struct.unpack("<Q", f.read(8))
                f.seek(slen, 1)
            elif vtype == 9:  # array
                (etype,) = struct.unpack("<I", f.read(4))
                (alen,) = struct.unpack("<Q", f.read(8))
                if etype in _SCALAR:
                    f.seek(_SCALAR[etype] * alen, 1)
                elif etype == 8:
                    for _ in range(alen):
                        (slen,) = struct.unpack("<Q", f.read(8))
                        f.seek(slen, 1)
                else:
                    raise SystemExit(f"unsupported array element type {etype}")
            else:
                raise SystemExit(f"unsupported value type {vtype} for key {key}")
            entries.append((key, vtype, vpos))
        return entries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", type=Path)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--varying", type=int, choices=(0, 1), default=None)
    ap.add_argument("--copy-to", type=Path, default=None, help="patch a copy")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    target = args.gguf
    if args.copy_to is not None:
        if not args.copy_to.exists():
            print(f"copying {args.gguf} -> {args.copy_to} ...", flush=True)
            shutil.copyfile(args.gguf, args.copy_to)
        target = args.copy_to

    entries = read_kv_table(target)
    loop_keys = [(k, t, o) for k, t, o in entries if "loop" in k]
    for k, t, o in loop_keys:
        print(f"  {k} (type {t}) @ {o}")
    if args.list:
        return

    def patch(key_suffix: str, vtype: int, pack: str, value) -> bool:
        found = False
        with open(target, "r+b") as f:
            for k, t, o in entries:
                if k.endswith(key_suffix):
                    if t != vtype:
                        raise SystemExit(f"{k}: unexpected type {t} (want {vtype})")
                    f.seek(o)
                    f.write(struct.pack(pack, value))
                    print(f"patched {k} = {value} @ {o}")
                    found = True
        return found

    if args.count is not None:
        if not patch(".loop.count", 4, "<I", args.count):
            if not patch(".loop_count", 4, "<I", args.count):
                raise SystemExit("loop count key not found")
    if args.varying is not None:
        if not patch(".loop.varying", 7, "<?", bool(args.varying)):
            if not patch(".loop.is_varloop", 7, "<?", bool(args.varying)):
                print("warning: varying key not found; leaving as-is")
    print("done")


if __name__ == "__main__":
    main()
