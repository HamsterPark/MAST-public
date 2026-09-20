"""Nanonis file readers for offline data access.

Supports .sxm (scan), .3ds (grid spectroscopy), and .dat (point spectroscopy).

References: pySPM (SXM.py) + stmpy (io.py).
"""

from __future__ import annotations

import logging
import math
import os
import re
import struct
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Upper bound for reading an entire Nanonis file into memory.  Real .sxm /
# .3ds / .dat files top out at a few hundred MB even for large grids; anything
# beyond this is almost certainly a corrupt header pointing at a runaway size
# or a non-Nanonis file handed to the wrong reader.  We refuse such files with
# a clear error instead of letting ``f.read()`` exhaust RAM.
MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB

# Hard ceiling on the number of array elements any single reader will allocate
# from header-declared dimensions.  Guards against a corrupt header claiming a
# multi-billion-element grid (which would raise MemoryError deep inside numpy).
MAX_GRID_ELEMENTS = 512 * 1024 * 1024  # 512M float64 elements ≈ 4 GiB


def parse_frame_grab(parsed, *, shape_2d: bool = False):
    """Extract a channel's samples from a ``Scan_FrameDataGrab`` reply.

    On real hardware the reply body is a HETEROGENEOUS list
    ``[name_len, name(str), rows(int), cols(int), data_2D(ndarray), dir(int)]``.
    Calling ``np.asarray(body)`` on the WHOLE list raises "inhomogeneous shape"
    (mixes int/str/ndarray), which made the CheckScanForCrash / drift parses fail
    on EVERY real Nanonis scan while only "working" on flat-list stubs. This was
    fixed once in full_scan._grab_channel_array (2026-06-29) but the fix was
    never propagated to scan_frame.py / drift_track.py.

    Returns the samples as a float64 ndarray (raveled 1-D by default, or 2-D
    ``(rows, cols)`` when ``shape_2d`` and the header dims are usable), or
    ``None`` when there's no usable channel data.
    """
    if not (isinstance(parsed, (list, tuple)) and len(parsed) > 2):
        return None
    body = parsed[2]
    rows = cols = None
    arr = None
    if isinstance(body, np.ndarray):
        arr = body.astype(np.float64) if body.size else None
    elif isinstance(body, (list, tuple)) and body:
        # Pull the ndarray element (the 2-D data), and the header dims if present.
        ints = [x for x in body
                if isinstance(x, (int,)) and not isinstance(x, bool)]
        for el in body:
            if isinstance(el, np.ndarray):
                arr = el.astype(np.float64) if el.size else None
                break
        if arr is None and not any(isinstance(x, str) for x in body):
            # Flat-list fallback (stub / flat instruments): pure numeric body.
            try:
                nums = [float(x) for x in body
                        if isinstance(x, (int, float)) and not isinstance(x, bool)]
            except (TypeError, ValueError):
                nums = []
            if nums:
                arr = np.asarray(nums, dtype=np.float64)
        # header rows/cols: [name_len, name, rows, cols, data, dir] → ints[1], ints[2]
        if len(ints) >= 3:
            rows, cols = ints[1], ints[2]
    if arr is None or arr.size == 0:
        return None
    if shape_2d:
        if arr.ndim == 2:
            return arr
        if rows and cols and rows * cols == arr.size:
            return arr.reshape(int(rows), int(cols))
        # Best-effort square-ish reshape for a raveled trace.
        n = arr.size
        side = int(round(n ** 0.5))
        if side * side == n:
            return arr.reshape(side, side)
        return None  # can't form a 2-D image reliably
    return arr.ravel()


def scalar_float(x) -> "float | None":
    """Coerce one decoded Nanonis scalar to ``float``, or ``None``.

    Sibling of :func:`scalar_int`, and it **refuses the same things**:

    * bare number, 1-element list/tuple/ndarray, numpy scalar → the value;
    * **multi-element sequence → ``None``**, NOT ``seq[0]``;
    * string/bytes, ``None``, non-numeric → ``None``;
    * ``NaN`` / ``inf`` → ``None`` (a non-finite reading is not a measurement).

    ⚠️ **The refusals are the whole point — do not "helpfully" widen this.**
    A converter that accepts everything is a machine for turning a shape error
    into a plausible wrong number. Taking ``seq[0]`` from a 2-element reply
    would silently pick one of two channels; returning ``None`` makes the
    caller decide, and callers here already know how to say "could not read".

    ⚠️ It also never returns ``0.0`` as a fallback. Two of the three
    hand-written ``_scalar`` copies this replaces did (``float(v[0]) if v else
    0.0``) — a failed read becoming a plausible measurement is exactly the
    defect class catalogued in ``90_教训与反例.md``. Nesting is unwrapped
    recursively, so ``((1.5,),)`` still reads as ``1.5``.
    """
    if isinstance(x, (str, bytes)) or isinstance(x, bool):
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) != 1:
            return None
        return scalar_float(x[0])
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def frame_acquired_lines(parsed) -> "tuple[int, int] | None":
    """``(lines_acquired, lines_in_buffer)`` from a ``Scan_FrameDataGrab`` reply.

    Nanonis fills NOT-YET-ACQUIRED scan-buffer rows with NaN, so the NaN front IS
    the scan front and counting rows that are not entirely NaN measures progress
    exactly. That is not a guess: it is the measurement the scan-vision monitor
    was rebuilt around after the clock-based estimate was caught burning every
    milestone in the first slice of a scan (``vision/scan_monitor.py::_measure_acquired_frac``).

    Returns ``None`` when the reply carries no usable 2-D frame — the caller must
    then treat progress as UNKNOWN, never as zero.

    **Difference from ``_measure_acquired_frac``, on purpose.** That function
    returns ``None`` when the frame contains no NaN rows at all, because it must
    not *claim* 100 % from an ambiguous reading (a hypothetical instrument that
    backfills stale data instead of NaN-filling would look complete forever).
    Here the same reading is reported as ``(rows, rows)``: this function answers
    "is anything visibly missing?", and on such an instrument the honest answer
    degrades to "nothing visibly missing", which is exactly the pre-existing
    behaviour rather than a new false alarm. Callers that need the distinction
    should ask whether ``lines_acquired == lines_in_buffer`` themselves.
    """
    arr = parse_frame_grab(parsed, shape_2d=True)
    if arr is None or arr.ndim != 2 or arr.size == 0:
        return None
    rows = int(arr.shape[0])
    blank = np.isnan(arr).all(axis=1)
    return int((~blank).sum()), rows


def reply_body(return_value):
    """剥掉 Nanonis 回包的 (error, raw_bytes, body) 信封，返回 body。

    已是 body 或裸值时原样返回。统一解包避免各调用方对空 body、扁平
    元组数组和信封形状作出不同解释。仅第三项是序列不足以判定信封：
    ((a,), (b,), (c,)) 是扁平数值数组，不应被剥成 c。
    """
    if not (isinstance(return_value, (list, tuple)) and len(return_value) >= 3):
        return return_value
    head, raw, body = return_value[0], return_value[1], return_value[2]
    # 信封的**头两段**有固定形状:错误串(str)+ 原始字节(bytes)。
    # 只看「第三项是不是序列」会把扁平的 1-元组串 ``((1,),(2,),(3,))``
    # 当成信封剥掉,只剩 (3,) —— 第一版就是这么写的,docstring 里写了要防
    # 这个,实现却没防住。判据必须落在**能区分两者的那个分量**上。
    if not isinstance(head, str) or not isinstance(raw, (bytes, bytearray)):
        return return_value
    if isinstance(body, (list, tuple, np.ndarray)) and len(body) >= 1:
        return body
    return return_value


def decode_reply(return_value):
    """返回回包实际载荷：剥掉信封后再展开单元素 body。

    单值设定点返回标量，多值增益或限值保持列表。共用解包函数，
    避免把整个三段信封当成仪器读数或控制器状态。
    """
    body = reply_body(return_value)
    if body is return_value:
        return return_value
    payload = list(body)
    return payload[0] if len(payload) == 1 else payload


def scalar_int_from_reply(return_value):
    """将回包解为 int：先剥信封，再展开单元素载荷；失败返回 None。

    scalar_int 只接收载荷，不能直接用来处理完整回包。
    """
    body = reply_body(return_value)
    if isinstance(body, (list, tuple, np.ndarray)) and len(body) == 1:
        return scalar_int(body[0])
    return scalar_int(body)


def scalar_float_from_reply(return_value):
    """同上,但要 ``float``。"""
    body = reply_body(return_value)
    if isinstance(body, (list, tuple, np.ndarray)) and len(body) == 1:
        return scalar_float(body[0])
    return scalar_float(body)


def scalar_int(x):
    """Coerce one Nanonis scalar to ``int``, unwrapping a 1-element sequence.

    Returns ``None`` when it cannot be coerced — callers decide what a missing
    value means. NEVER raises: the whole point is that an unexpected reply shape
    must not blow up inside a skill's ``execute`` (see :func:`parse_buffer_get`).
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) != 1:
            return None
        x = x[0]
    if isinstance(x, (str, bytes)):
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def channel_ids_from_buffer(body) -> list[int]:
    """Extract channel ids from a Scan_BufferGet reply body.

    The body contains channel count, channel indexes, pixels and lines. Array
    indexes may be integers or one-element tuples depending on the parser path.
    Accept those forms, ndarrays and scalars; skip unreadable elements without
    losing the valid channel identifiers."""
    if body is None:
        return []
    if not isinstance(body, (list, tuple, np.ndarray)) or len(body) < 2:
        return []
    raw = body[1]
    if isinstance(raw, (str, bytes)):
        return []
    if not isinstance(raw, (list, tuple, np.ndarray)):
        # Single channel returned unwrapped.
        n = scalar_int(raw)
        return [] if n is None else [n]
    ids: list[int] = []
    for item in raw:
        n = scalar_int(item)
        if n is not None:
            ids.append(n)
    return ids


def parse_buffer_get(parsed) -> "dict | None":
    """Parse a full ``Scan_BufferGet`` reply into
    ``{num_channels, channel_indexes, pixels, lines}``.

    ``parsed`` is the raw ``return_value`` — ``(error_str, raw_bytes, body)``.
    Returns ``None`` when the reply is not a usable buffer-get reply at all, so
    callers can tell "cannot parse" (refuse to write) apart from "parsed fine,
    but no channels are selected" (a different, also-refusable condition).

    ``num_channels`` is what the INSTRUMENT declared (``body[0]``), reported
    verbatim rather than recomputed as ``len(channel_indexes)`` — if the two ever
    disagree, that is a parse failure the caller deserves to see, not one to
    paper over.

    Channel ids come from :func:`channel_ids_from_buffer`; ``num_channels`` /
    ``pixels`` / ``lines`` go through the same tolerant scalar coercion, so a
    future firmware that wraps THOSE in 1-tuples too degrades to ``None``
    instead of raising.
    """
    if not isinstance(parsed, (list, tuple)) or len(parsed) <= 2:
        return None
    body = parsed[2]
    if not isinstance(body, (list, tuple, np.ndarray)) or len(body) < 4:
        return None
    return {
        "num_channels": scalar_int(body[0]),
        "channel_indexes": channel_ids_from_buffer(body),
        "pixels": scalar_int(body[2]),
        "lines": scalar_int(body[3]),
    }


def _checked_file_size(path: str) -> int:
    """Return the on-disk size of ``path`` after enforcing :data:`MAX_FILE_BYTES`.

    Raises:
        FileNotFoundError: propagated from ``os.stat`` if the path is missing.
        ValueError: if the file is larger than the supported maximum.
    """
    size = os.path.getsize(path)
    if size > MAX_FILE_BYTES:
        raise ValueError(
            f"File too large to read ({size} bytes > {MAX_FILE_BYTES} byte limit): "
            f"{path}. Refusing to load to avoid exhausting memory; if this is a "
            f"genuine file, raise MAX_FILE_BYTES or stream it instead."
        )
    return size


def load_scan_file(path: str) -> dict:
    """Auto-detect file type and load.

    Returns dict with data arrays and metadata.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".sxm":
        return read_sxm(path)
    elif ext == ".3ds":
        return read_3ds(path)
    elif ext == ".dat":
        return read_dat(path)
    elif ext == ".sm4":
        return read_sm4(path)
    elif ext in (".txt", ".csv", ".asc", ".tsv", ".xyz"):
        return read_txt(path)
    else:
        raise ValueError(
            f"Unsupported file type: {ext}. Use .sxm, .3ds, .dat, .sm4, or .txt/.csv"
        )


_TXT_COMMENT_PREFIXES = ("#", "%", ";", "!", "//")


def read_txt(path: str) -> dict:
    """Read a generic delimited numeric text file with NO Nanonis ``[DATA]`` marker.

    Handles the everyday case ``read_dat`` cannot: a plain ``.txt`` / ``.csv`` /
    ``.asc`` / ``.tsv`` matrix of numbers (Gwyddion/Origin/WSxM export, a saved
    height map, or columnar spectra) that has no ``[DATA]`` section. Comment lines
    (``# % ; ! //``) and a single non-numeric column-header row are skipped.

    Returns:
        header  : {}  (a generic text file carries no structured header)
        columns : {col_name | column_j: 1D ndarray} — column view, for line plots
        matrix  : 2D ndarray of the numeric rows      — raw view, for image heatmaps
    """
    _checked_file_size(path)
    p = Path(path)
    delimiter = "," if p.suffix.lower() == ".csv" else None
    with open(path, "r", errors="replace") as f:
        lines = f.readlines()

    def _split(s: str) -> list[str]:
        toks = s.split(delimiter) if delimiter else s.split()
        return [t for t in toks if t.strip() != ""]

    column_names: list[str] = []
    rows: list[list[float]] = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith(_TXT_COMMENT_PREFIXES):
            continue
        toks = _split(s)
        try:
            rows.append([float(t) for t in toks])
        except ValueError:
            # A non-numeric line is treated as the column-header row (keep latest).
            column_names = [t.strip() for t in toks]
            continue

    if not rows:
        # NEVER return an empty matrix here. It used to, and the emptiness then
        # travelled downstream until numpy raised
        #   "zero-size array to reduction operation fmin which has no identity"
        # at some unrelated call site — an error an agent can neither act on nor
        # attribute, so it retries the same wrong file (# data_processing ran load_scan on artifacts/tool_returns/AcquireSTS_*.txt).
        # Fail here, where we still know WHICH file and WHY.
        head = ""
        for line in lines[:5]:
            s = line.strip()
            if s:
                head = s
                break
        hint = ""
        if head.startswith(("{", "[")):
            # A skill's tool-return sidecar: repr()/JSON of a result dict, not a
            # data table. The spectrum really is in there, just not as columns.
            hint = (" This looks like a tool-return record (a dict/JSON dump), "
                    "not a data table — it is NOT a scan file. Read the "
                    "instrument's own saved file instead (.sxm for images, .dat "
                    "for spectra), or parse this one as JSON rather than loading "
                    "it as a scan.")
        raise ValueError(
            f"No numeric rows found in {p.name}: every line failed to parse as "
            f"numbers.{hint} First non-blank line: {head[:120]!r}"
        )

    # NaN-pad ragged rows to the modal width — same policy as read_dat so a single
    # truncated/extra line never discards the whole dataset (NumPy 2.x raises on
    # ragged np.array).
    widths = [len(r) for r in rows]
    if len(set(widths)) > 1:
        from collections import Counter
        ncols = Counter(widths).most_common(1)[0][0]
        logger.warning(
            "txt %s: ragged rows (widths %s); normalising to %d cols with NaN pad",
            path, sorted(set(widths)), ncols,
        )
        rows = [
            r[:ncols] if len(r) >= ncols else r + [float("nan")] * (ncols - len(r))
            for r in rows
        ]
    matrix = np.array(rows, dtype=np.float64)

    columns: dict[str, np.ndarray] = {}
    for j in range(matrix.shape[1]):
        name = column_names[j] if j < len(column_names) else f"column_{j}"
        columns[name] = matrix[:, j]

    return {"header": {}, "columns": columns, "matrix": matrix}


# ---------------------------------------------------------------------------
# SM4 (RHK)
# ---------------------------------------------------------------------------
# Ported from HamsterPark/Nanonis-RHK-SPM-PyTools (sxm_preview.py: read_sm4_z_forward
# + _read_sm4_* helpers). Walks the SM4 file-object list + page-index tree, decodes
# topographic image pages (forward AND backward), applies zscale/zoffset for physical
# heights and the xscale/yscale sign flips. Returns the SAME dict shape as read_sxm —
# {header, channels: {name: {"forward": 2D, "backward": 2D}}} — so the shared loaders
# (load_image_2d) and the data viewer handle .sm4 transparently. Topographic IMAGE
# pages only (not SM4 spectroscopy / line pages).

SM4_OBJECT_PAGE_INDEX_HEADER = 1
SM4_OBJECT_PAGE_INDEX_ARRAY = 2
SM4_OBJECT_PAGE_HEADER = 3
SM4_OBJECT_PAGE_DATA = 4
SM4_PAGE_DATA_IMAGE = 0
SM4_PAGE_TOPOGRAPHIC = 1
SM4_SCAN_FORWARD = 0
SM4_FLOAT_LINE_TYPES = frozenset({1, 6, 9, 10, 11, 13, 18, 19, 21, 22})
# A corrupt header could claim billions of pages; cap the page walk.
_SM4_MAX_PAGES = 4096


def _sm4_arr(handle, dtype, count):
    data = np.fromfile(handle, dtype=dtype, count=count)
    if data.size != count:
        raise ValueError("Unexpected end of file while reading SM4 data")
    return data


def _sm4_val(handle, dtype):
    return _sm4_arr(handle, dtype, 1)[0]


def _sm4_objects(handle, count):
    objs = []
    for _ in range(count):
        oid = int(_sm4_val(handle, np.uint32))
        off = int(_sm4_val(handle, np.uint32))
        size = int(_sm4_val(handle, np.uint32))
        objs.append((oid, off, size))
    return objs


def _sm4_find(objects, target_id):
    for oid, off, _size in objects:
        if oid == target_id:
            return off
    return None


def _sm4_page_header(handle, offset):
    handle.seek(offset, 0)
    _sm4_val(handle, np.uint16)  # field size
    _sm4_val(handle, np.uint16)  # string count
    page_type = int(_sm4_val(handle, np.uint32))
    _sm4_val(handle, np.uint32)  # data sub source
    line_type = int(_sm4_val(handle, np.uint32))
    _sm4_val(handle, np.uint32)  # xcorner
    _sm4_val(handle, np.uint32)  # ycorner
    xsize = int(_sm4_val(handle, np.uint32))
    ysize = int(_sm4_val(handle, np.uint32))
    _sm4_val(handle, np.uint32)  # image type
    scan_type = int(_sm4_val(handle, np.uint32))
    _sm4_val(handle, np.uint32)  # group id
    page_data_size = int(_sm4_val(handle, np.uint32))
    _sm4_val(handle, np.uint32)  # min z
    _sm4_val(handle, np.uint32)  # max z
    xscale = float(_sm4_val(handle, np.float32))
    yscale = float(_sm4_val(handle, np.float32))
    zscale = float(_sm4_val(handle, np.float32))
    _sm4_val(handle, np.float32)  # xyscale
    _sm4_val(handle, np.float32)  # xoffset
    _sm4_val(handle, np.float32)  # yoffset
    zoffset = float(_sm4_val(handle, np.float32))
    _sm4_val(handle, np.float32)  # period
    _sm4_val(handle, np.float32)  # bias
    _sm4_val(handle, np.float32)  # current
    _sm4_val(handle, np.uint32)  # color info count
    _sm4_val(handle, np.uint32)  # grid xsize
    _sm4_val(handle, np.uint32)  # grid ysize
    _sm4_val(handle, np.uint32)  # object list count
    _sm4_val(handle, np.uint8)   # data flag
    _sm4_arr(handle, np.uint8, 3)   # reserved flags
    _sm4_arr(handle, np.uint8, 60)  # reserved
    if xsize <= 0 or ysize <= 0:
        return None
    return {
        "page_type": page_type, "scan_type": scan_type, "line_type": line_type,
        "xsize": xsize, "ysize": ysize, "xscale": xscale, "yscale": yscale,
        "zscale": zscale, "zoffset": zoffset, "page_data_size": page_data_size,
    }


def _sm4_image(handle, data_offset, info):
    page_data_size = info["page_data_size"]
    if page_data_size <= 0:
        return None
    xsize, ysize = info["xsize"], info["ysize"]
    if xsize * ysize > MAX_GRID_ELEMENTS:
        raise ValueError(
            f"SM4 page declares {xsize * ysize} elements, exceeding the "
            f"{MAX_GRID_ELEMENTS}-element safety limit"
        )
    count = page_data_size // 4
    if count <= 0:
        return None
    handle.seek(data_offset, 0)
    dtype = np.float32 if info["line_type"] in SM4_FLOAT_LINE_TYPES else np.int32
    raw = np.fromfile(handle, dtype=dtype, count=count)
    expected = xsize * ysize
    if raw.size < expected:
        return None
    raw = raw[:expected]
    data = raw.reshape((xsize, ysize))
    if info["xscale"] < 0:
        data = np.flip(data, axis=1)
    if info["yscale"] > 0:
        data = np.flip(data, axis=0)
    return data.astype(np.float64) * info["zscale"] + info["zoffset"]


def read_sm4(path: str) -> dict:
    """Read an RHK .sm4 file's topographic image pages (forward + backward).

    Returns the same shape as :func:`read_sxm` —
    ``{header, channels: {name: {"forward": 2D, "backward": 2D}}}`` — so the shared
    loaders + data viewer treat .sm4 like any other scan. ``header`` carries
    ``scan_pixels`` + ``pixel_size_m``. Topographic image pages only.
    """
    _checked_file_size(path)
    fwd_ref = None  # (data_offset, info) for the forward topo page
    bwd_ref = None  # ... backward
    empty = {"header": {"format": "sm4"}, "channels": {}}
    with open(path, "rb") as handle:
        header_size = int(_sm4_val(handle, np.uint16))
        _sm4_arr(handle, np.uint16, 18)
        _sm4_val(handle, np.uint32)  # total page count
        object_list_count = int(_sm4_val(handle, np.uint32))
        _sm4_val(handle, np.uint32)  # object field size
        _sm4_arr(handle, np.uint32, 2)

        handle.seek(header_size + 2, 0)
        file_objects = _sm4_objects(handle, object_list_count)
        pih = _sm4_find(file_objects, SM4_OBJECT_PAGE_INDEX_HEADER)
        if pih is None:
            logger.warning("sm4 %s: missing page index header", path)
            return empty

        handle.seek(pih, 0)
        page_count = int(_sm4_val(handle, np.uint32))
        page_index_obj_count = int(_sm4_val(handle, np.uint32))
        _sm4_arr(handle, np.uint32, 2)
        page_index_objects = _sm4_objects(handle, page_index_obj_count)
        pia = _sm4_find(page_index_objects, SM4_OBJECT_PAGE_INDEX_ARRAY)
        if pia is None:
            logger.warning("sm4 %s: missing page index array", path)
            return empty

        handle.seek(pia, 0)
        # Collect page references first (reading a page header seeks away, so we
        # restore position each iteration); decode the image bytes AFTER the walk
        # so an image read's seek cannot corrupt the next page entry.
        for _ in range(min(page_count, _SM4_MAX_PAGES)):
            _sm4_arr(handle, np.uint16, 8)
            page_data_type = int(_sm4_val(handle, np.uint32))
            _sm4_val(handle, np.uint32)  # page source type
            page_obj_count = int(_sm4_val(handle, np.uint32))
            _sm4_val(handle, np.uint32)  # minor version
            page_objects = _sm4_objects(handle, page_obj_count)
            if page_data_type != SM4_PAGE_DATA_IMAGE:
                continue
            header_offset = _sm4_find(page_objects, SM4_OBJECT_PAGE_HEADER)
            data_offset = _sm4_find(page_objects, SM4_OBJECT_PAGE_DATA)
            if header_offset is None or data_offset is None:
                continue
            pos = handle.tell()
            info = _sm4_page_header(handle, header_offset)
            handle.seek(pos, 0)  # restore for the next page entry
            if info is None or info["page_type"] != SM4_PAGE_TOPOGRAPHIC:
                continue
            if info["scan_type"] == SM4_SCAN_FORWARD:
                if fwd_ref is None:
                    fwd_ref = (data_offset, info)
            elif bwd_ref is None:
                bwd_ref = (data_offset, info)

        forward = _sm4_image(handle, *fwd_ref) if fwd_ref else None
        backward = _sm4_image(handle, *bwd_ref) if bwd_ref else None

    channels: dict = {}
    z: dict = {}
    if forward is not None:
        z["forward"] = forward
    if backward is not None:
        z["backward"] = backward
    if z:
        channels["Z"] = z

    info_keep = (fwd_ref or bwd_ref or (None, None))[1]
    header: dict = {"format": "sm4"}
    if info_keep is not None:
        header["scan_pixels"] = [info_keep["xsize"], info_keep["ysize"]]
        header["pixel_size_m"] = [abs(info_keep["xscale"]), abs(info_keep["yscale"])]
    return {"header": header, "channels": channels}


def read_sxm(path: str) -> dict:
    """Read Nanonis .sxm scan file.

    Returns dict with:
        header: dict of header fields
        channels: dict of {channel_name: {"forward": 2D array, "backward": 2D array}}
    """
    _checked_file_size(path)
    with open(path, "rb") as f:
        content = f.read()

    # Find header end marker (\x1a\x04, sometimes written as literal \1A\04)
    marker = b"\\1A\\04"
    header_end = content.find(marker)
    if header_end < 0:
        marker = b"\x1a\x04"
        header_end = content.find(marker)
    if header_end < 0:
        raise ValueError(f"Cannot find header end marker in {path}")

    header_raw = content[:header_end].decode("utf-8", errors="replace")
    data_raw = content[header_end + len(marker):]

    # Parse header
    header = _parse_sxm_header(header_raw)

    # scan_pixels may be missing or contain non-integer tokens (handled in the
    # parser, which stores nothing for unparseable values).  Coerce defensively.
    pixels = header.get("scan_pixels", [0, 0])
    try:
        nx = int(pixels[0])
        ny = int(pixels[1])
    except (ValueError, TypeError, IndexError):
        logger.warning("sxm %s: unparseable scan_pixels %r; no channels read", path, pixels)
        return {"header": header, "channels": {}}

    # A non-positive pixel count is meaningless.  Guarding with ``<= 0`` (not
    # ``== 0``) is critical: a negative nx/ny would be silently re-interpreted
    # by ndarray.reshape as a "-1 infer this dim" wildcard, fabricating a frame
    # of the wrong shape from arbitrary bytes.
    if nx <= 0 or ny <= 0:
        if nx < 0 or ny < 0:
            logger.warning("sxm %s: negative scan_pixels (%d, %d); no channels read", path, nx, ny)
        return {"header": header, "channels": {}}

    # Parse channel info
    channel_names = header.get("channel_names", [])
    channel_directions = header.get("channel_directions", [])
    n_channels = len(channel_names)

    # Data is stored as float32, channel by channel. A "both" channel stores a
    # forward THEN a backward frame; a "fwd"/"bwd" channel stores ONE frame. The
    # old reader always consumed two frames per channel, so a single-direction
    # channel desynced the offset and shifted every subsequent channel's data
    # — honour the per-channel Direction from DATA_INFO.
    pixels_per_frame = nx * ny
    channels = {}

    offset = 0
    for i, ch_name in enumerate(channel_names):
        ch_data = {}
        direction_flag = (channel_directions[i]
                          if i < len(channel_directions) else "both")
        if "fwd" in direction_flag and "bwd" not in direction_flag:
            dirs = ["forward"]
        elif "bwd" in direction_flag and "fwd" not in direction_flag:
            dirs = ["backward"]
        else:
            dirs = ["forward", "backward"]  # "both" (default)
        for direction in dirs:
            n_bytes = pixels_per_frame * 4
            if offset + n_bytes > len(data_raw):
                break
            frame = np.frombuffer(
                data_raw[offset:offset + n_bytes], dtype=">f4"
            ).reshape(ny, nx).astype(np.float64)
            ch_data[direction] = frame
            offset += n_bytes
        channels[ch_name] = ch_data

    return {"header": header, "channels": channels}


def read_sxm_header(path: str, *, max_header_bytes: int = 1_048_576) -> dict:
    """Parse ONLY the .sxm text header — channel/geometry metadata WITHOUT
    reading the (up to hundreds of MB) binary frame block.

    For scan discovery (list_scan_dir / glob_scans) we want each file's channels
    and frame geometry to populate the scan registry, but loading every full
    frame just to list a directory would be wasteful. This reads at most
    ``max_header_bytes`` (headers are a few KB), finds the ``\\x1a\\x04`` end
    marker, and returns the same header dict :func:`read_sxm` would — or ``{}``
    if the marker isn't within the cap. Best-effort: never raises on a
    truncated/corrupt/oversized file.
    """
    try:
        with open(path, "rb") as f:
            chunk = f.read(max(0, int(max_header_bytes)))
    except OSError:
        return {}
    marker = b"\\1A\\04"
    header_end = chunk.find(marker)
    if header_end < 0:
        header_end = chunk.find(b"\x1a\x04")
    if header_end < 0:
        return {}  # header longer than the cap, or not an .sxm — give up cheaply
    header_raw = chunk[:header_end].decode("utf-8", errors="replace")
    return _parse_sxm_header(header_raw)


def sxm_frame_meta(header: dict) -> dict:
    """Compact, JSON-friendly frame descriptor from a parsed .sxm header —
    ``channels`` plus whichever of ``scan_offset`` / ``scan_range`` /
    ``scan_pixels`` are present. Safe to store in the scan registry / a
    checkpoint (plain lists/strings, never ndarrays)."""
    out: dict = {}
    chans = header.get("channel_names")
    if chans:
        out["channels"] = list(chans)
    for k in ("scan_offset", "scan_range", "scan_pixels"):
        if k in header:
            v = header[k]
            out[k] = list(v) if isinstance(v, (list, tuple)) else v
    return out


# Best-guess physical unit per channel name. The .sxm ``:DATA_INFO:`` block does
# carry a Unit column, but ``_parse_sxm_header`` has never kept it and adding a
# key there would change what every existing consumer sees. This table covers the
# channels that actually matter for display/analysis; anything else → "".
_CHANNEL_UNIT_HINT: dict[str, str] = {
    "z": "m", "current": "A", "bias": "V", "phase": "deg",
    "amplitude": "m", "frequency shift": "Hz", "excitation": "V",
}


def _first_float(value, default: float | None = None) -> float | None:
    """First whitespace-separated float in ``value`` (headers are free text)."""
    try:
        return float(str(value).split()[0])
    except (TypeError, ValueError, IndexError):
        return default


def rows_top_first(arr, scan_dir) -> "np.ndarray":
    """One frame with **row 0 = the high-y (top) edge**, whatever ``:SCAN_DIR:`` says.

    THE single rule for that flip. It used to live, spelled differently, in three
    places (``sxm_oriented_frames`` here, ``mosaic.load_scan_for_mosaic``, and
    nowhere at all in ``webui.scan_preview`` — which is how operator 「扫图的 up 和 down 好像只有一个方向会自动显示在扫描地图上」 happened: with
    Nanonis *bouncy* on, consecutive frames alternate direction, and the scan-map
    underlay drew every second one mirrored top-to-bottom against its neighbour.

    Nanonis writes lines in ACQUISITION order. ``down`` starts at the top of the
    frame, so row 0 is already the top; ``up`` starts at the bottom, so row 0 is
    the bottom and the frame needs ``flipud``.

    ``scan_dir`` unreadable (missing header field, ``None``, junk) → returned
    UNCHANGED, and that is a decision, not an oversight: the identity is the only
    operation that does not claim knowledge we do not have. ``mosaic`` used to
    flip on ``!= "down"``, i.e. it flipped files whose direction it could not
    read — the two rules agree on every real Nanonis file (they all say up or
    down) and disagree only where the evidence is missing, which is exactly where
    the quieter answer belongs. **To overturn this you need to answer**: is there
    a producer that writes .sxm WITHOUT ``:SCAN_DIR:`` whose rows are bottom-first?
    None has been observed here (11 local files, all up/down); if one turns up,
    the fix is to read its real convention, not to restore a coin flip.
    """
    a = np.asarray(arr)
    if str(scan_dir or "").strip().lower() == "up":
        return np.flipud(a)
    return a


def sxm_oriented_frames(scan: dict, channel: str = "Z") -> dict:
    """One channel's frames put into a COMMON geometric orientation, plus scale.

    :func:`read_sxm` deliberately returns the raw blocks exactly as Nanonis wrote
    them; this helper is the additive layer that makes two frames comparable:

    * the **backward** block is acquired along −x, so it is stored mirrored →
      ``np.fliplr``. Without this, any forward/backward comparison (registration,
      trace/retrace instability, drift) is measuring an image against its own
      mirror image and the number it returns is meaningless.
    * with ``:SCAN_DIR: up`` the first acquired line is the BOTTOM of the frame →
      ``np.flipud`` so row 0 is the top of the frame for every file. Frames taken
      in opposite scan directions are otherwise vertically flipped relative to
      each other, which silently breaks mosaics, drift tracking, and any
      row-indexed report ("tip changed near row 241" pointing at the wrong end).
      (``mast.io.mosaic`` already applies this same correction locally.)

    Returns a plain dict — ``forward`` / ``backward`` (ndarray | None), plus
    ``nm_per_px``, ``width_nm``, ``height_nm``, ``bias_v``, ``setpoint_a``,
    ``scan_dir``, ``unit``, ``rec_time``. Never raises: a missing channel yields
    ``forward=None`` and the caller decides.
    """
    header = (scan or {}).get("header") or {}
    channels = (scan or {}).get("channels") or {}
    ch = channels.get(channel)
    if ch is None:
        # Case-insensitive second chance — Nanonis names vary by rig ("Z" / "z").
        lowered = {str(k).lower(): k for k in channels}
        key = lowered.get(str(channel).lower())
        ch = channels.get(key) if key else None
    fwd = bwd = None
    if ch:
        fwd = ch.get("forward")
        bwd = ch.get("backward")
        if bwd is not None:
            bwd = np.fliplr(np.asarray(bwd))
        if fwd is None and bwd is not None:
            # bwd-only channel: the (already un-mirrored) backward block IS the
            # only frame this channel has — hand it back as ``forward``.
            fwd, bwd = bwd, None
        scan_dir = str(header.get("scan_dir", "")).strip().lower()
        if fwd is not None:
            fwd = rows_top_first(fwd, scan_dir)
        if bwd is not None:
            bwd = rows_top_first(bwd, scan_dir)

    rng = str(header.get("scan_range", ""))
    parts = rng.split()
    width_m = _first_float(parts[0] if parts else None)
    height_m = _first_float(parts[1] if len(parts) > 1 else None, width_m)
    width_nm = width_m * 1e9 if width_m else None
    height_nm = height_m * 1e9 if height_m else None
    nm_per_px = None
    if width_nm and fwd is not None and np.asarray(fwd).ndim == 2:
        cols = int(np.asarray(fwd).shape[1])
        if cols > 0:
            nm_per_px = float(width_nm) / cols

    name = str(channel).strip().lower()
    return {
        "channel": channel,
        "forward": fwd,
        "backward": bwd,
        "nm_per_px": nm_per_px,
        "width_nm": width_nm,
        "height_nm": height_nm,
        "bias_v": _first_float(header.get("bias")),
        "setpoint_a": _first_float(header.get("z-controller>setpoint")),
        "scan_dir": str(header.get("scan_dir", "")).strip().lower() or None,
        "unit": _CHANNEL_UNIT_HINT.get(name, ""),
        "rec_time": (f"{str(header.get('rec_date', '')).strip()} "
                     f"{str(header.get('rec_time', '')).strip()}").strip(),
    }


def read_3ds(path: str) -> dict:
    """Read Nanonis .3ds grid spectroscopy file.

    Layout (verified against real Nanonis V5e output):
        <CRLF-terminated text header>
        \\r\\n:HEADER_END:\\r\\n
        <binary blob>

    Per-pixel binary layout, big-endian float32:
        [ p_0, p_1, ..., p_{num_params-1},        # per-pixel params
          ch0_pt0, ch0_pt1, ..., ch0_pt{N-1},     # channel 0 over the sweep
          ch1_pt0, ...,                            # channel 1, ...
          ... ]

    The bias / sweep axis is NOT in the header on V5e — the first two
    per-pixel parameters are "Sweep Start" and "Sweep End" (named in the
    header's "Fixed parameters" field). We read them out of pixel (0,0)
    and rebuild the axis via np.linspace.

    Returns:
        header  : parsed header dict (see _parse_3ds_header)
        grid    : 3D ndarray (ny, nx, n_points) of the FIRST channel
        params  : dict with nx, ny, n_points, fixed_param_names,
                  param_array (ny, nx, num_params) of the raw per-pixel
                  parameters — exposes e.g. X(m), Y(m), Z(m), Sweep Start, ...
        bias    : 1D ndarray (n_points,) — sweep axis in volts (or whichever
                  unit the Sweep Signal is in)
    """
    _checked_file_size(path)
    with open(path, "rb") as f:
        content = f.read()

    header_end = content.find(b"\r\n:HEADER_END:\r\n")
    marker = b"\r\n:HEADER_END:\r\n"
    if header_end < 0:
        marker = b":HEADER_END:"
        header_end = content.find(marker)
    if header_end < 0:
        raise ValueError(f"Cannot find header end in {path}")

    header_raw = content[:header_end].decode("utf-8", errors="replace")
    data_raw = content[header_end + len(marker):]

    header = _parse_3ds_header(header_raw)

    empty = {"header": header, "grid": np.array([]), "params": {}, "bias": np.array([])}

    grid_dim = header.get("grid_dim", [1, 1])
    try:
        nx, ny = int(grid_dim[0]), int(grid_dim[1])
        n_points = int(header.get("points", 0))
        n_params = int(header.get("num_parameters", 0))
        n_channels = int(header.get("num_channels", 1))
    except (ValueError, TypeError, IndexError):
        logger.warning("3ds %s: unparseable grid/point/param header fields", path)
        return empty

    # Guard EVERY header-declared dimension with ``<= 0`` *before* it reaches
    # np.zeros / range().  A negative grid_dim or Points would otherwise raise
    # ``ValueError: negative dimensions are not allowed`` from np.zeros ;
    # a negative n_params would do the same for the params array.  We treat any
    # non-positive geometry as "no data" rather than crashing the caller.
    if n_points <= 0 or nx <= 0 or ny <= 0:
        return empty
    if n_params < 0 or n_channels <= 0:
        logger.warning(
            "3ds %s: non-positive num_parameters=%d / num_channels=%d; no data read",
            path, n_params, n_channels,
        )
        return empty

    # Clip the requested grid to what the binary blob can actually back, and
    # refuse pathologically large header-declared grids before np.zeros tries
    # to allocate them (a corrupt header claiming e.g. 100000x100000 would
    # raise MemoryError deep in numpy — #71).  We bound by both the element
    # ceiling and the real data length.
    floats_per_point = n_params + n_channels * n_points
    point_bytes = floats_per_point * 4
    declared_pixels = nx * ny
    grid_elements = declared_pixels * n_points
    if grid_elements > MAX_GRID_ELEMENTS:
        raise ValueError(
            f"3ds header declares {grid_elements} grid elements "
            f"(nx={nx}, ny={ny}, points={n_points}), exceeding the "
            f"{MAX_GRID_ELEMENTS}-element safety limit in {path}. Header is "
            f"likely corrupt; refusing to allocate."
        )
    # How many whole pixels does the binary blob actually contain?  When the
    # header over-states the grid (truncated/corrupt file) we keep the declared
    # (ny, nx, n_points) shape — the read loop below stops at the blob boundary
    # and leaves the missing trailing pixels zero-filled, preserving the public
    # shape contract.  The MAX_GRID_ELEMENTS check above is what bounds the
    # allocation; this only logs the mismatch and short-circuits empty blobs.
    available_pixels = len(data_raw) // point_bytes if point_bytes > 0 else 0
    if available_pixels < declared_pixels:
        logger.warning(
            "3ds %s: header declares %d pixels but blob only backs %d; "
            "trailing pixels will be zero-filled",
            path, declared_pixels, available_pixels,
        )
    if available_pixels <= 0:
        return empty

    grid_data = np.zeros((ny, nx, n_points), dtype=np.float64)
    params_data = np.zeros((ny, nx, n_params), dtype=np.float64) if n_params else None

    for iy in range(ny):
        for ix in range(nx):
            offset = (iy * nx + ix) * point_bytes
            if offset + point_bytes > len(data_raw):
                break
            values = np.frombuffer(data_raw[offset:offset + point_bytes], dtype=">f4")
            if len(values) < floats_per_point:
                break
            if n_params:
                params_data[iy, ix, :] = values[:n_params].astype(np.float64)
            ch0 = values[n_params:n_params + n_points]
            grid_data[iy, ix, :] = ch0.astype(np.float64)

    # Sweep axis: prefer header sweep_start/end (older firmwares), else read
    # them out of the per-pixel "Fixed parameters" at pixel (0,0).
    bias_start = header.get("sweep_start")
    bias_end = header.get("sweep_end")
    fixed_names = header.get("fixed_parameters", [])
    if (bias_start is None or bias_end is None) and params_data is not None:
        try:
            i_start = fixed_names.index("Sweep Start")
            i_end = fixed_names.index("Sweep End")
            bias_start = float(params_data[0, 0, i_start])
            bias_end = float(params_data[0, 0, i_end])
        except (ValueError, IndexError):
            pass
    if bias_start is None or bias_end is None:
        bias_start, bias_end = 0.0, 1.0
    bias = np.linspace(bias_start, bias_end, n_points)

    return {
        "header": header,
        "grid": grid_data,
        "params": {
            "nx": nx,
            "ny": ny,
            "n_points": n_points,
            "fixed_param_names": fixed_names,
            "experiment_param_names": header.get("experiment_parameters", []),
            "param_array": params_data,
        },
        "bias": bias,
    }


def read_dat(path: str) -> dict:
    """Read Nanonis .dat point spectroscopy file.

    File layout (verified against real Bias-Spectroscopy .dat from Nanonis V5e):
        <tab-separated key>\\t<value>
        ...
        [DATA]
        <tab-separated column header line>
        <numeric row 1>
        <numeric row 2>
        ...

    Earlier versions used `=` as the header separator which is wrong —
    real Nanonis .dat uses `\\t`. The column header is the line directly
    AFTER ``[DATA]``, not the line before it.

    Returns dict with:
        header  : dict of header fields (tab-separated key→value pairs)
        columns : dict of {column_name: 1D ndarray}
    """
    _checked_file_size(path)
    with open(path, "r", errors="replace") as f:
        lines = f.readlines()

    header: dict[str, str] = {}
    data_start = 0

    # Header: everything before the [DATA] marker.
    # Each header row is `key<TAB>value`. Some rows have only a key (e.g.
    # 'User', 'Date') — we store them with empty value for completeness.
    for i, line in enumerate(lines):
        s = line.rstrip("\r\n")
        if s.strip() == "[DATA]":
            data_start = i + 1
            break
        if not s.strip():
            continue
        parts = s.split("\t")
        key = parts[0].strip()
        if not key:
            continue
        # Multi-tab rows (rare): join everything after the first tab.
        val = "\t".join(parts[1:]).strip() if len(parts) > 1 else ""
        header[key] = val

    if data_start == 0:
        return {"header": header, "columns": {}}

    # Column header sits right after [DATA].
    column_names: list[str] = []
    if data_start < len(lines):
        col_line = lines[data_start].rstrip("\r\n")
        column_names = [c.strip() for c in col_line.split("\t") if c.strip()]
        data_start += 1

    # Parse numeric rows.
    values: list[list[float]] = []
    for line in lines[data_start:]:
        s = line.strip()
        if not s:
            continue
        try:
            row = [float(x) for x in s.split("\t")]
            values.append(row)
        except ValueError:
            continue

    if not values:
        return {"header": header, "columns": {}}

    # Rows can be ragged: a truncated final row, an extra stray column, or a
    # corrupt line that parsed to a different width than its neighbours.  In
    # NumPy 2.x ``np.array([[1,2],[3]])`` raises ValueError instead of building
    # an object array, so a single bad row would otherwise lose the ENTIRE
    # dataset.  Pad every row to the modal (most common) width with NaN and
    # drop rows that are wildly off (longer than the modal width is treated as
    # corrupt and truncated to it; shorter rows are NaN-filled).
    widths = [len(r) for r in values]
    if len(set(widths)) > 1:
        # Use the most common width as the canonical column count; this matches
        # the bulk of the sweep and isn't skewed by one stray short/long line.
        from collections import Counter
        ncols = Counter(widths).most_common(1)[0][0]
        logger.warning(
            "dat %s: ragged numeric rows (widths %s); normalising to %d columns "
            "with NaN padding",
            path, sorted(set(widths)), ncols,
        )
        normalised: list[list[float]] = []
        for row in values:
            if len(row) >= ncols:
                normalised.append(row[:ncols])
            else:
                normalised.append(row + [float("nan")] * (ncols - len(row)))
        values = normalised

    data_array = np.array(values, dtype=np.float64)
    columns: dict[str, np.ndarray] = {}
    for j, name in enumerate(column_names):
        if j < data_array.shape[1]:
            columns[name] = data_array[:, j]
    # If column header was missing or shorter than the data row, fill the
    # remaining columns with positional names so callers don't lose data.
    for j in range(len(column_names), data_array.shape[1]):
        columns[f"column_{j}"] = data_array[:, j]
    if not columns:
        for j in range(data_array.shape[1]):
            columns[f"column_{j}"] = data_array[:, j]

    return {"header": header, "columns": columns}


def _parse_sxm_header(raw: str) -> dict:
    """Parse .sxm header key-value pairs."""
    header: dict = {}
    current_key = ""

    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith(":"):
            current_key = line.strip(":").lower().replace(" ", "_")
        elif current_key:
            if current_key == "scan_pixels":
                parts = line.split()
                if len(parts) >= 2:
                    # Tokens are normally plain integers but a corrupt header may
                    # carry junk ("128.0", "NaN", "-"); don't let int() blow up
                    # the whole read — skip the field and let read_sxm fall back
                    # to the empty-channels path.
                    try:
                        header[current_key] = [int(parts[0]), int(parts[1])]
                    except ValueError:
                        logger.warning("sxm header: non-integer scan_pixels tokens %r", parts[:2])
            elif current_key == "data_info":
                if "channel_names" not in header:
                    header["channel_names"] = []
                    header["channel_directions"] = []
                parts = line.split("\t")
                # Skip header row ("Channel  Name  Unit  Direction ...")
                if len(parts) >= 2 and parts[1].strip():
                    name = parts[1].strip()
                    if name == "Name":
                        continue
                    header["channel_names"].append(name)
                    # Direction column (index 3): "both" / "fwd" / "bwd". A
                    # single-direction channel stores ONE frame, not two — the
                    # reader must honour this or the byte offset desyncs and every
                    # later channel's data is shifted.
                    direction = ""
                    if len(parts) >= 4:
                        direction = parts[3].strip().lower()
                    header.setdefault("channel_directions", []).append(
                        direction or "both")
            else:
                header[current_key] = line

    return header


def _parse_3ds_header(raw: str) -> dict:
    """Parse a Nanonis .3ds header into a structured dict.

    Recognised keys (post-normalisation):
        grid_dim              : [nx, ny]
        points                : int
        num_parameters        : int   (from `# Parameters (4 byte)=N`)
        num_channels          : int   (derived from `Channels=A;B`)
        channels              : list[str]
        fixed_parameters      : list[str]   (e.g. ["Sweep Start", "Sweep End"])
        experiment_parameters : list[str]
        sweep_start/end       : float (only if present — newer files omit)
        ...all other key=value pairs are kept as raw strings.
    """
    header: dict = {}

    for line in raw.split("\n"):
        line = line.strip().rstrip("\r")
        if "=" not in line:
            continue
        key_raw, _, val = line.partition("=")
        val = val.strip()
        # Strip a single layer of surrounding double quotes ("..." → ...).
        if len(val) >= 2 and val.startswith('"') and val.endswith('"'):
            val = val[1:-1]

        key_lower = key_raw.strip().lower()

        # `# Parameters (4 byte)=12` → num_parameters
        if key_lower.startswith("# parameters"):
            try:
                header["num_parameters"] = int(val)
            except ValueError:
                pass
            continue

        if key_lower == "grid dim":
            parts = val.replace("x", " ").split()
            try:
                if len(parts) >= 2:
                    header["grid_dim"] = [int(parts[0]), int(parts[1])]
            except ValueError:
                pass
            continue

        if key_lower == "points":
            try:
                header["points"] = int(val)
            except ValueError:
                pass
            continue

        if key_lower == "channels":
            ch_list = [c.strip() for c in val.split(";") if c.strip()]
            header["channels"] = ch_list
            header["num_channels"] = len(ch_list)
            continue

        if key_lower == "fixed parameters":
            header["fixed_parameters"] = [p.strip() for p in val.split(";") if p.strip()]
            continue

        if key_lower == "experiment parameters":
            header["experiment_parameters"] = [p.strip() for p in val.split(";") if p.strip()]
            continue

        if key_lower in ("sweep start", "sweep end"):
            try:
                header[key_lower.replace(" ", "_")] = float(val)
            except ValueError:
                pass
            continue

        # Generic fallback: snake-cased key, raw string value.
        normalised = key_lower.replace(" ", "_")
        header[normalised] = val

    return header
