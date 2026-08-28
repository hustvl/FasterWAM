"""MessagePack wire format shared with the Trossen remote-policy client."""

from __future__ import annotations

import io
from typing import Any

import msgpack
import numpy as np
from PIL import Image

_NDARRAY_KEY = "__ndarray__"
_JPEG_KEY = "__jpeg__"


def _default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        array = np.ascontiguousarray(obj)
        return {
            _NDARRAY_KEY: True,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data": array.tobytes(),
        }
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Cannot serialize object of type {type(obj)!r}")


def _decode_jpeg(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _object_hook(obj: dict[str, Any]) -> Any:
    if obj.get(_NDARRAY_KEY):
        array = np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"]))
        return array.reshape(obj["shape"]).copy()
    if _JPEG_KEY in obj:
        return _decode_jpeg(obj[_JPEG_KEY])
    return obj


def packb(obj: Any) -> bytes:
    return msgpack.packb(obj, default=_default, use_bin_type=True)


def unpackb(data: bytes | bytearray) -> Any:
    return msgpack.unpackb(data, object_hook=_object_hook, raw=False, strict_map_key=False)
