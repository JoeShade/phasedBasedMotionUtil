"""This file keeps optional CuPy capability detection and one explicit CPU/GPU backend bridge so hot-path worker/core code can share real acceleration without pushing device logic into the UI."""

from __future__ import annotations

import os
import site
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


_REGISTERED_CUDA_DLL_DIRECTORIES: set[str] = set()
_CUDA_DLL_DIRECTORY_HANDLES: list[Any] = []


@dataclass(frozen=True)
class AccelerationCapability:
    """This model records whether the optional CuPy backend can actually be used on the current machine."""

    backend_name: str | None
    importable: bool
    usable: bool
    status: str
    detail: str
    installed_version: str | None = None
    device_name: str | None = None
    device_total_bytes: int | None = None
    device_free_bytes: int | None = None


@dataclass(frozen=True)
class AccelerationDecision:
    """This model captures the operator request plus the effective runtime choice that pre-flight and the worker will use."""

    requested: bool
    active: bool
    status: str
    detail: str
    backend_name: str | None = None
    device_name: str | None = None


@dataclass
class TransferBuffer:
    """This small wrapper keeps one writable host buffer alive while decode/encode passes a memoryview between stages."""

    buffer: object
    size_bytes: int

    def writable_view(self) -> memoryview:
        return memoryview(self.buffer)[: self.size_bytes]

    def slice(self, valid_bytes: int) -> memoryview:
        return self.writable_view()[:valid_bytes]

    def numpy_uint8_view(self, valid_bytes: int) -> np.ndarray:
        return np.frombuffer(self.buffer, dtype=np.uint8, count=valid_bytes)


class CpuProcessingBackend:
    """This backend keeps the authoritative NumPy path explicit and unchanged for CPU-only installs."""

    name = "cpu"
    active = False
    xp = np
    device_name: str | None = None

    def __init__(self, decision: AccelerationDecision) -> None:
        self.decision = decision

    def allocate_transfer_buffer(self, size_bytes: int) -> None:
        return None

    def asarray(self, array: Any, *, dtype: Any | None = None, copy: bool = False) -> np.ndarray:
        if dtype is None:
            return np.array(array, copy=copy)
        return np.array(array, dtype=dtype, copy=copy)

    def to_host(self, array: Any) -> np.ndarray:
        return np.asarray(array)

    def bytes_to_float_frames(
        self,
        frame_bytes: bytes | bytearray | memoryview,
        *,
        width: int,
        height: int,
    ) -> np.ndarray:
        frame_size = width * height * 3
        frame_count = len(frame_bytes) // max(frame_size, 1)
        array = np.frombuffer(frame_bytes, dtype=np.uint8, count=frame_count * frame_size)
        frames = array.reshape(frame_count, height, width, 3)
        float_frames = np.empty(frames.shape, dtype=np.float32)
        np.multiply(
            frames,
            np.float32(1.0 / 255.0),
            out=float_frames,
            casting="unsafe",
        )
        return float_frames

    def float_frames_to_rgb24_bytes(
        self,
        frames: np.ndarray,
        *,
        transfer_buffer: TransferBuffer | None = None,
    ) -> bytes:
        del transfer_buffer
        clipped = np.clip(frames, 0.0, 1.0)
        frame_uint8 = np.rint(clipped * np.float32(255.0)).astype(np.uint8)
        return frame_uint8.tobytes()

    def synchronize(self) -> None:
        return None

    def release_unused_memory(self) -> None:
        return None


class CupyProcessingBackend:
    """This backend keeps the CuPy path close to the existing NumPy structure while reusing device-resident chunks and reference state."""

    name = "cupy"
    active = True

    def __init__(self, decision: AccelerationDecision) -> None:
        self.decision = decision
        self.device_name = decision.device_name
        self._cp = _import_cupy()
        self.xp = self._cp

    def allocate_transfer_buffer(self, size_bytes: int) -> TransferBuffer:
        return TransferBuffer(
            buffer=self._cp.cuda.alloc_pinned_memory(size_bytes),
            size_bytes=size_bytes,
        )

    def asarray(self, array: Any, *, dtype: Any | None = None, copy: bool = False):
        if isinstance(array, self._cp.ndarray):
            if dtype is None:
                return array.copy() if copy else array
            return array.astype(dtype, copy=copy)
        return self._cp.array(array, dtype=dtype, copy=copy)

    def to_host(self, array: Any) -> np.ndarray:
        if isinstance(array, self._cp.ndarray):
            return self._cp.asnumpy(array)
        return np.asarray(array)

    def bytes_to_float_frames(
        self,
        frame_bytes: bytes | bytearray | memoryview,
        *,
        width: int,
        height: int,
    ):
        frame_size = width * height * 3
        frame_count = len(frame_bytes) // max(frame_size, 1)
        host_array = np.frombuffer(
            frame_bytes,
            dtype=np.uint8,
            count=frame_count * frame_size,
        ).reshape(frame_count, height, width, 3)
        device_uint8 = self._cp.asarray(host_array)
        return device_uint8.astype(self._cp.float32) * self._cp.float32(1.0 / 255.0)

    def float_frames_to_rgb24_bytes(
        self,
        frames,
        *,
        transfer_buffer: TransferBuffer | None = None,
    ) -> bytes | memoryview:
        clipped = self._cp.clip(frames, 0.0, 1.0)
        frame_uint8 = self._cp.rint(clipped * self._cp.float32(255.0)).astype(
            self._cp.uint8,
            copy=False,
        )
        if transfer_buffer is None:
            return self._cp.asnumpy(frame_uint8).tobytes()
        host_view = transfer_buffer.numpy_uint8_view(frame_uint8.size).reshape(
            frame_uint8.shape
        )
        frame_uint8.get(out=host_view)
        self._cp.cuda.Stream.null.synchronize()
        return transfer_buffer.slice(frame_uint8.size)

    def synchronize(self) -> None:
        self._cp.cuda.Stream.null.synchronize()

    def release_unused_memory(self) -> None:
        self.synchronize()
        self._cp.get_default_memory_pool().free_all_blocks()


def clear_acceleration_capability_cache() -> None:
    detect_acceleration_capability.cache_clear()


def resolve_acceleration_request(
    requested: bool,
    capability: AccelerationCapability | None = None,
) -> AccelerationDecision:
    """Return the effective runtime choice so the UI, pre-flight, and worker all describe the same backend outcome."""

    resolved_capability = (
        detect_acceleration_capability() if capability is None else capability
    )
    if not requested:
        if resolved_capability.usable:
            detail = (
                "Hardware acceleration is available but disabled. The authoritative CPU path will be used."
            )
        else:
            detail = "Hardware acceleration is disabled. The authoritative CPU path will be used."
        return AccelerationDecision(
            requested=False,
            active=False,
            status="cpu_selected",
            detail=detail,
            backend_name=resolved_capability.backend_name,
            device_name=resolved_capability.device_name,
        )

    if resolved_capability.usable:
        device_text = resolved_capability.device_name or "the first CUDA device"
        return AccelerationDecision(
            requested=True,
            active=True,
            status="gpu_active",
            detail=f"Hardware acceleration is enabled. Using CuPy on {device_text}.",
            backend_name=resolved_capability.backend_name,
            device_name=resolved_capability.device_name,
        )

    return AccelerationDecision(
        requested=True,
        active=False,
        status="gpu_requested_cpu_fallback",
        detail=(
            "Hardware acceleration was requested, but "
            f"{resolved_capability.detail} CPU fallback will be used."
        ),
        backend_name=resolved_capability.backend_name,
        device_name=resolved_capability.device_name,
    )


def build_processing_backend(
    requested: bool,
) -> tuple[AccelerationDecision, CpuProcessingBackend | CupyProcessingBackend]:
    """Build one concrete backend instance for a render or analysis run."""

    decision = resolve_acceleration_request(requested)
    if decision.active:
        return decision, CupyProcessingBackend(decision)
    return decision, CpuProcessingBackend(decision)


@lru_cache(maxsize=1)
def detect_acceleration_capability() -> AccelerationCapability:
    """Detect whether the optional CuPy backend can execute real kernels on the current machine."""

    try:
        cp = _import_cupy()
    except ImportError:
        return AccelerationCapability(
            backend_name="cupy",
            importable=False,
            usable=False,
            status="absent",
            detail="the optional CuPy backend is not installed.",
        )

    version_text = str(getattr(cp, "__version__", "unknown"))
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        return AccelerationCapability(
            backend_name="cupy",
            importable=True,
            usable=False,
            status="installed_but_unusable",
            detail=(
                "CuPy is installed but could not query a compatible CUDA device: "
                f"{type(exc).__name__}: {exc}"
            ),
            installed_version=version_text,
        )

    if device_count < 1:
        return AccelerationCapability(
            backend_name="cupy",
            importable=True,
            usable=False,
            status="installed_but_unusable",
            detail="CuPy is installed but no compatible CUDA device was detected.",
            installed_version=version_text,
        )

    try:
        properties = cp.cuda.runtime.getDeviceProperties(0)
        device_name = _decode_device_name(properties.get("name"))
        test_array = cp.arange(4, dtype=cp.float32).reshape(2, 2)
        float(test_array.sum().item())
        cp.fft.fft2(test_array)
        cp.cuda.Stream.null.synchronize()
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    except Exception as exc:
        return AccelerationCapability(
            backend_name="cupy",
            importable=True,
            usable=False,
            status="installed_but_unusable",
            detail=(
                "CuPy is installed but the first CUDA device could not execute a test kernel: "
                f"{type(exc).__name__}: {exc}"
            ),
            installed_version=version_text,
        )

    return AccelerationCapability(
        backend_name="cupy",
        importable=True,
        usable=True,
        status="available",
        detail="CuPy and a compatible CUDA device are available.",
        installed_version=version_text,
        device_name=device_name,
        device_total_bytes=int(total_bytes),
        device_free_bytes=int(free_bytes),
    )


def _import_cupy():
    _prepare_cupy_runtime()
    import cupy as cp

    return cp


def _prepare_cupy_runtime() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    if "CUDA_PATH" not in os.environ:
        runtime_root = _packaged_cuda_runtime_root()
        if runtime_root is not None:
            os.environ["CUDA_PATH"] = str(runtime_root)
    for directory in _candidate_cuda_dll_directories():
        directory_text = str(directory)
        if directory_text in _REGISTERED_CUDA_DLL_DIRECTORIES:
            continue
        _CUDA_DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(directory_text))
        _REGISTERED_CUDA_DLL_DIRECTORIES.add(directory_text)
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        if directory_text not in path_entries:
            os.environ["PATH"] = (
                directory_text
                if not path_entries or path_entries == [""]
                else directory_text + os.pathsep + os.environ["PATH"]
            )


def _candidate_cuda_dll_directories() -> tuple[Path, ...]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def add_candidate(path: Path) -> None:
        if not path.exists():
            return
        resolved = str(path.resolve())
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(path)

    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        add_candidate(Path(cuda_path) / "bin")

    site_roots = [Path(path) for path in site.getsitepackages()]
    user_site = site.getusersitepackages()
    if user_site:
        site_roots.append(Path(user_site))
    for root in site_roots:
        for package_name in ("cuda_runtime", "cuda_nvrtc", "cufft", "nvjitlink"):
            add_candidate(root / "nvidia" / package_name / "bin")

    return tuple(candidates)


def _packaged_cuda_runtime_root() -> Path | None:
    site_roots = [Path(path) for path in site.getsitepackages()]
    user_site = site.getusersitepackages()
    if user_site:
        site_roots.append(Path(user_site))
    for root in site_roots:
        candidate = root / "nvidia" / "cuda_runtime"
        if candidate.exists():
            return candidate
    return None


def _decode_device_name(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip("\x00") or None
    return str(value)


# ######################################################################################################################
#
#
#                                         AAAAAAAA
#                                       AAAA    AAAAA              AAAAAAAA
#                                     AAA          AAA           AAAA    AAA
#                                     AA            AA          AAA       AAA
#                                     AA            AAAAAAAAAA  AAA       AAAAAAAAAA
#                                     AAA                  AAA  AAA               AA
#                                      AAA                AAA    AAAAA            AA
#                                       AAAAA            AAA        AAA           AA
#                                          AAA          AAA                       AA
#                                          AAA         AAA                        AA
#                                          AA         AAA                         AA
#                                          AA        AAA                          AA
#                                         AAA       AAAAAAAAA                     AA
#                                         AAA       AAAAAAAAA                     AA
#                                         AA                   AAAAAAAAAAAAAA     AA
#                                         AA  AAAAAAAAAAAAAAAAAAAAAAAA    AAAAAAA AA
#                                        AAAAAAAAAAA                           AA AA
#                                                                            AAA  AA
#                                                                          AAAA   AA
#                                                                       AAAA      AA
#                                                                    AAAAA        AA
#                                                                AAAAA            AA
#                                                             AAAAA               AA
#                                                         AAAAAA                  AA
#                                                     AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
#
#
# ######################################################################################################################
#
#                                                 Copyright (c) JoeShade
#                               Licensed under the GNU Affero General Public License v3.0
#
# ######################################################################################################################
#
#                                         +44 (0) 7356 042702 | joe@jshade.co.uk
#
# ######################################################################################################################
