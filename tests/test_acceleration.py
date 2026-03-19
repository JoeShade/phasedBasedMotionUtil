"""This file tests the small acceleration capability bridge so the UI, pre-flight, and worker all agree on one concrete CPU-or-GPU decision."""

from __future__ import annotations

import numpy as np

import phase_motion_app.core.acceleration as acceleration_module
from phase_motion_app.core.acceleration import (
    AccelerationCapability,
    build_processing_backend,
    resolve_acceleration_request,
)


def test_resolve_acceleration_request_reports_cpu_selection_when_disabled() -> None:
    decision = resolve_acceleration_request(
        False,
        capability=AccelerationCapability(
            backend_name="cupy",
            importable=True,
            usable=True,
            status="available",
            detail="CuPy and a compatible CUDA device are available.",
            installed_version="13.0",
            device_name="Example GPU",
        ),
    )

    assert decision.requested is False
    assert decision.active is False
    assert decision.status == "cpu_selected"
    assert decision.backend_name == "cupy"
    assert "disabled" in decision.detail


def test_resolve_acceleration_request_reports_cpu_fallback_when_backend_is_absent() -> None:
    decision = resolve_acceleration_request(
        True,
        capability=AccelerationCapability(
            backend_name="cupy",
            importable=False,
            usable=False,
            status="absent",
            detail="the optional CuPy backend is not installed.",
        ),
    )

    assert decision.requested is True
    assert decision.active is False
    assert decision.status == "gpu_requested_cpu_fallback"
    assert "CPU fallback" in decision.detail


def test_build_processing_backend_returns_cpu_backend_when_gpu_is_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        acceleration_module,
        "detect_acceleration_capability",
        lambda: AccelerationCapability(
            backend_name="cupy",
            importable=True,
            usable=False,
            status="installed_but_unusable",
            detail="CuPy is installed but no compatible CUDA device was detected.",
            installed_version="13.0",
        ),
    )

    decision, backend = build_processing_backend(True)

    assert decision.active is False
    assert backend.name == "cpu"
    assert backend.active is False


def test_detect_acceleration_capability_requires_a_real_test_kernel(
    monkeypatch,
) -> None:
    class _FakeRuntime:
        @staticmethod
        def getDeviceCount() -> int:
            return 1

        @staticmethod
        def getDeviceProperties(index: int) -> dict[str, bytes]:
            assert index == 0
            return {"name": b"Example GPU"}

    class _FakeNullStream:
        @staticmethod
        def synchronize() -> None:
            return None

    class _FakeStream:
        null = _FakeNullStream()

    class _FakeCuda:
        runtime = _FakeRuntime()
        Stream = _FakeStream

    class _FakeSumResult:
        @staticmethod
        def item() -> float:
            return 0.0

    class _FakeArray:
        @staticmethod
        def sum() -> _FakeSumResult:
            return _FakeSumResult()

    class _FakeCupy:
        __version__ = "13.6.0"
        cuda = _FakeCuda()
        float32 = object()

        @staticmethod
        def arange(*args, **kwargs) -> _FakeArray:
            raise RuntimeError("nvrtc missing")

    acceleration_module.clear_acceleration_capability_cache()
    monkeypatch.setattr(acceleration_module, "_import_cupy", lambda: _FakeCupy())

    try:
        capability = acceleration_module.detect_acceleration_capability()
    finally:
        acceleration_module.clear_acceleration_capability_cache()

    assert capability.importable is True
    assert capability.usable is False
    assert capability.status == "installed_but_unusable"
    assert "test kernel" in capability.detail


def test_prepare_cupy_runtime_uses_packaged_cuda_runtime_and_nvrtc_paths(
    tmp_path,
    monkeypatch,
) -> None:
    runtime_root = tmp_path / "nvidia" / "cuda_runtime"
    nvrtc_bin = tmp_path / "nvidia" / "cuda_nvrtc" / "bin"
    (runtime_root / "bin").mkdir(parents=True)
    (runtime_root / "include").mkdir()
    nvrtc_bin.mkdir(parents=True)

    added_directories: list[str] = []

    acceleration_module._REGISTERED_CUDA_DLL_DIRECTORIES.clear()
    acceleration_module._CUDA_DLL_DIRECTORY_HANDLES.clear()
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(acceleration_module.os, "name", "nt")
    monkeypatch.setattr(
        acceleration_module.os,
        "add_dll_directory",
        lambda path: added_directories.append(path) or path,
    )
    monkeypatch.setattr(
        acceleration_module.site,
        "getsitepackages",
        lambda: [str(tmp_path)],
    )
    monkeypatch.setattr(
        acceleration_module.site,
        "getusersitepackages",
        lambda: "",
    )

    acceleration_module._prepare_cupy_runtime()

    assert acceleration_module.os.environ["CUDA_PATH"] == str(runtime_root)
    assert str(runtime_root / "bin") in added_directories
    assert str(nvrtc_bin) in added_directories


def test_transfer_buffer_writable_view_is_capped_to_requested_size() -> None:
    buffer = acceleration_module.TransferBuffer(
        buffer=bytearray(range(16)),
        size_bytes=8,
    )

    assert len(buffer.writable_view()) == 8
    assert bytes(buffer.slice(6)) == bytes(range(6))
    assert buffer.numpy_uint8_view(5).tolist() == [0, 1, 2, 3, 4]


def test_cupy_backend_synchronizes_after_pinned_transfer_copy(monkeypatch) -> None:
    synchronized = {"count": 0}

    class _FakeNullStream:
        @staticmethod
        def synchronize() -> None:
            synchronized["count"] += 1

    class _FakeStream:
        null = _FakeNullStream()

    class _FakeCuda:
        Stream = _FakeStream

        @staticmethod
        def alloc_pinned_memory(size_bytes: int) -> bytearray:
            return bytearray(size_bytes)

    class _FakeArray:
        def __init__(self, payload: np.ndarray) -> None:
            self._payload = payload
            self.shape = payload.shape
            self.size = payload.size

        def __mul__(self, scalar) -> "_FakeArray":
            return _FakeArray(self._payload * scalar)

        def astype(self, dtype, copy: bool = False) -> "_FakeArray":
            return _FakeArray(self._payload.astype(dtype, copy=copy))

        def get(self, *, out: np.ndarray) -> None:
            out[...] = self._payload

    class _FakeCupy:
        cuda = _FakeCuda()
        uint8 = np.uint8
        float32 = np.float32
        ndarray = _FakeArray

        @staticmethod
        def clip(frames, _low, _high):
            return frames

        @staticmethod
        def rint(frames):
            return _FakeArray(np.rint(frames._payload))

    backend = acceleration_module.CupyProcessingBackend.__new__(
        acceleration_module.CupyProcessingBackend
    )
    backend._cp = _FakeCupy()
    backend.xp = backend._cp
    backend.decision = None
    backend.device_name = "Example GPU"
    frames = _FakeArray(
        (np.arange(12, dtype=np.float32) / np.float32(255.0)).reshape(1, 2, 2, 3)
    )
    transfer_buffer = acceleration_module.TransferBuffer(buffer=bytearray(12), size_bytes=12)

    payload = backend.float_frames_to_rgb24_bytes(
        frames,
        transfer_buffer=transfer_buffer,
    )

    assert len(payload) == 12
    assert bytes(payload) == bytes(range(12))
    assert synchronized["count"] == 1


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
