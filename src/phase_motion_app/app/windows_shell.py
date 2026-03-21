"""This file centralizes Windows shell identity so the PyQt window icon, taskbar button, and relaunch metadata stay aligned without leaking platform-specific calls into the core domain layer."""

from __future__ import annotations

import ctypes
import sys
import uuid
from ctypes import wintypes
from pathlib import Path
from typing import Sequence

from PyQt6.QtGui import QIcon

WINDOWS_APP_USER_MODEL_ID = "uk.co.jshade.phase_motion_app"
WINDOWS_RELAUNCH_DISPLAY_NAME = "Phase-based Motion Amplification Utility"

_VT_EMPTY = 0
_VT_LPWSTR = 31
_APP_USER_MODEL_FORMAT_ID = "9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3"
_IID_IPROPERTYSTORE = "886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99"


def _asset_root() -> Path:
    return Path(__file__).resolve().parents[3] / "assets"


def load_shell_icon() -> QIcon | None:
    """Load the repo-local shell icon when the checkout asset is available."""

    icon_path = _asset_root() / "programIcon.ico"
    if not icon_path.exists():
        return None
    icon = QIcon(str(icon_path))
    if icon.isNull():
        return None
    return icon


def configure_windows_process_identity() -> None:
    """Give the GUI process a stable Windows taskbar identity instead of inheriting Python's."""

    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(  # type: ignore[attr-defined]
            WINDOWS_APP_USER_MODEL_ID
        )
    except (AttributeError, OSError):
        return


def apply_windows_window_identity(window: object) -> bool:
    """Attach relaunch metadata to the main window so the Windows taskbar uses the app icon."""

    if sys.platform != "win32":
        return False
    hwnd = int(window.winId())
    return _set_window_property_values(hwnd, _window_property_values())


def clear_windows_window_identity(window: object) -> bool:
    """Remove relaunch metadata before the window closes as required by the Windows shell API."""

    if sys.platform != "win32":
        return False
    hwnd = int(window.winId())
    return _set_window_property_values(hwnd, _window_property_clear_values())


def _window_property_values() -> tuple[tuple[int, str | None], ...]:
    command_line = _current_process_command_line()
    values: list[tuple[int, str | None]] = []
    if command_line:
        values.append((2, command_line))
        values.append((4, WINDOWS_RELAUNCH_DISPLAY_NAME))
    if (icon_resource := _relaunch_icon_resource()) is not None:
        values.append((3, icon_resource))
    values.append((5, WINDOWS_APP_USER_MODEL_ID))
    return tuple(values)


def _window_property_clear_values() -> tuple[tuple[int, str | None], ...]:
    return (
        (2, None),
        (4, None),
        (3, None),
        (5, None),
    )


def _relaunch_icon_resource() -> str | None:
    icon_path = (_asset_root() / "programIcon.ico").resolve()
    if not icon_path.exists():
        return None
    return f"{icon_path},0"


def _current_process_command_line() -> str | None:
    if sys.platform != "win32":
        return None
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.GetCommandLineW.restype = wintypes.LPWSTR
        command_line = kernel32.GetCommandLineW()
    except AttributeError:
        return None
    return command_line or None


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    @classmethod
    def from_string(cls, value: str) -> "_GUID":
        return cls.from_buffer_copy(uuid.UUID(value).bytes_le)


class _PROPERTYKEY(ctypes.Structure):
    _fields_ = [
        ("fmtid", _GUID),
        ("pid", wintypes.DWORD),
    ]


class _PROPVARIANT_VALUE(ctypes.Union):
    _fields_ = [("pwszVal", wintypes.LPWSTR)]


class _PROPVARIANT(ctypes.Structure):
    _anonymous_ = ("value",)
    _fields_ = [
        ("vt", wintypes.USHORT),
        ("wReserved1", wintypes.USHORT),
        ("wReserved2", wintypes.USHORT),
        ("wReserved3", wintypes.USHORT),
        ("value", _PROPVARIANT_VALUE),
    ]


_HRESULT = ctypes.c_long


class _IPropertyStore(ctypes.Structure):
    pass


_IPropertyStorePointer = ctypes.POINTER(_IPropertyStore)
_QueryInterface = ctypes.WINFUNCTYPE(
    _HRESULT,
    _IPropertyStorePointer,
    ctypes.POINTER(_GUID),
    ctypes.POINTER(ctypes.c_void_p),
)
_AddRef = ctypes.WINFUNCTYPE(wintypes.ULONG, _IPropertyStorePointer)
_Release = ctypes.WINFUNCTYPE(wintypes.ULONG, _IPropertyStorePointer)
_GetCount = ctypes.WINFUNCTYPE(_HRESULT, _IPropertyStorePointer, ctypes.POINTER(wintypes.DWORD))
_GetAt = ctypes.WINFUNCTYPE(
    _HRESULT,
    _IPropertyStorePointer,
    wintypes.DWORD,
    ctypes.POINTER(_PROPERTYKEY),
)
_GetValue = ctypes.WINFUNCTYPE(
    _HRESULT,
    _IPropertyStorePointer,
    ctypes.POINTER(_PROPERTYKEY),
    ctypes.POINTER(_PROPVARIANT),
)
_SetValue = ctypes.WINFUNCTYPE(
    _HRESULT,
    _IPropertyStorePointer,
    ctypes.POINTER(_PROPERTYKEY),
    ctypes.POINTER(_PROPVARIANT),
)
_Commit = ctypes.WINFUNCTYPE(_HRESULT, _IPropertyStorePointer)


class _IPropertyStoreVtbl(ctypes.Structure):
    _fields_ = [
        ("QueryInterface", _QueryInterface),
        ("AddRef", _AddRef),
        ("Release", _Release),
        ("GetCount", _GetCount),
        ("GetAt", _GetAt),
        ("GetValue", _GetValue),
        ("SetValue", _SetValue),
        ("Commit", _Commit),
    ]


_IPropertyStore._fields_ = [("lpVtbl", ctypes.POINTER(_IPropertyStoreVtbl))]


def _set_window_property_values(
    hwnd: int,
    values: Sequence[tuple[int, str | None]],
) -> bool:
    property_store = _window_property_store(hwnd)
    if property_store is None:
        return False
    try:
        for property_id, value in values:
            if not _set_property_value(property_store, property_id, value):
                return False
        return True
    finally:
        property_store.contents.lpVtbl.contents.Release(property_store)


def _window_property_store(hwnd: int) -> _IPropertyStorePointer | None:
    try:
        shell32 = ctypes.windll.shell32  # type: ignore[attr-defined]
    except AttributeError:
        return None
    property_store_ptr = ctypes.c_void_p()
    shell32.SHGetPropertyStoreForWindow.argtypes = [
        wintypes.HWND,
        ctypes.POINTER(_GUID),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    shell32.SHGetPropertyStoreForWindow.restype = _HRESULT
    result = shell32.SHGetPropertyStoreForWindow(
        wintypes.HWND(hwnd),
        ctypes.byref(_GUID.from_string(_IID_IPROPERTYSTORE)),
        ctypes.byref(property_store_ptr),
    )
    if result != 0 or not property_store_ptr.value:
        return None
    return ctypes.cast(property_store_ptr, _IPropertyStorePointer)


def _set_property_value(
    property_store: _IPropertyStorePointer,
    property_id: int,
    value: str | None,
) -> bool:
    key = _PROPERTYKEY(_GUID.from_string(_APP_USER_MODEL_FORMAT_ID), property_id)
    if value is None:
        variant = _empty_property_variant()
        string_buffer = None
    else:
        variant, string_buffer = _string_property_variant(value)
    result = property_store.contents.lpVtbl.contents.SetValue(
        property_store,
        ctypes.byref(key),
        ctypes.byref(variant),
    )
    _ = string_buffer
    return result == 0


def _empty_property_variant() -> _PROPVARIANT:
    variant = _PROPVARIANT()
    variant.vt = _VT_EMPTY
    return variant


def _string_property_variant(value: str) -> tuple[_PROPVARIANT, ctypes.Array[ctypes.c_wchar]]:
    buffer = ctypes.create_unicode_buffer(value)
    variant = _PROPVARIANT()
    variant.vt = _VT_LPWSTR
    variant.pwszVal = ctypes.cast(buffer, wintypes.LPWSTR)
    return variant, buffer


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
