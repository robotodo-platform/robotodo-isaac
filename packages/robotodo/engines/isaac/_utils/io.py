# SPDX-License-Identifier: Apache-2.0

"""
IO utilities.
"""


import os
import ctypes
import contextlib


# TODO ref https://github.com/minrk/wurlitzer/blob/main/wurlitzer.py
def _get_libc_streams_cffi():
    """Use CFFI to lookup stdout/stderr pointers

    Should work ~everywhere, but requires compilation
    """
    try:
        import cffi
    except ImportError:
        raise ImportError(
            "Failed to lookup stdout symbols in libc. Fallback requires cffi."
        )

    _ffi = cffi.FFI()
    _ffi.cdef("const size_t c_stdout_p();")
    _ffi.cdef("const size_t c_stderr_p();")
    _lib = _ffi.verify(
        '\n'.join(
            [
                "#include <stdio.h>",
                "const size_t c_stdout_p() { return (size_t) (void*) stdout; }",
                "const size_t c_stderr_p() { return (size_t) (void*) stderr; }",
            ]
        )
    )
    c_stdout_p = ctypes.c_void_p(_lib.c_stdout_p())
    c_stderr_p = ctypes.c_void_p(_lib.c_stderr_p())

    return c_stdout_p, c_stderr_p


def _get_libc_streams():
    libc = ctypes.CDLL(None)
    c_stdout_p = c_stderr_p = None
    try:
        c_stdout_p = ctypes.c_void_p.in_dll(libc, 'stdout')
        c_stderr_p = ctypes.c_void_p.in_dll(libc, 'stderr')
    except ValueError:
        # libc.stdout has a funny name on macOS
        try:
            c_stdout_p = ctypes.c_void_p.in_dll(libc, '__stdoutp')
            c_stderr_p = ctypes.c_void_p.in_dll(libc, '__stderrp')
        except ValueError:
            c_stdout_p, c_stderr_p = _get_libc_streams_cffi()
    return c_stdout_p, c_stderr_p


@contextlib.contextmanager
def redirect_stdout_libc(new_target: str | None):
    if new_target is None:
        new_target = os.devnull
    
    libc = ctypes.CDLL(None)

    stdout_cptr, _ = _get_libc_streams()

    libc.fflush(None)

    old_stream = stdout_cptr.value
    # TODO
    new_stream = libc.fopen(str.encode(new_target), b"w")
    if not new_stream:
        raise OSError(f"fopen on {new_target} failed")

    # Swap
    stdout_cptr.value = new_stream
    try:
        yield
    finally:
        libc.fflush(None)
        stdout_cptr.value = old_stream
        libc.fclose(new_stream)

