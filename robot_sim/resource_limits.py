"""Execute a process tree on one CPU without library affinity overrides."""
import ctypes
import errno
import os
import sys


def main():
    if len(sys.argv) < 2:
        raise SystemExit("A command is required")
    cpu = int(os.environ.get("ROBOT_SIM_CPU", min(os.sched_getaffinity(0))))
    os.sched_setaffinity(0, {cpu})
    library = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    library.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    library.seccomp_load.argtypes = [ctypes.c_void_p]
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    context = library.seccomp_init(0x7fff0000)
    if not context:
        raise RuntimeError("Cannot initialize the CPU affinity lock")
    try:
        syscall = library.seccomp_syscall_resolve_name(b"sched_setaffinity")
        if syscall < 0:
            raise RuntimeError("Cannot resolve sched_setaffinity")
        result = library.seccomp_rule_add(context, 0x50000 | errno.EPERM, syscall, 0)
        if result != 0 or library.seccomp_load(context) != 0:
            raise RuntimeError("Cannot enforce the CPU affinity lock")
    finally:
        library.seccomp_release(context)
    os.environ["ROBOT_CPU_LOCKED"] = "1"
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
