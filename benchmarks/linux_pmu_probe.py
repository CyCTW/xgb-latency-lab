"""Read-only Linux PMU readiness probe; measures only the calling thread.

No perf executable, privilege escalation or kernel setting changes are needed.
This is an availability check, not an inference performance measurement.
"""
import ctypes
import json
import os
from pathlib import Path
import platform
import re
import shutil
import struct


def syscall_number():
    machine=platform.machine()
    paths={
        'aarch64':['/usr/include/asm-generic/unistd.h'],
        'x86_64':['/usr/include/x86_64-linux-gnu/asm/unistd_64.h','/usr/include/asm/unistd_64.h'],
    }.get(machine,[])
    for name in paths:
        path=Path(name)
        if path.is_file():
            match=re.search(r'^#define\s+__NR_perf_event_open\s+(\d+)',path.read_text(),re.M)
            if match:return int(match.group(1)),name
    raise RuntimeError('Cannot resolve perf_event_open from this architecture\'s installed syscall headers')


def probe():
    if platform.system()!='Linux':
        raise RuntimeError('Run this probe inside Linux')
    root=Path('/sys/bus/event_source/devices')
    paranoid=Path('/proc/sys/kernel/perf_event_paranoid')
    result=dict(kernel=platform.release(),machine=platform.machine(),uid=os.geteuid(),
                perf=shutil.which('perf'),clang=shutil.which('clang'),python=shutil.which('python3'),
                pmu_devices={p.name:(p/'type').read_text().strip() for p in root.iterdir() if (p/'type').is_file()} if root.exists() else {},
                perf_event_paranoid=paranoid.read_text().strip() if paranoid.is_file() else None)
    try:
        nr,header=syscall_number()
    except RuntimeError as error:
        result['probe_error']=str(error)
        return result
    result['syscall_number'],result['syscall_header']=nr,header
    result['perf_event_open_probe']={}
    libc=ctypes.CDLL(None,use_errno=True);libc.syscall.restype=ctypes.c_long
    for name,event_type,config in [('task_clock',1,1),('cycles',0,0),('instructions',0,1)]:
        attr=ctypes.create_string_buffer(128)
        # perf_event_attr: type, size, config, sample_period, sample_type,
        # read_format, flags. Enabled immediately; exclude kernel and hypervisor.
        struct.pack_into('IIQQQQQ',attr,0,event_type,128,config,0,0,0,(1<<5)|(1<<6))
        fd=libc.syscall(ctypes.c_long(nr),ctypes.byref(attr),ctypes.c_int(0),ctypes.c_int(-1),ctypes.c_int(-1),ctypes.c_ulong(0))
        if fd<0:
            error=ctypes.get_errno()
            event=dict(opened=False,errno=error,error=os.strerror(error))
        else:
            try:
                checksum=sum(range(500000))
                event=dict(opened=True,count=struct.unpack('Q',os.read(fd,8))[0],checksum=checksum)
            finally:
                os.close(fd)
        result['perf_event_open_probe'][name]=event
    return result


if __name__=='__main__':
    print(json.dumps(probe(),indent=2))
