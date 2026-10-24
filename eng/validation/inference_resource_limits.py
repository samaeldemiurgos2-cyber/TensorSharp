"""Read effective Linux cgroup limits separately from physical host capacity."""
from pathlib import Path


def cgroup_snapshot(root=Path('/sys/fs/cgroup')):
    root = Path(root)

    def read(path):
        try:
            return path.read_text().strip()
        except OSError:
            return None

    def integer(path):
        value = read(path)
        return int(value) if value is not None and value not in ('max', '') else None

    if (root / 'memory.max').exists():
        version = 2
        memory = root
        maximum, current, peak = 'memory.max', 'memory.current', 'memory.peak'
        cpu_max = (read(root / 'cpu.max') or '').split()
        quota = int(cpu_max[0]) if cpu_max and cpu_max[0] != 'max' else None
        period = int(cpu_max[1]) if len(cpu_max) > 1 else None
        cpuset = read(root / 'cpuset.cpus.effective') or read(root / 'cpuset.cpus')
        events = read(root / 'memory.events')
    elif (root / 'memory/memory.limit_in_bytes').exists():
        version = 1
        memory = root / 'memory'
        maximum, current, peak = 'memory.limit_in_bytes', 'memory.usage_in_bytes', 'memory.max_usage_in_bytes'
        cpu = next((p for p in (root / 'cpu', root / 'cpu,cpuacct')
                    if (p / 'cpu.cfs_quota_us').exists()), root / 'cpu')
        quota, period = integer(cpu / 'cpu.cfs_quota_us'), integer(cpu / 'cpu.cfs_period_us')
        cpuset = read(root / 'cpuset/cpuset.cpus')
        events = read(memory / 'memory.failcnt')
    else:
        return {'version': None, 'status': 'unavailable', 'root': str(root)}
    limit = integer(memory / maximum)
    if limit is not None and limit >= 2**60:
        limit = None  # v1's page-aligned LONG_MAX means unlimited.
    if quota is not None and quota < 0:
        quota = None
    return {'version': version, 'status': 'observed', 'root': str(root),
            'memory_limit_bytes': limit, 'memory_current_bytes': integer(memory / current),
            'memory_peak_bytes': integer(memory / peak), 'memory_events': events,
            'cpu_quota_us': quota, 'cpu_period_us': period,
            'cpu_quota_cores': quota / period if quota is not None and period else None,
            'cpuset_cpus': cpuset}


def stable_limits(snapshot):
    return {key: snapshot.get(key) for key in ('version', 'status', 'memory_limit_bytes',
            'cpu_quota_us', 'cpu_period_us', 'cpu_quota_cores', 'cpuset_cpus')}
