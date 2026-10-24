#!/usr/bin/env python3
"""Low-frequency Linux process/GPU observations for inference validation.

Samples contain total device usage, not memory uniquely attributable to one
server. CPU percent uses 100% per fully occupied core. Five-second sampling can
miss short peaks; summarized maxima are observed maxima, never exact peaks.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from inference_resource_limits import cgroup_snapshot


def parse_process_stat(stat, ticks, page_size):
    # comm may itself contain spaces/parentheses. Fields after the last ')' start
    # at proc field3 (state); major faults are field12, not child faults field13.
    fields = stat[stat.rfind(')') + 2:].split()
    return {'start_ticks': int(fields[19]),
            'cpu_seconds': (int(fields[11]) + int(fields[12])) / ticks,
            'rss_mib': int(fields[21]) * page_size / 1024**2,
            'threads': int(fields[17]), 'minor_faults': int(fields[7]),
            'major_faults': int(fields[9])}


def gpu_snapshot():
    fields = ['index', 'uuid', 'memory.used', 'memory.total', 'utilization.gpu',
              'utilization.memory', 'power.draw', 'temperature.gpu']
    process = subprocess.run(['nvidia-smi', '--query-gpu=' + ','.join(fields),
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=4)
    if process.returncode:
        return [], process.stderr.strip()
    devices = []
    for line in process.stdout.splitlines():
        values = [v.strip() for v in line.split(',')]
        if len(values) != len(fields):
            return [], 'Unexpected nvidia-smi CSV row: ' + line
        row = {'index': int(values[0]), 'uuid': values[1]}
        for name, value in zip(['memory_used_mib', 'memory_total_mib', 'gpu_util_percent',
                                'memory_util_percent', 'power_watts', 'temperature_celsius'], values[2:]):
            try:
                row[name] = float(value)
            except ValueError:
                row[name] = None
        devices.append(row)
    return devices, None


def process_snapshots(previous, now):
    results, current = [], {}
    ticks = os.sysconf('SC_CLK_TCK')
    page_size = os.sysconf('SC_PAGE_SIZE')
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            command = (path / 'cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
            if not command:
                continue
            executable = Path(command[0]).name
            if executable == 'llama-server':
                engine = 'llamacpp'
            elif executable == 'dotnet' and any(Path(p).name == 'TensorSharp.Server.Host.dll' for p in command[1:]):
                engine = 'tensorsharp'
            else:
                continue
            values = parse_process_stat((path / 'stat').read_text(), ticks, page_size)
            try:
                io = {name: int(value) for name, value in
                      (line.split(':', 1) for line in (path / 'io').read_text().splitlines())}
            except OSError:
                io = {}
            cpu_seconds = values['cpu_seconds']
            key = (int(path.name), values['start_ticks'])
            read_bytes = io.get('read_bytes')
            current[key] = (now, cpu_seconds, values['major_faults'], read_bytes)
            prior = previous.get(key)
            elapsed = now - prior[0] if prior and now > prior[0] else None
            percent = (cpu_seconds - prior[1]) / elapsed * 100 if elapsed else None
            major_delta = values['major_faults'] - prior[2] if elapsed else None
            read_delta = read_bytes - prior[3] if elapsed and read_bytes is not None and prior[3] is not None else None
            results.append({'pid': key[0], 'engine': engine, 'command': command, **values,
                            'cpu_percent': percent, 'read_bytes': read_bytes, 'read_characters': io.get('rchar'),
                            'major_faults_delta': major_delta, 'read_bytes_delta': read_delta,
                            'major_faults_per_second': major_delta / elapsed if elapsed else None,
                            'read_bytes_per_second': read_delta / elapsed if read_delta is not None else None})
        except (OSError, ValueError, IndexError):
            continue  # Process exited while being read.
    return results, current


def sample(args):
    if not Path('/proc').is_dir():
        raise RuntimeError('Sampling requires Linux /proc')
    import fcntl
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.set())
    started = time.monotonic()
    previous = {}
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('a', encoding='utf-8') as output:
        fcntl.flock(output.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not stopped.is_set():
            now = time.monotonic()
            processes, previous = process_snapshots(previous, now)
            try:
                gpus, error = gpu_snapshot()
            except (OSError, subprocess.TimeoutExpired) as failure:
                gpus, error = [], str(failure)
            row = {'utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': now - started,
                   'sampler_pid': os.getpid(), 'interval_seconds': args.interval,
                   'sampler_sha256': script_hash,
                   'processes': processes, 'gpus': gpus, 'gpu_query_error': error,
                   'cgroup': cgroup_snapshot()}
            output.write(json.dumps(row, allow_nan=False) + '\n')
            output.flush()
            stopped.wait(max(0, args.interval - (time.monotonic() - now)))


def summarize(args):
    gpu_max, servers, samples, errors, cgroup = {}, {}, 0, 0, {}
    for line in args.input.read_text().splitlines():
        row = json.loads(line)
        samples += 1
        errors += bool(row['gpu_query_error'])
        if row.get('cgroup', {}).get('status') == 'observed':
            cgroup['latest'] = row['cgroup']
            used = row['cgroup'].get('memory_current_bytes')
            if used is not None:
                cgroup['memory_current_bytes_observed_max'] = max(cgroup.get('memory_current_bytes_observed_max', 0), used)
        for device in row['gpus']:
            entry = gpu_max.setdefault(device['uuid'], {'index': device['index']})
            for key in ['memory_used_mib', 'gpu_util_percent', 'power_watts', 'temperature_celsius']:
                if device[key] is not None:
                    entry[key + '_observed_max'] = max(entry.get(key + '_observed_max', 0), device[key])
        for process in row['processes']:
            key = f"{process['pid']}:{process['start_ticks']}"
            entry = servers.setdefault(key, {'pid': process['pid'], 'engine': process['engine'],
                                             'command': process['command'], 'first_utc': row['utc']})
            entry['last_utc'] = row['utc']
            for metric in ['rss_mib', 'cpu_percent', 'threads', 'major_faults', 'read_bytes',
                           'major_faults_per_second', 'read_bytes_per_second']:
                if process.get(metric) is not None:
                    entry[metric + '_observed_max'] = max(entry.get(metric + '_observed_max', 0), process[metric])
    result = {'samples': samples, 'gpu_query_errors': errors, 'devices': gpu_max, 'servers': servers, 'cgroup': cgroup,
              'scope': 'Observed maxima at sampling cadence. Device memory includes all users. CPU percent is per core; short peaks may be missed.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'samples': samples, 'gpu_query_errors': errors, 'server_count': len(servers)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode', required=True)
    sampling = commands.add_parser('sample')
    sampling.add_argument('--output', type=Path, required=True)
    sampling.add_argument('--interval', type=float, default=5)
    summary = commands.add_parser('summarize')
    summary.add_argument('--input', type=Path, required=True)
    summary.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'sample' and args.interval < 5:
        parser.error('Use an interval of at least 5 seconds to limit measurement interference')
    (sample if args.mode == 'sample' else summarize)(args)


if __name__ == '__main__':
    main()
