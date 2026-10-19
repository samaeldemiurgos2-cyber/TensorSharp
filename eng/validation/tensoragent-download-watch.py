#!/usr/bin/env python3
"""Observe a real app download without starting or replacing its download path.

Use a fresh TENSORAGENT_VALIDATION_ROOT. An optional reference directory saves
disk space by hard-linking each *completed, independently hash-matched* download
after recording it. Partial files are never changed and references are read-only.
The private loopback cookie is used locally and excluded from the report.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen
from urllib.parse import urlsplit


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connection', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, action='append', default=[])
    parser.add_argument('--timeout', type=float, default=10800)
    args = parser.parse_args()
    connection = json.loads(args.connection.read_text())
    base = connection['baseUrl'].rstrip('/')
    if urlsplit(base).hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('An isolated loopback app is required')
    root = args.connection.parent.resolve()
    cache = Path(connection['cacheRoot']).resolve()
    if root not in cache.parents or args.output.exists():
        raise ValueError('Use a validation cache under the connection root and a fresh report')
    if Path(args.model).name != args.model:
        raise ValueError('Expected a catalog id')
    model = cache / 'models' / args.model
    if model.exists() and any(model.iterdir()):
        raise ValueError('Start the observer before downloading into an empty model directory')
    references = {}
    for directory in args.reference:
        for file in directory.rglob('*'):
            if file.is_file():
                references.setdefault((file.name, file.stat().st_size), file)
    report = {'status': 'observing', 'model': args.model, 'processId': connection['processId'],
              'startedAtUnix': time.time(), 'files': [], 'states': [], 'maxPartBytes': {},
              'scope': 'Native app-owned download; observer never starts transfers or supplies cached bytes.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    def get(route):
        with urlopen(Request(base + route, headers={'Cookie': connection['cookie']}), timeout=30) as response:
            return json.load(response)
    seen, previous = set(), None
    save()
    try:
        while time.time() - report['startedAtUnix'] < args.timeout:
            state = get('/api/agent/catalog/' + args.model)
            download = state.get('download')
            signature = json.dumps(download, sort_keys=True)
            if signature != previous:
                report['states'].append({'atUnix': time.time(), 'download': download,
                                         'state': state['state']})
                previous = signature
            for file in model.rglob('*') if model.exists() else []:
                if not file.is_file() or file.is_symlink():
                    continue
                relative = str(file.relative_to(model))
                if file.name.endswith('.part'):
                    report['maxPartBytes'][relative] = max(file.stat().st_size, report['maxPartBytes'].get(relative, 0))
                    continue
                if relative in seen:
                    continue
                row = {'path': relative, 'bytes': file.stat().st_size, 'sha256': sha(file),
                       'completedAtUnix': time.time(), 'deduplicated': False}
                reference = references.get((file.name, row['bytes']))
                if reference is not None and not os.path.samefile(file, reference) and sha(reference) == row['sha256']:
                    temporary = file.with_name(file.name + '.validated-link')
                    os.link(reference, temporary)
                    os.replace(temporary, file)
                    row['deduplicated'] = True
                report['files'].append(row)
                seen.add(relative)
                print('Verified', relative, row['bytes'], flush=True)
            save()
            if state['state'] == 'Installed' and download and not download.get('running', False):
                report.update(status='passed', elapsedSeconds=time.time() - report['startedAtUnix'], final=state)
                save()
                return
            if download and download.get('error'):
                raise RuntimeError(download['error'])
            time.sleep(1)
        raise TimeoutError('Download observation timed out')
    except Exception as error:
        report.update(status='failed', error=str(error), elapsedSeconds=time.time() - report['startedAtUnix'])
        save()
        raise


if __name__ == '__main__':
    main()
