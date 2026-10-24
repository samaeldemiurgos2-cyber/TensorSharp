#!/usr/bin/env python3
"""Record tracked source hashes, or verify a remote source copy without .git.

Build products and other untracked files are excluded. A source-clean claim is
only made when the original Git tree was clean and every tracked file matches.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['record', 'verify'])
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--git-archive', action='store_true', help='Hash committed archive bytes, avoiding Windows CRLF checkout conversion')
    parser.add_argument('--applied-working-tree', action='store_true', help='Committed archive plus exact modified/untracked working files')
    parser.add_argument('--allow-dirty', action='store_true', help='Verify content equality without claiming that the source is pristine')
    args = parser.parse_args()
    if args.mode == 'record':
        def git(*command):
            return subprocess.check_output(['git', '-C', str(args.directory), *command])
        files = {}
        if args.git_archive or args.applied_working_tree:
            process = subprocess.Popen(['git', '-C', str(args.directory), 'archive', '--format=tar', 'HEAD'], stdout=subprocess.PIPE)
            with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
                for member in archive:
                    if member.isdir():
                        continue
                    if not member.isfile():
                        parser.error('Archive contains a non-file entry: ' + member.name)
                    result = hashlib.sha256()
                    with archive.extractfile(member) as source:
                        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                            result.update(block)
                    files[member.name] = result.hexdigest()
            if process.wait() != 0:
                parser.error('git archive failed')
            if args.applied_working_tree:
                modified = git('diff', '--name-only', '-z', 'HEAD').decode().rstrip('\0').split('\0')
                untracked = git('ls-files', '--others', '--exclude-standard', '-z').decode().rstrip('\0').split('\0')
                for relative in set(modified + untracked) - {''}:
                    path = args.directory / relative
                    if path.is_file():
                        files[relative] = digest(path)
                    else:
                        files.pop(relative, None)
        else:
            paths = git('ls-files', '-z').decode().rstrip('\0').split('\0')
            for relative in paths:
                path = args.directory / relative
                if not path.is_file():
                    parser.error('Tracked entry is missing or is a submodule: ' + relative)
                files[relative] = digest(path)
        data = {'revision': git('rev-parse', 'HEAD').decode().strip(),
                'original_clean': git('status', '--porcelain').strip() == b'',
                'hash_basis': ('archive_plus_working_changes' if args.applied_working_tree else
                               'git_archive' if args.git_archive else 'working_tree'), 'files': files}
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(data, indent=2) + '\n')
        print(json.dumps({k: v for k, v in data.items() if k != 'files'}), len(files), 'files')
        return 0
    if args.output is None:
        parser.error('--output is required in verify mode')
    manifest = json.loads(args.manifest.read_text())
    failures = []
    for relative, expected in manifest['files'].items():
        path = (args.directory / relative).resolve()
        if not path.is_relative_to(args.directory.resolve()):
            parser.error('Manifest path escapes source directory')
        actual = digest(path) if path.is_file() else None
        if actual != expected:
            failures.append({'file': relative, 'expected': expected, 'actual': actual})
    data = {'revision': manifest['revision'], 'original_clean': manifest['original_clean'],
            'manifest_sha256': digest(args.manifest), 'checked_files': len(manifest['files']),
            'failures': failures, 'status': 'pass' if (manifest['original_clean'] or args.allow_dirty) and not failures else 'fail'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(data))
    return int(data['status'] != 'pass')


if __name__ == '__main__':
    raise SystemExit(main())
