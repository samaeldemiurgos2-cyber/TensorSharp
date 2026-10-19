#!/usr/bin/env python3
"""Measure real TensorAgent chat media turns on an already activated model.

Pass the private connection.json of an isolated host or native app. Native UI
actions and visual review must be verified separately. No model is substituted,
selected, or downloaded by this probe. Cookies and inline previews aren't saved.
"""
import argparse
import hashlib
import json
import mimetypes
from pathlib import Path
import struct
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connection', type=Path, required=True)
    parser.add_argument('--family', choices=['qwen-image', 'ltx-2.5'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1:
        raise ValueError('Use a fresh output folder and positive repeats')
    connection = json.loads(args.connection.read_text())
    base = connection['baseUrl'].rstrip('/')
    if urlsplit(base).hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('An isolated loopback host is required')
    args.output.mkdir(parents=True)
    def open_route(route, data=None, content_type='application/json'):
        return urlopen(Request(base + route, data=data, headers={
            'Cookie': connection['cookie'], 'Content-Type': content_type}), timeout=1800)
    def read_json(route, body=None):
        with open_route(route, None if body is None else json.dumps(body).encode()) as response:
            return json.load(response)
    report = {'status': 'running', 'processId': connection['processId'], 'family': args.family,
              'scope': 'Real activated host/app HTTP chat, upload and conversation paths; UI controls covered separately.',
              'limitations': ['Serial samples with the model default seed policy; not a statistical quality or latency study.',
                              'Generated files need visual review and full media decoding.'], 'cases': []}
    def save():
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    try:
        report['model'] = read_json('/api/models')
        assert args.family in report['model'].get('architecture', '').replace('_', '-'), report['model']
        assert report['model'].get('loadedBackend') == 'ggml_metal', 'Expected native Metal selection'
        uploads = []
        if args.reference:
            boundary = 'native-validation-' + uuid.uuid4().hex
            name = args.reference.name.replace('"', '_')
            data = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n'
                    f'Content-Type: {mimetypes.guess_type(name)[0] or "image/png"}\r\n\r\n').encode()
            data += args.reference.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
            with open_route('/api/upload', data, 'multipart/form-data; boundary=' + boundary) as response:
                upload = json.load(response)
            assert upload.get('ok') and upload.get('mediaType') == 'image', upload
            uploads.append(upload)
        for index in range(args.repeats):
            session = read_json('/api/sessions?conversation=new', {})
            message = {'role': 'user', 'content': args.prompt,
                       'stillImagePaths': [u['file'] for u in uploads], 'attachments': uploads}
            body = {'sessionId': session['sessionId'], 'messages': [message]}
            row = {'index': index, 'request': body, 'conversationId': session['conversationId'],
                   'events': [], 'status': 'running'}
            report['cases'].append(row)
            save()
            started, steps = time.monotonic(), []
            with open_route('/api/chat', json.dumps(body).encode()) as response, (args.output / f'{index}.jsonl').open('w') as log:
                assert response.headers.get_content_type() == 'text/event-stream'
                for line in response:
                    if not line.startswith(b'data: '):
                        continue
                    event = json.loads(line[6:])
                    filtered = {k: v for k, v in event.items() if k not in ('preview', 'image')}
                    filtered['receivedSeconds'] = time.monotonic() - started
                    log.write(json.dumps(filtered) + '\n')
                    log.flush()
                    row['events'].append(filtered)
                    if event.get('error'):
                        raise RuntimeError(event['error'])
                    if 'image_step' in event:
                        steps.append(filtered)
            row['seconds'] = time.monotonic() - started
            assert any(e.get('done') and not e.get('aborted') for e in row['events']), 'Turn did not complete'
            media_key, suffix = ('imageUrl', '.png') if args.family == 'qwen-image' else ('videoUrl', '.mp4')
            final = next(e for e in row['events'] if e.get(media_key))
            row['terminal'] = final
            if args.family == 'qwen-image':
                assert steps and steps[-1]['image_step'] == 8 and all(s['image_steps'] == 8 for s in steps), 'Wrong Turbo schedule'
                assert final['imageSources'] == [u['file'] for u in uploads], 'References changed'
                row['steadyStepSeconds'] = (steps[-1]['receivedSeconds'] - steps[0]['receivedSeconds']) / 7
            else:
                assert (final['width'], final['height'], final['frames'], final['fps'], final['hasAudio']) == (1024, 1024, 17, 24, True), 'Wrong LTX recipe'
            with open_route(final[media_key]) as response:
                data = response.read()
            target = args.output / (str(index) + suffix)
            target.write_bytes(data)
            row['media'] = {'path': str(target.resolve()), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            if suffix == '.png':
                assert data[:8] == b'\x89PNG\r\n\x1a\n' and struct.unpack('>II', data[16:24]) == (final['width'], final['height'])
            else:
                with open_route(final[media_key][:-4] + '.wav') as response:
                    wav = response.read()
                (args.output / (str(index) + '.wav')).write_bytes(wav)
                assert wav[:4] == b'RIFF' and struct.unpack_from('<I', wav, 24)[0] == 48000 and struct.unpack_from('<H', wav, 22)[0] == 2
            saved = read_json('/api/agent/conversations/' + session['conversationId'])
            assert any(m.get(media_key) == final[media_key] for m in saved['messages']), 'Saved conversation lost media'
            (args.output / f'{index}-conversation.json').write_text(json.dumps(saved, indent=2) + '\n')
            row['status'] = 'passed'
            save()
            print(json.dumps({k: row[k] for k in ('index', 'status', 'seconds', 'media')}), flush=True)
        report['status'] = 'passed'
        save()
    except Exception as error:
        report.update(status='failed', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
