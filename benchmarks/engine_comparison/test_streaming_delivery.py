"""Exercise timing-sensitive stream delivery over a real loopback socket."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest

import engines


class StreamingDeliveryTests(unittest.TestCase):
    def test_flushed_short_frame_reaches_client_before_server_continues(self):
        received = threading.Event()
        delivered_before_finish = []

        class Lines(list):
            def append(self, line):
                super().append(line)
                if '"content": "first"' in line:
                    received.set()

        class Log(dict):
            def __setitem__(self, key, value):
                super().__setitem__(key, Lines() if key == 'sse_lines' else value)

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get('Content-Length', 0)))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Connection', 'close')
                self.end_headers()
                first = {'choices': [{'delta': {'content': 'first'}}]}
                frame = ('data: ' + json.dumps(first) + '\n\n').encode()
                assert len(frame) < 512
                self.wfile.write(frame)
                self.wfile.flush()
                # No fragile elapsed-time threshold: the server waits for the
                # actual client parser to consume this frame before finishing.
                delivered_before_finish.append(received.wait(3))
                last = {'choices': [{'delta': {}, 'finish_reason': 'stop'}],
                        'usage': {'prompt_tokens': 2, 'completion_tokens': 1}}
                self.wfile.write(('data: ' + json.dumps(last) + '\n\ndata: [DONE]\n\n').encode())
                self.wfile.flush()
                self.close_connection = True

        with ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                result = engines._run_streaming(
                    f'http://127.0.0.1:{server.server_port}/v1/chat/completions', {}, 5, Log())
            finally:
                server.shutdown()
                thread.join(timeout=5)
        self.assertEqual([True], delivered_before_finish)
        self.assertEqual('first', result['output_text'])
        self.assertEqual('stop', result['finish_reason'])
        self.assertTrue(result['usage_present'])


if __name__ == '__main__':
    unittest.main()
