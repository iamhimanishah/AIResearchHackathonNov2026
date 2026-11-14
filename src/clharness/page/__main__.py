"""`python -m clharness.page --dir results/agent`: serve the page, re-rendered on every load."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .render import render


def make_server(out_dir: Path, rows: Path | None, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.split("?")[0] not in ("/", "/index.html"):
                self.send_error(404)
                return
            body = render(out_dir, rows).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    return ThreadingHTTPServer((host, port), Handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m clharness.page", description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("results/agent"))
    parser.add_argument("--rows", type=Path, default=None, help="default: <dir>/rows.jsonl")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--write", type=Path, default=None, help="write one HTML file and exit")
    args = parser.parse_args(argv)
    if args.write:
        args.write.write_text(render(args.dir, args.rows, refresh_s=None), encoding="utf-8")
        return 0
    server = make_server(args.dir, args.rows, args.host, args.port)
    print(f"serving {args.dir} on http://{args.host}:{server.server_port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
