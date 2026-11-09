"""Standard-library HTTP server: JSON API, artifact downloads and the static web UI."""
from __future__ import annotations

import argparse
import hmac
import json
import logging
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .config import Settings, load_settings
from .jobs import JobError, JobService

log = logging.getLogger("photo3d.http")

WEB_ROOT = (Path(__file__).resolve().parent.parent / "web").resolve()
JOB_RE = re.compile(r"^/api/jobs/([0-9a-f]{12})$")
PHOTO_RE = re.compile(r"^/api/jobs/([0-9a-f]{12})/photos$")
START_RE = re.compile(r"^/api/jobs/([0-9a-f]{12})/start$")
FILE_RE = re.compile(r"^/api/jobs/([0-9a-f]{12})/files/([A-Za-z0-9._-]+)$")
JSON_LIMIT = 64 * 1024
MEDIA_TYPES = {".stl": "model/stl", ".3mf": "model/3mf", ".png": "image/png"}
DOWNLOAD_SUFFIXES = {".stl", ".3mf"}


def make_handler(service: JobService, settings: Settings):
    class Handler(BaseHTTPRequestHandler):
        server_version = "Photo3D/0.1"

        def log_message(self, fmt, *args):  # route access logs through logging
            log.info("%s %s", self.address_string(), fmt % args)

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

        # ----- plumbing ------------------------------------------------------

        def _dispatch(self, method: str) -> None:
            path = urlparse(self.path).path
            try:
                if path.startswith("/api/"):
                    if not self._authorised():
                        raise JobError(401, "Missing or wrong access token.")
                    self._api(method, path)
                elif method == "GET":
                    self._static(path)
                else:
                    raise JobError(405, "Method not allowed.")
            except JobError as exc:
                self._send_json(exc.status, {"error": str(exc)})
            except Exception:
                log.exception("Unhandled error on %s %s", method, path)
                self._send_json(500, {"error": "Internal error."})

        def _authorised(self) -> bool:
            if not settings.app_token:
                return True
            supplied = self.headers.get("Authorization", "")
            expected = f"Bearer {settings.app_token}"
            return hmac.compare_digest(supplied.encode(), expected.encode())

        def _read_body(self, limit: int) -> bytes:
            length = int(self.headers.get("Content-Length") or 0)
            if length > limit:
                raise JobError(413, "Upload is too large.")
            return self.rfile.read(length) if length > 0 else b""

        def _json_body(self) -> dict:
            raw = self._read_body(JSON_LIMIT)
            if not raw:
                return {}
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise JobError(400, "Request body must be JSON.") from exc
            if not isinstance(data, dict):
                raise JobError(400, "Request body must be a JSON object.")
            return data

        def _send(self, status: int, body: bytes, content_type: str, headers=()) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            for key, value in headers:
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

        # ----- routes --------------------------------------------------------

        def _api(self, method: str, path: str) -> None:
            if path == "/api/jobs":
                if method == "GET":
                    return self._send_json(200, {"jobs": service.list_jobs()})
                if method == "POST":
                    body = self._json_body()
                    return self._send_json(201, service.create(body.get("reference_width_mm"), body.get("name")))
            if method == "GET" and (m := JOB_RE.match(path)):
                return self._send_json(200, service.get(m.group(1)))
            if method == "POST" and (m := PHOTO_RE.match(path)):
                raw = self._read_body(settings.max_upload_bytes)
                return self._send_json(201, service.add_photo(m.group(1), raw))
            if method == "POST" and (m := START_RE.match(path)):
                return self._send_json(202, service.start(m.group(1)))
            if method == "GET" and (m := FILE_RE.match(path)):
                return self._send_file(service.artifact_path(m.group(1), m.group(2)))
            raise JobError(404, "Not found.")

        def _send_file(self, file_path: Path) -> None:
            suffix = file_path.suffix.lower()
            headers = []
            if suffix in DOWNLOAD_SUFFIXES:
                headers.append(("Content-Disposition", f'attachment; filename="{file_path.name}"'))
            self._send(200, file_path.read_bytes(), MEDIA_TYPES.get(suffix, "application/octet-stream"), headers)

        def _static(self, path: str) -> None:
            rel = unquote(path).lstrip("/") or "index.html"
            target = (WEB_ROOT / rel).resolve()
            if not target.is_relative_to(WEB_ROOT) or not target.is_file():
                raise JobError(404, "Not found.")
            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if content_type.startswith("text/"):
                content_type += "; charset=utf-8"
            self._send(200, target.read_bytes(), content_type)

    return Handler


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Photo to 3D server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    if args.host not in ("127.0.0.1", "localhost", "::1") and not settings.app_token:
        log.warning("Listening on %s without P3D_TOKEN. Anyone who can reach this port can use the app.", args.host)

    service = JobService(settings)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service, settings))
    log.info("Serving on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
