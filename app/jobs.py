"""Job records, the background worker and the service layer used by the HTTP server."""
from __future__ import annotations

import copy
import json
import logging
import queue
import re
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .imaging import PhotoError, decode_photo
from .pipeline import PipelineError, run_pipeline

log = logging.getLogger("photo3d.jobs")

JOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")
PHOTO_NAME_RE = re.compile(r"^photo-([1-5])\.png$")
STATIC_ARTIFACTS = {"mask.png", "mesh.stl", "model.3mf"}
STAGE_PROGRESS = {
    "queued": 0, "segment": 15, "reconstruct": 40, "scale": 60,
    "clean": 75, "check": 85, "export": 95, "done": 100,
}


class JobError(Exception):
    """An error with an HTTP status code the user can see."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = Path(settings.data_dir) / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: dict[str, dict] = {}
        self._starts: deque[float] = deque()
        self._queue: queue.Queue[str] = queue.Queue()
        self._load_existing()
        threading.Thread(target=self._worker, name="pipeline-worker", daemon=True).start()
        for job_id, job in list(self._jobs.items()):
            if job["status"] == "queued":
                self._queue.put(job_id)

    # ----- persistence -------------------------------------------------------

    def _job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def _write(self, job: dict) -> None:
        (self._job_dir(job["id"]) / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")

    def _load_existing(self) -> None:
        for path in self.root.glob("*/job.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                log.warning("Skipping unreadable job record %s", path)
                continue
            if job.get("status") == "running":
                job.update(status="failed", error="Interrupted by a server restart. Start the job again.",
                           error_stage=job.get("stage"))
                self._write(job)
            self._jobs[job["id"]] = job

    # ----- lookups -----------------------------------------------------------

    def _require(self, job_id: str) -> dict:
        if not JOB_ID_RE.match(job_id):
            raise JobError(404, "Job not found.")
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise JobError(404, "Job not found.")
        return job

    def _public(self, job: dict) -> dict:
        with self._lock:
            out = copy.deepcopy(job)
        out["artifacts"] = ["mask.png", "mesh.stl", "model.3mf"] if job["status"] == "done" else []
        out["artifacts"] += [f"photo-{i}.png" for i in range(1, job["photos"] + 1)]
        return out

    # ----- commands ----------------------------------------------------------

    def create(self, reference_width_mm=None, name=None) -> dict:
        if reference_width_mm is not None:
            valid = isinstance(reference_width_mm, (int, float)) and not isinstance(reference_width_mm, bool)
            if not valid or not 1 <= reference_width_mm <= 2000:
                raise JobError(400, "Reference width must be a number between 1 and 2000 mm.")
            reference_width_mm = float(reference_width_mm)
        if name is not None:
            if not isinstance(name, str):
                raise JobError(400, "Name must be text.")
            name = name.strip()[:80] or None

        job_id = uuid.uuid4().hex[:12]
        now = _now()
        job = {
            "id": job_id, "name": name, "status": "draft", "stage": "draft", "progress": 0,
            "photos": 0, "reference_width_mm": reference_width_mm,
            "created_at": now, "updated_at": now,
            "error": None, "error_stage": None, "scale": None, "report": None,
            "notes": [], "backend": None,
        }
        (self._job_dir(job_id) / "photos").mkdir(parents=True)
        with self._lock:
            self._jobs[job_id] = job
            self._write(job)
        return self._public(job)

    def add_photo(self, job_id: str, raw: bytes) -> dict:
        job = self._require(job_id)
        with self._lock:
            if job["status"] != "draft":
                raise JobError(409, "Photos can only be added before a job starts.")
            if job["photos"] >= self.settings.max_photos:
                raise JobError(400, f"A job takes at most {self.settings.max_photos} photos.")
        if not raw:
            raise JobError(400, "The upload was empty.")
        if len(raw) > self.settings.max_upload_bytes:
            raise JobError(413, "Photo is larger than the upload limit.")
        try:
            image = decode_photo(raw, self.settings.max_pixels)
        except PhotoError as exc:
            raise JobError(400, str(exc)) from exc

        with self._lock:
            number = job["photos"] + 1
            image.save(self._job_dir(job_id) / "photos" / f"photo-{number}.png", format="PNG")
            job["photos"] = number
            job["updated_at"] = _now()
            self._write(job)
            return self._public(job)

    def start(self, job_id: str) -> dict:
        job = self._require(job_id)
        with self._lock:
            if job["status"] != "draft":
                raise JobError(409, "This job has already started.")
            if job["photos"] == 0:
                raise JobError(400, "Add at least one photo before starting.")
            now = time.monotonic()
            while self._starts and now - self._starts[0] > self.settings.rate_window_s:
                self._starts.popleft()
            if len(self._starts) >= self.settings.jobs_per_window:
                raise JobError(429, "Too many conversions recently. Try again in a few minutes.")
            self._starts.append(now)
            job.update(status="queued", stage="queued", progress=0, updated_at=_now())
            self._write(job)
            public = self._public(job)
        self._queue.put(job_id)
        return public

    def list_jobs(self) -> list[dict]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j["created_at"], reverse=True)
        return [self._public(j) for j in jobs]

    def get(self, job_id: str) -> dict:
        return self._public(self._require(job_id))

    def artifact_path(self, job_id: str, name: str) -> Path:
        job = self._require(job_id)
        if name in STATIC_ARTIFACTS:
            path = self._job_dir(job_id) / name
        else:
            match = PHOTO_NAME_RE.match(name)
            if not match or int(match.group(1)) > job["photos"]:
                raise JobError(404, "File not found.")
            path = self._job_dir(job_id) / "photos" / name
        if not path.is_file():
            raise JobError(404, "File not found.")
        return path

    # ----- worker ------------------------------------------------------------

    def _worker(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._run(job_id)
            except Exception:
                log.exception("Worker failed on job %s", job_id)
            finally:
                self._queue.task_done()

    def _update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.update(fields)
            job["updated_at"] = _now()
            self._write(job)

    def _run(self, job_id: str) -> None:
        job = self._jobs[job_id]
        self._update(job_id, status="running")

        def progress(stage: str) -> None:
            self._update(job_id, stage=stage, progress=STAGE_PROGRESS.get(stage, 0))

        try:
            result = run_pipeline(self._job_dir(job_id), self.settings, job["reference_width_mm"], progress)
        except PipelineError as exc:
            log.info("Job %s stopped at %s: %s", job_id, exc.stage, exc)
            self._update(job_id, status="failed", error=str(exc), error_stage=exc.stage)
            return
        except Exception:
            log.exception("Pipeline crashed for job %s", job_id)
            self._update(job_id, status="failed", error="Unexpected error. The server log has details.",
                         error_stage=job["stage"])
            return

        self._update(
            job_id, status="done", stage="done", progress=100,
            scale=result["scale"], report=result["report"], notes=result["notes"],
            backend=result["backend"], error=None, error_stage=None,
        )
