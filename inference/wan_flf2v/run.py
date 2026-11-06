"""Submit first/last-frame segments to ComfyUI and resume incomplete jobs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import mimetypes
import os
import re
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import requests


def atomic_json(path, value):
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        try:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.close()
            temp.replace(path)
        finally:
            f.close()
            temp.unlink(missing_ok=True)


def relative_name(value):
    value = str(value).replace("\\", "/")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or any(part in {"", ".."} for part in path.parts):
        raise ValueError("Expected a relative output prefix")
    if not re.fullmatch(r"[A-Za-z0-9_. /-]+", value):
        raise ValueError("Output prefix contains unsupported characters")
    return path.as_posix()


def case_id(value):
    value = str(value)
    if value in {"", ".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError("Job id must contain letters, digits, underscores, dots or hyphens")
    return value


def api_json(method, server, route, timeout, **kwargs):
    try:
        with requests.request(method, server + route, timeout=timeout, allow_redirects=False, **kwargs) as response:
            if response.status_code >= 300:
                raise RuntimeError(f"ComfyUI {method} returned HTTP {response.status_code}")
            return response.json()
    except requests.RequestException:
        raise RuntimeError(f"ComfyUI {method} network or response error") from None


def upload_image(server, image, timeout):
    mime = mimetypes.guess_type(image.name)[0] or "application/octet-stream"
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    with image.open("rb") as stream:
        result = api_json("POST", server, "/upload/image", timeout,
                          files={"image": (digest + image.suffix.lower(), stream, mime)},
                          data={"type": "input", "subfolder": "acm_mm_flf2v", "overwrite": "true"})
    if result.get("type", "input") != "input" or not result.get("name"):
        raise RuntimeError("Unexpected ComfyUI image upload response")
    return relative_name(str(PurePosixPath(result.get("subfolder", "")) / result["name"]))


def jobs(path):
    seen = set()
    with path.open(encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                job = json.loads(line)
                identifier = case_id(job["id"])
                if identifier in seen:
                    raise ValueError("Duplicate job id")
                seen.add(identifier)
                if not isinstance(job["prompt"], str) or not job["prompt"].strip():
                    raise ValueError("Empty prompt")
                for key in ("start_image", "end_image"):
                    if not isinstance(job[key], str) or not job[key]:
                        raise ValueError("Expected an image filename")
                    job[key] = path.parent / job[key]
                if "output_prefix" in job:
                    relative_name(job["output_prefix"])
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"Invalid job on JSONL line {lineno}: {type(error).__name__}") from None
            yield job


def apply_job(template, job, args, start_name, end_name):
    graph = copy.deepcopy(template)
    graph["80"]["inputs"]["image"] = start_name
    graph["89"]["inputs"]["image"] = end_name
    graph["90"]["inputs"]["text"] = job["prompt"].strip()
    graph["83"]["inputs"]["filename_prefix"] = relative_name(job.get("output_prefix", "wan_flf2v/" + case_id(job["id"])))
    for node_id in ("84", "87"):
        graph[node_id]["inputs"]["noise_seed"] = int(job.get("seed", args.seed))
    image_node = graph["81"]["inputs"]
    image_node["length"] = int(job.get("length", args.length if args.length is not None else image_node["length"]))
    if image_node["length"] <= 0 or image_node["length"] % 4 != 1:
        raise ValueError("Frame length must be 4*n+1, for example 17 or 81")
    for key in ("width", "height"):
        if getattr(args, key) is not None:
            image_node[key] = getattr(args, key)
        if image_node[key] <= 0 or image_node[key] % 16:
            raise ValueError("Width and height must be positive multiples of 16")
    if args.fps is not None:
        graph["86"]["inputs"]["fps"] = args.fps
    for arg, node, field in (("unet_high", "76", "unet_name"), ("unet_low", "77", "unet_name"),
                             ("clip_name", "72", "clip_name"), ("vae_name", "79", "vae_name")):
        if getattr(args, arg):
            graph[node]["inputs"][field] = getattr(args, arg)
    return graph


def output_video(history):
    # SaveVideo uses the built-in PreviewVideo result, which exposes file metadata.
    outputs = history.get("outputs", {}).get("83", {})
    found = []
    for kind in ("images", "videos", "gifs"):
        for entry in outputs.get(kind, []):
            if isinstance(entry, dict) and Path(entry.get("filename", "")).suffix.lower() in {".mp4", ".webm", ".mov", ".mkv"}:
                found.append(entry)
    if len(found) != 1:
        raise RuntimeError("Expected one video from SaveVideo node 83")
    return found[0]


def wait_for_video(server, prompt_id, args):
    deadline = time.monotonic() + args.max_wait_seconds
    while True:
        data = api_json("GET", server, "/history/" + prompt_id, args.request_timeout).get(prompt_id)
        if data:
            status = data.get("status", {})
            if status.get("status_str") == "error":
                raise TerminalJobError("ComfyUI execution failed; inspect the ComfyUI server log")
            if status.get("completed"):
                return output_video(data)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("Polling timed out; rerun to resume")
        time.sleep(min(args.poll_interval, remaining))


class TerminalJobError(RuntimeError):
    pass


def download_video(server, item, target, timeout):
    params = {key: item.get(key, "") for key in ("filename", "subfolder", "type")}
    params["type"] = params["type"] or "output"
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            with requests.get(server + "/view", params=params, timeout=timeout, stream=True, allow_redirects=False) as response:
                if response.status_code != 200:
                    raise RuntimeError(f"ComfyUI video retrieval returned HTTP {response.status_code}")
                for chunk in response.iter_content(1024 * 1024):
                    stream.write(chunk)
            if not stream.tell():
                raise RuntimeError("ComfyUI returned an empty video")
            stream.close()
            temp.replace(target)
        except requests.RequestException:
            raise RuntimeError("Video retrieval failed; rerun to resume") from None
        finally:
            stream.close()
            temp.unlink(missing_ok=True)


@contextmanager
def locked(path):
    lock = path.with_suffix(".busy")
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise RuntimeError("Job is busy; delete its .busy file only after confirming no runner is still processing it") from None
    os.close(descriptor)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def run_job(job, template, args):
    identifier = case_id(job["id"])
    target = args.output_dir / (identifier + ".mp4")
    state_path = args.output_dir / (identifier + ".pending.json")
    with locked(target):
        if target.is_file() and target.stat().st_size and not state_path.exists() and not args.overwrite:
            return "skipped"
        # Validate graph overrides before uploading images or submitting a task.
        graph = apply_job(template, job, args, "first.png", "last.png")
        input_hashes = [hashlib.sha256(job[key].read_bytes()).hexdigest() for key in ("start_image", "end_image")]
        fingerprint = hashlib.sha256(json.dumps([args.server, graph, input_hashes], sort_keys=True).encode()).hexdigest()
        if args.dry_run:
            return "validated"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("fingerprint") != fingerprint or not state.get("prompt_id"):
                raise RuntimeError("Previous submission is unresolved or inputs changed; check the ComfyUI history/queue before removing .pending.json")
            prompt_id = state["prompt_id"]
        else:
            start = upload_image(args.server, job["start_image"], args.request_timeout)
            end = upload_image(args.server, job["end_image"], args.request_timeout)
            graph = apply_job(template, job, args, start, end)
            atomic_json(state_path, {"fingerprint": fingerprint})
            response = api_json("POST", args.server, "/prompt", args.request_timeout,
                                json={"prompt": graph, "client_id": uuid.uuid4().hex})
            prompt_id = response.get("prompt_id")
            if not isinstance(prompt_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", prompt_id):
                raise RuntimeError("Submission returned no valid prompt ID; inspect ComfyUI history")
            atomic_json(state_path, {"fingerprint": fingerprint, "prompt_id": prompt_id})
        try:
            result = wait_for_video(args.server, prompt_id, args)
        except TerminalJobError:
            state_path.unlink(missing_ok=True)
            raise
        download_video(args.server, result, target, args.request_timeout)
        state_path.unlink()
        return "success"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, default=os.environ.get("JOBS_JSONL"))
    parser.add_argument("--workflow", type=Path, default=os.environ.get("WORKFLOW_API", str(Path(__file__).parent / "workflows/wan2_2_14b_flf2v.api.json")))
    parser.add_argument("--server", default=os.environ.get("COMFY_SERVER", "http://127.0.0.1:8188"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--length", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--unet-high", default=os.environ.get("WAN_UNET_HIGH"))
    parser.add_argument("--unet-low", default=os.environ.get("WAN_UNET_LOW"))
    parser.add_argument("--clip-name", default=os.environ.get("WAN_CLIP_NAME"))
    parser.add_argument("--vae-name", default=os.environ.get("WAN_VAE_NAME"))
    parser.add_argument("--poll-interval", type=float, default=5)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--max-wait-seconds", type=float, default=7200)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate jobs and workflow without HTTP requests")
    args = parser.parse_args()
    if not args.jobs:
        parser.error("provide --jobs or JOBS_JSONL")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be in [0, num-shards)")
    if min(args.poll_interval, args.request_timeout, args.max_wait_seconds) <= 0 or (args.fps is not None and args.fps <= 0):
        parser.error("timeouts, polling interval and FPS must be positive")
    parsed = urlsplit(args.server)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        parser.error("server must be an HTTP(S) origin or base URL without credentials or query")
    args.server = args.server.rstrip("/")
    return args


def main():
    args = parse_args()
    template = json.loads(args.workflow.read_text(encoding="utf-8-sig"))
    # Validate all JSONL identities first to avoid discovering duplicates after submission.
    entries = list(jobs(args.jobs))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = {"success": 0, "skipped": 0, "failed": 0, "validated": 0}
    for index, job in enumerate(entries):
        if index % args.num_shards != args.shard_index:
            continue
        try:
            status = run_job(job, template, args)
            counts[status] += 1
            print(f"{job['id']}: {status}")
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            counts["failed"] += 1
            message = str(error) if isinstance(error, RuntimeError) else type(error).__name__
            print(f"{job['id']}: {message}", file=sys.stderr)
            if args.stop_on_error:
                break
    print(json.dumps(counts))
    return int(counts["failed"] != 0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        sys.exit("Invalid workflow, manifest or file: " + type(error).__name__)
