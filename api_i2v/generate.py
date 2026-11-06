"""Generate Track 1 videos with a Seedance-compatible Ark endpoint."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote, urlsplit

import requests


def extract_task_id(data: Dict[str, Any]) -> str:
    candidates = [data.get('id'), data.get('task_id'), data.get('taskId'), data.get('data', {}).get('id') if isinstance(data.get('data'), dict) else None, data.get('data', {}).get('task_id') if isinstance(data.get('data'), dict) else None, data.get('result', {}).get('id') if isinstance(data.get('result'), dict) else None, data.get('result', {}).get('task_id') if isinstance(data.get('result'), dict) else None]
    for item in candidates:
        if isinstance(item, str) and item:
            return item
    raise RuntimeError('Cannot extract task id from create response:\n' + json.dumps(data, ensure_ascii=False, indent=2))


def extract_status(data: Dict[str, Any]) -> str:
    status = data.get('status')
    if status is None and isinstance(data.get('data'), dict):
        status = data['data'].get('status')
    if status is None and isinstance(data.get('result'), dict):
        status = data['result'].get('status')
    return str(status or '').lower()


def extract_video_url(data: Dict[str, Any]) -> str:
    paths = [('content', 'video_url'), ('content', 'url'), ('output', 'video_url'), ('output', 'url'), ('data', 'content', 'video_url'), ('data', 'output', 'video_url'), ('result', 'content', 'video_url'), ('result', 'output', 'video_url')]
    for path in paths:
        obj: Any = data
        ok = True
        for key in path:
            if not isinstance(obj, dict) or key not in obj:
                ok = False
                break
            obj = obj[key]
        if ok and isinstance(obj, str) and obj:
            return obj
    for key_path in [('output', 'urls'), ('data', 'output', 'urls'), ('result', 'output', 'urls'), ('urls',)]:
        obj = data
        ok = True
        for key in key_path:
            if not isinstance(obj, dict) or key not in obj:
                ok = False
                break
            obj = obj[key]
        if ok and isinstance(obj, list) and obj and isinstance(obj[0], str):
            return obj[0]
    raise RuntimeError('Task succeeded but cannot extract video URL from response:\n' + json.dumps(data, ensure_ascii=False, indent=2))


def build_payload(prompt: str, image_data_url: str, resolution: str, ratio: str, duration: int, seed: Optional[int], watermark: bool, generate_audio: bool, role: str, *, model) -> Dict[str, Any]:
    payload: Dict[str, Any] = {'model': model, 'content': [{'type': 'text', 'text': prompt}, {'type': 'image_url', 'image_url': {'url': image_data_url}, 'role': role}], 'resolution': resolution, 'ratio': ratio, 'duration': duration, 'watermark': watermark, 'generate_audio': generate_audio}
    if seed is not None:
        payload['seed'] = seed
    return payload


def atomic_json(path, data):
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temp_path = Path(stream.name)
        try:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        except BaseException:
            stream.close()
            temp_path.unlink(missing_ok=True)
            raise
    try:
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def data_url(image):
    mime = mimetypes.guess_type(image.name)[0]
    if mime not in {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif"}:
        raise ValueError("Unsupported input image extension")
    return f"data:{mime};base64," + base64.b64encode(image.read_bytes()).decode("ascii")


def filename(value):
    value = str(value)
    if not value or value in {".", ".."} or any(ch in value for ch in '/\\:*?"<>|\0'):
        raise ValueError("Expected a single portable filename")
    return value


def batch_items(args):
    if args.json_file:
        rows = json.loads(args.json_file.read_text(encoding="utf-8-sig"))
        if isinstance(rows, dict):
            rows = next((rows[k] for k in ("items", "data", "cases", "records")
                         if isinstance(rows.get(k), list)), None)
        if not isinstance(rows, list) or not rows:
            raise ValueError("Expected a nonempty JSON list of cases")
        return rows

    dirs = [args.input_dir] if args.input_dir else sorted(p for p in args.case_root.iterdir() if p.is_dir())
    if not dirs:
        raise ValueError("No case directories found")
    return dirs


def resolve_case(item, index, args):
    if args.json_file:
        if not isinstance(item, dict):
            raise ValueError("Expected a case object")
        case_id = next((str(item[k]) for k in (args.case_id_key, "id_index", "case_id", "id", "name")
                        if item.get(k) is not None and str(item[k]).strip()), "")
        safe_id = re.sub(r"[^\w.-]+", "_", case_id).strip("._")
        dirname = f"{index:06d}_{safe_id}" if safe_id else f"item_{index:06d}"
        prompt = item.get(args.prompt_key)
        image = item.get(args.image_key)
        if not isinstance(prompt, str) or not prompt.strip() or not isinstance(image, str) or not image:
            raise ValueError("Expected an image path and prompt text")
        prompt = prompt.strip()
        image = args.json_file.parent / image
        case_id = case_id or dirname
        name = args.output_name or item.get(args.output_name_key) or args.default_output_name
    else:
        folder = item
        prompt = (folder / args.prompt_file).read_text(encoding="utf-8-sig").strip()
        if not prompt:
            raise ValueError("Empty prompt file")
        image = folder / args.image_file
        name = args.output_name or f"{args.prompt_file.stem}__{image.stem}__seedance2.mp4"
        case_id = folder.resolve().name
        dirname = filename(case_id)
    name = filename(name)
    if not name.lower().endswith(".mp4"):
        name += ".mp4"
    return case_id, dirname, prompt, image, name


def api_json(method, url, headers, timeout, **kwargs):
    try:
        with requests.request(method, url, headers=headers, timeout=timeout, allow_redirects=False, **kwargs) as response:
            if not 200 <= response.status_code < 300:
                raise RuntimeError(f"API {method} returned HTTP {response.status_code}")
            return response.json()
    except requests.RequestException:
        raise RuntimeError(f"API {method} network or response error") from None


def poll(url, headers, args):
    deadline = time.monotonic() + args.max_wait_seconds
    while True:
        result = api_json("GET", url, headers, args.request_timeout)
        status = str(response_field(extract_status, result) or "").lower()
        if status in {"succeeded", "success", "completed", "finished", "done"}:
            url = response_field(extract_video_url, result)
            if not isinstance(url, str) or urlsplit(url).scheme != "https":
                raise RuntimeError("Successful task returned no HTTPS video URL")
            return url
        if status in {"failed", "failure", "error", "expired", "cancelled", "canceled"}:
            raise TerminalTaskError("Generation ended with status " + status)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("Polling timed out; rerun with the same settings to resume")
        time.sleep(min(args.poll_interval, remaining))


class TerminalTaskError(RuntimeError):
    pass


def response_field(extractor, data):
    try:
        return extractor(data)
    except (KeyError, TypeError, ValueError, RuntimeError):
        raise RuntimeError("Unexpected API response structure") from None


def download_video(url, target, timeout):
    # The signed download link is separate from the authenticated task endpoint.
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            with requests.get(url, timeout=timeout, stream=True) as res:
                if not 200 <= res.status_code < 300:
                    raise RuntimeError(f"Video download returned HTTP {res.status_code}")
                for chunk in res.iter_content(chunk_size=1024 * 1024):
                    stream.write(chunk)
            if stream.tell() == 0:
                raise RuntimeError("Empty video download")
            stream.close()
            tmp.replace(target)
        except requests.RequestException:
            raise RuntimeError("Video download failed; rerun to resume") from None
        finally:
            stream.close()
            tmp.unlink(missing_ok=True)


@contextmanager
def exclusive_job(target):
    lock = target.with_suffix(target.suffix + ".busy")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError("Case is busy; delete its .busy file only after confirming no process is still working on it") from None
    os.close(fd)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def generate(case, args, endpoint, headers, model):
    case_id, dirname, prompt, image, name = case
    folder = args.output_root / dirname
    folder.mkdir(parents=True, exist_ok=True)
    video = folder / name
    with exclusive_job(video):
        state_path = video.with_suffix(video.suffix + ".pending.json")
        meta_path = video.with_suffix(video.suffix + ".case.json")
        if not args.overwrite and not state_path.exists() and video.is_file() and video.stat().st_size:
            atomic_json(meta_path, {"id_index": case_id})
            return "skipped"
        payload = build_payload(prompt, data_url(image), args.resolution, args.ratio, args.duration,
                                args.seed, args.watermark, args.generate_audio, args.role, model=model)
        digest = hashlib.sha256(json.dumps([endpoint, payload], sort_keys=True).encode()).hexdigest()
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("fingerprint") != digest or not state.get("id"):
                raise RuntimeError("Unresolved or mismatched previous submission; check the provider console before removing .pending.json")
            task_id = state["id"]
        else:
            atomic_json(state_path, {"fingerprint": digest})
            task = api_json("POST", endpoint, headers, args.request_timeout, json=payload)
            task_id = response_field(extract_task_id, task)
            if not task_id:
                raise RuntimeError("Submission returned no task ID; check your provider console")
            atomic_json(state_path, {"fingerprint": digest, "id": task_id})
        try:
            url = poll(endpoint + "/" + quote(str(task_id), safe=""), headers, args)
        except TerminalTaskError:
            state_path.unlink(missing_ok=True)
            raise
        download_video(url, video, args.request_timeout)
        atomic_json(meta_path, {"id_index": case_id})
        state_path.unlink()
        return "success"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_dir", type=Path, help="One case directory")
    group.add_argument("--case_root", type=Path, help="Dataset root containing case directories")
    group.add_argument("--json_file", type=Path, help="Generation manifest")
    p.add_argument("--output_root", required=True, type=Path)
    p.add_argument("--image_file", type=Path)
    p.add_argument("--prompt_file", type=Path)
    p.add_argument("--image_key", default="img")
    p.add_argument("--prompt_key", default="prompt")
    p.add_argument("--case_id_key", default="id_index")
    p.add_argument("--output_name_key", default="output_name")
    p.add_argument("--output_name")
    p.add_argument("--default_output_name", default="seedance2_fast.mp4", help="Fallback MP4 name in JSON mode")
    p.add_argument("--resolution", choices=["480p", "720p", "1080p"], default="720p")
    p.add_argument("--ratio", choices=["adaptive","16:9","4:3","1:1","3:4","9:16","21:9"], default="adaptive")
    p.add_argument("--duration", type=int, default=5)
    p.add_argument("--role", choices=["first_frame", "reference_image"], default="first_frame")
    p.add_argument("--seed", type=int)
    p.add_argument("--watermark", action="store_true")
    p.add_argument("--generate_audio", action="store_true")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--skip_existing", dest="overwrite", action="store_false", help="Skip existing outputs (default)")
    mode.add_argument("--overwrite", dest="overwrite", action="store_true", help="Submit a new generation for completed outputs")
    p.set_defaults(overwrite=False)
    p.add_argument("--stop_on_error", action="store_true", help="End a batch at the first failure")
    p.add_argument("--poll_interval", type=float, default=10)
    p.add_argument("--request_timeout", type=float, default=120)
    p.add_argument("--max_wait_seconds", type=float, default=1800)
    args = p.parse_args()
    if not args.json_file and not (args.image_file and args.prompt_file):
        p.error("directory mode requires --image_file and --prompt_file")
    if min(args.poll_interval, args.request_timeout, args.max_wait_seconds) <= 0:
        p.error("timeouts and intervals must be positive")
    return args


def main():
    args = parse_args()
    model = os.environ.get("SEEDANCE_MODEL", "").strip()
    token = os.environ.get("ARK_API_KEY", "").strip()
    if not token or not model:
        raise SystemExit("Set ARK_API_KEY and SEEDANCE_MODEL in the environment.")
    base = os.getenv("SEEDANCE_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3").rstrip("/")
    parsed = urlsplit(base)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise SystemExit("SEEDANCE_BASE_URL must be an HTTPS API base URL without credentials or query parameters.")
    endpoint = base + "/contents/generations/tasks"
    headers = {"Authorization": "Bearer " + token, "Content-Type": "application/json"}
    args.output_root.mkdir(parents=True, exist_ok=True)
    counts = {"processed": 0, "success": 0, "failed": 0, "skipped": 0}
    for index, item in enumerate(batch_items(args), 1):
        counts["processed"] += 1
        label = f"item_{index:06d}" if args.json_file else item.name
        try:
            case = resolve_case(item, index, args)
            label = case[1]
            status = generate(case, args, endpoint, headers, model)
            counts[status] += 1
            print(f"{label}: {status}")
        except (OSError, ValueError, RuntimeError) as error:
            counts["failed"] += 1
            message = str(error) if isinstance(error, (RuntimeError, ValueError)) else type(error).__name__
            print(f"{label}: {message}", file=sys.stderr)
            if args.stop_on_error:
                break
    print(json.dumps(counts))
    return int(counts["failed"] != 0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as error:
        sys.exit("Invalid manifest or unavailable input: " + type(error).__name__)
