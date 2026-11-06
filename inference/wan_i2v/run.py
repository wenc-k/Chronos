"""Run the archived Wan Track 1 or Track 2 inference on one or more GPUs."""
import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from _workflow import merge_track1, prepare_track2_data


def option(arguments, name, default=None):
    result = default
    for index, token in enumerate(arguments):
        if token == name:
            if index + 1 == len(arguments):
                raise ValueError(f'{name} needs a value')
            result = arguments[index + 1]
        if token.startswith(name + '='):
            result = token.split('=', 1)[1]
    return result


def with_option(arguments, name, value):
    result = []
    skip = False
    for token in arguments:
        if skip:
            skip = False
            continue
        if token == name:
            skip = True
            continue
        if not token.startswith(name + '='):
            result.append(token)
    result.extend((name, str(value)))
    return result


def parse_ranges(text, gpus, track):
    if text is None:
        if len(gpus) != 1:
            raise ValueError('Supply one disjoint --ranges interval per GPU')
        return [None]
    ranges = []
    numeric_ranges = []
    for field in text.split(','):
        bounds = field.split(':')
        if len(bounds) != 2:
            raise ValueError('Use start:end intervals separated by commas')
        numbers = []
        for bound in bounds:
            expression = r'(?:id)?([0-9]+)' if track == 'track1' else r'([0-9]+)'
            match = re.fullmatch(expression, bound, flags=re.IGNORECASE)
            if not match:
                raise ValueError('Range bounds must be numeric IDs, optionally prefixed by id in Track 1')
            numbers.append(int(match.group(1)))
        start, end = numbers
        exclusive_end = end + 1 if track == 'track1' else end
        if start >= exclusive_end:
            raise ValueError('Every range must contain at least one case')
        numeric_ranges.append((start, exclusive_end))
        ranges.append(tuple(bounds))
    if len(ranges) != len(gpus):
        raise ValueError('The number of ranges must equal the number of GPUs')
    ordered = sorted(numeric_ranges)
    if any(right[0] < left[1] for left, right in zip(ordered, ordered[1:])):
        raise ValueError('GPU ranges must not overlap')
    return ranges


def prepare_plan(args):
    forwarded = args.arguments
    if forwarded and forwarded[0] == '--':
        forwarded = forwarded[1:]
    if args.ranges and any(option(forwarded, flag) is not None for flag in ('--start_id', '--end_id')):
        raise ValueError('Use --ranges or forwarded --start_id/--end_id flags in one command')
    gpus = [item.strip() for item in args.gpus.split(',')]
    if any(not item.strip() for item in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError('GPU IDs must be nonempty and distinct')
    ranges = parse_ranges(args.ranges, gpus, args.track)
    path_options = ('--ckpt_dir', '--batch_root', '--batch_json', '--batch_output_dir',
                    '--batch_output_json', '--json_file', '--output_dir')
    for flag in path_options:
        value = option(forwarded, flag)
        if value is not None:
            forwarded = with_option(forwarded, flag, Path(value).expanduser().resolve())
    if option(forwarded, '--ckpt_dir') is None:
        raise ValueError('Pass --ckpt_dir after the -- separator')
    if not Path(option(forwarded, '--ckpt_dir')).is_dir():
        raise ValueError('Checkpoint directory does not exist')
    wan_root = Path(args.wan_root).expanduser().resolve()
    if not (wan_root / 'wan' / 'image2video.py').is_file():
        raise ValueError('Install the Wan source with inference/environment, or specify --wan-root')
    final_json = None
    shard_dir = None
    if args.track == 'track1':
        batch_json, batch_root = option(forwarded, '--batch_json'), option(forwarded, '--batch_root')
        if bool(batch_json) == bool(batch_root):
            raise ValueError('Choose exactly one of --batch_json and --batch_root')
        if batch_json and not Path(batch_json).is_file():
            raise ValueError('Batch JSON does not exist')
        if batch_root and not Path(batch_root).is_dir():
            raise ValueError('Batch root does not exist')
        output_dir = Path(option(forwarded, '--batch_output_dir', 'outputs/wan_track1')).resolve()
        forwarded = with_option(forwarded, '--batch_output_dir', output_dir)
        final_json = Path(option(forwarded, '--batch_output_json', str(output_dir / 'results.json'))).resolve()
        fingerprint = hashlib.sha256(json.dumps([forwarded, ranges], sort_keys=True).encode()).hexdigest()[:16]
        shard_dir = final_json.parent / ('.' + final_json.stem + '-shards') / fingerprint
    else:
        manifest = option(forwarded, '--json_file')
        if not manifest:
            raise ValueError('Track 2 needs --json_file')
        with Path(manifest).open(encoding='utf-8-sig') as stream:
            data = json.load(stream)
        prepare_track2_data(data, manifest, int(option(forwarded, '--video_duration', '5')),
                            int(option(forwarded, '--video_fps', '16')))
        output_dir = Path(option(forwarded, '--output_dir', 'outputs/wan_track2')).resolve()
        forwarded = with_option(forwarded, '--output_dir', output_dir)
    script = 'generate_i2v_batch.py' if args.track == 'track1' else 'track2_eval.py'
    plans = []
    for index, (gpu, interval) in enumerate(zip(gpus, ranges)):
        flags = forwarded[:]
        if interval:
            flags = with_option(with_option(flags, '--start_id', interval[0]), '--end_id', interval[1])
        if final_json:
            flags = with_option(flags, '--batch_output_json', shard_dir / f'part-{index}.json')
        environment = os.environ.copy()
        environment['CUDA_VISIBLE_DEVICES'] = gpu
        environment['WAN22_ROOT'] = str(wan_root)
        command = [sys.executable, str(Path(__file__).with_name(script)), *flags]
        plans.append((command, environment))
    return plans, wan_root, final_json, shard_dir, output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('track', choices=('track1', 'track2'))
    parser.add_argument('--wan-root', default=os.environ.get('WAN22_ROOT', str(Path(__file__).resolve().parents[1] / '.deps' / 'Wan2.2')))
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--ranges', help='One disjoint start:end range per GPU. Track 1 ends are inclusive. Track 2 ends are exclusive.')
    parser.add_argument('--dry-run', action='store_true', help='Validate paths and print commands without loading a model')
    # parse_known_args retains the model-specific flags after the separator.
    args, args.arguments = parser.parse_known_args()
    try:
        plans, wan_root, final_json, shard_dir, output_dir = prepare_plan(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        for command, environment in plans:
            print('CUDA_VISIBLE_DEVICES=' + shlex.quote(environment['CUDA_VISIBLE_DEVICES']) + ' ' + shlex.join(command))
        return 0
    lock_root = final_json.parent if final_json else output_dir
    lock_root.mkdir(parents=True, exist_ok=True)
    lock = lock_root / ('.' + final_json.name + '.lock' if final_json else '.wan-track2.lock')
    try:
        with lock.open('x', encoding='utf-8') as stream:
            stream.write(str(os.getpid()) + '\n')
    except FileExistsError:
        parser.error(f'Another run holds {lock}; inspect that run before removing a stale lock')
    processes = []
    try:
        if shard_dir:
            shard_dir.mkdir(parents=True, exist_ok=True)
        for command, environment in plans:
            processes.append(subprocess.Popen(command, env=environment, cwd=wan_root))
        statuses = [process.wait() for process in processes]
        if any(status != 0 for status in statuses):
            print('At least one GPU process failed. Completed shard files are retained.', file=sys.stderr)
            return 1
        if final_json:
            parts = list(shard_dir.glob('*.json'))
            if not parts:
                raise RuntimeError('No Track 1 result shards were written')
            merge_track1(final_json, parts)
            print(f'Combined Track 1 results in {final_json}')
        return 0
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    raise SystemExit(main())
