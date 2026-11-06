"""Edit a reference image into the first frame specified by a video prompt."""

import argparse
import base64
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional


SYSTEM_PROMPT = """You are an image-to-image editor for generating the first-frame reference image used by an image-to-video model.

Goal:
Generate one I2V-ready first-frame image from the given reference identity image and raw video prompt.

Strict identity requirements:
- Preserve the reference person's identity completely.
- Keep the same facial structure, facial features, age, gender presentation, skin tone, hairstyle, and recognizable identity cues.
- Do not beautify, stylize, caricature, age, de-age, or change the identity.
- Do not replace the person with a different person.

I2V first-frame requirements:
- The main subject must be front-facing or near-front-facing.
- The face must be clear, sharp, well-lit, and visible.
- The person should be suitable as the identity anchor for the following I2V generation.
- Avoid back view, side profile, occluded face, tiny face, blurred face, sunglasses, masks, heavy shadow, or cropped-out head.
- centered composition, leave some space above and below the subject, safe for later crop to a 832x480 landscape frame

Scene requirements:
- Follow the semantic scene, action, outfit context, and environment implied by the raw prompt.
- If the raw prompt contains a human subject, make the reference person that subject.
- If the raw prompt contains multiple subjects, keep the reference person as the primary identity subject with a clearly visible frontal face.
- Keep the image natural, realistic, and consistent with the raw prompt.

Output requirements:
- Output only the edited image.
- No text, watermark, border, collage, or UI elements.
"""


SYSTEM_PROMPT_SKETCH = """You are an image-to-image editor specialized in character design line art.

Transform the input reference image into a front-facing line-art character design portrait of an anonymous fictional person.

Preserve the reference facial structure closely: eyes, eyebrows, nose, mouth, and face-shape proportions should remain highly consistent with the reference.

Render the result as high-definition realistic pencil line art with clean, sharp lines, a pure white background, and a front-facing head portrait composition.

Do not use anime exaggeration, photorealistic texture, celebrity-like styling, influencer-face styling, or direct real-person portrait replication.

Keep the face unobstructed and suitable for stable identity consistency in later video generation.

Output only the final image, with no text or watermarks.
"""


def build_i2i_prompt(raw_prompt: str, extra_instruction: Optional[str] = None) -> str:
    user_prompt = f"""Raw video prompt:
{raw_prompt}

Task:
Using the attached reference identity image, generate a single realistic front-facing first-frame image for an I2V model.

The generated image must match the raw prompt's semantic scene while preserving the reference person's identity exactly.
The subject's face must be frontal or near-frontal, clear, sharp, and unobstructed.
"""

    if extra_instruction:
        user_prompt += f"\nAdditional instruction:\n{extra_instruction.strip()}\n"

    return f"{SYSTEM_PROMPT.strip()}\n\n{user_prompt.strip()}"


def build_i2i_sketch_prompt(raw_prompt: str, extra_instruction: Optional[str] = None) -> str:
    user_prompt = f"""Task:
Please generate a front-facing character design line drawing based on the reference image.

The result should be an anonymous fictional character, but the face must closely follow the reference image, especially the eyes, eyebrows, nose, mouth, and face-shape proportions.

Style: realistic hand-drawn pencil line art, clean and clear lines, pure white background, front-facing head portrait, character design sheet style, realistic human proportions, no anime exaggeration, no photorealistic texture.

The face must be clear, stable, symmetric, and unobstructed, and suitable for consistent later video generation.
"""

    if extra_instruction:
        user_prompt += f"\nAdditional instruction:\n{extra_instruction.strip()}\n"

    return f"{SYSTEM_PROMPT_SKETCH.strip()}\n\n{user_prompt.strip()}"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_dir', type=Path, required=True,
                        help='A case directory or a parent containing case directories.')
    parser.add_argument('--ref_image_name', default='image.png')
    parser.add_argument('--prompt_file_name', default='prompt1.txt')
    parser.add_argument('--output_image_name', help='Output filename inside each case directory.')
    parser.add_argument('--model', default=os.environ.get('OPENAI_IMAGE_MODEL', 'gpt-image-2'))
    parser.add_argument('--size', default='1536x1024')
    parser.add_argument('--quality', default='high')
    parser.add_argument('--output_format', choices=('png', 'jpeg', 'webp'), default='png')
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--max_retry', type=int, default=3,
                        help='Maximum number of attempts for retryable API failures.')
    parser.add_argument('--extra_instruction')
    parser.add_argument('--use_sketch_prompt', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.timeout <= 0 or args.max_retry < 1:
        parser.error('--timeout must be positive and --max_retry must be at least one')
    if args.output_image_name is None:
        args.output_image_name = 'first_frame.' + args.output_format
    for name in (args.ref_image_name, args.prompt_file_name, args.output_image_name):
        if not name or name in {'.', '..'} or Path(name).name != name:
            parser.error('Input and output names must be simple filenames')
    suffix = Path(args.output_image_name).suffix.lower().lstrip('.')
    if suffix == 'jpg':
        suffix = 'jpeg'
    if suffix != args.output_format:
        parser.error('Output filename extension must match --output_format')
    if args.output_image_name in {args.ref_image_name, args.prompt_file_name}:
        parser.error('Choose separate output and input filenames')
    return args


def case_directories(args):
    root = args.input_dir
    if not root.is_dir():
        raise ValueError('Input directory does not exist')
    def has_input(folder):
        return (folder / args.ref_image_name).is_file() or (folder / args.prompt_file_name).is_file()
    if has_input(root):
        return [root]
    cases = sorted(folder for folder in root.iterdir() if folder.is_dir() and has_input(folder))
    if not cases:
        raise ValueError('No case directories containing the reference or prompt were found')
    return cases


def render_case(client, case, args):
    image_path = case / args.ref_image_name
    prompt_path = case / args.prompt_file_name
    destination = case / args.output_image_name
    if destination.resolve() in {image_path.resolve(), prompt_path.resolve()}:
        raise ValueError('Output resolves to an input file')
    if destination.is_file() and destination.stat().st_size and not args.overwrite:
        return 'skipped'
    raw_prompt = prompt_path.read_text(encoding='utf-8-sig').strip()
    if not raw_prompt:
        raise ValueError('Empty input prompt')
    build = build_i2i_sketch_prompt if args.use_sketch_prompt else build_i2i_prompt
    with image_path.open('rb') as reference:
        result = client.images.edit(model=args.model, image=reference,
                                    prompt=build(raw_prompt, args.extra_instruction),
                                    size=args.size, quality=args.quality,
                                    output_format=args.output_format, n=1)
    payload = result.data[0].b64_json if result.data else None
    if not payload:
        raise ValueError('The image service returned no base64 image')
    image = base64.b64decode(payload, validate=True)
    with NamedTemporaryFile(dir=case, prefix='.first-frame-', delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        temporary.write_bytes(image)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return 'written'


def main():
    args = parse_args()
    try:
        cases = case_directories(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    if not os.environ.get('OPENAI_API_KEY'):
        raise SystemExit('Set OPENAI_API_KEY before running image generation')
    from openai import OpenAI
    client = OpenAI(timeout=args.timeout, max_retries=args.max_retry - 1)
    errors = 0
    for case in cases:
        try:
            print(f'{case.name}: {render_case(client, case, args)}')
        except (FileNotFoundError, PermissionError) as exc:
            print(f'{case.name}: {type(exc).__name__} ({Path(exc.filename).name})', file=sys.stderr)
            errors += 1
        except Exception as exc:
            status = getattr(exc, 'status_code', None)
            detail = f' (HTTP {status})' if isinstance(status, int) else ''
            print(f'{case.name}: {type(exc).__name__}{detail}', file=sys.stderr)
            errors += 1
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
