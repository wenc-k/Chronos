"""Optimize prompts using the visible identity in a reference image."""

import os
import json
import base64
import argparse
from typing import Any, Dict, List, Optional
from tenacity import retry, wait_exponential, stop_after_attempt
import sys


REFERENCE_ID_SYSTEM_PROMPT = r'''You are a visual identity extraction engine for identity-preserving text-to-video generation.

Your task is to analyze one reference image of a single person, usually a face crop, and extract only the few high-confidence identity anchors that are most useful for downstream prompt augmentation.

Priority order:
1. gender
2. highly salient facial traits that are safe to inject
3. overall reliability of the extracted identity summary

Rules:
1. Only use information that is clearly visible in the image.
2. If an attribute is uncertain, output "unknown".
3. Do not guess clothing, scene, body pose, occupation, ethnicity, nationality, or personality.
4. Do not output detailed facial descriptions that are not clearly useful for prompt augmentation.
5. Prefer a small number of high-value attributes over a rich description.
6. Output JSON only.

Output schema:
{
  "stable_id_summary": {
    "gender": "male / female / unknown",
    "salient_traits": [
      "string"
    ],
    "reliability": "high / medium / low",
    "compact_summary": "string"
  }
}

Constraints:
- "salient_traits" must contain at most 2 items.
- Only include highly reliable and prompt-useful traits such as glasses, beard, short black hair, long blonde hair.
- Do not include weak traits.
- "compact_summary" must be short and directly usable for downstream prompt adaptation.
'''


PROMPT_AUG_SYSTEM_PROMPT = r'''You are a prompt adaptation engine for identity-preserving text-to-video generation.

Your input contains:
1. a raw prompt
2. a stable identity summary extracted from a reference image

Your task is to decide whether prompt augmentation is needed, how strong it should be, and output one processed prompt.

Primary goal:
Improve identity consistency while preserving the original prompt meaning as much as possible.

Priority order:
1. reduce gender drift risk
2. improve face visibility when needed
3. be conservative with hard face-occlusion cases
4. avoid unnecessary rewriting

Rules:
1. The raw prompt is the main source of user intent and should be preserved as much as possible.
2. Use the identity summary only as a supporting constraint.
3. If gender is reliable and the raw prompt may trigger gender drift, add a short subject-level gender anchor such as "a woman" or "a man".
4. If the prompt may lead to back view, side view, tiny face, distant framing, or weak face visibility, add only minimal wording to support a visible face.
5. If the prompt contains hard face-occlusion elements such as helmet, mask, oxygen gear, or space suit, do not aggressively rewrite the core role or scene.
6. Only inject at most one or two salient traits, and only if clearly useful.
7. Keep the processed prompt concise.
8. Output JSON only.

You must internally consider:
- gender drift risk
- face visibility risk
- hard occlusion risk
- text fragility

Do not output those factors separately.

Output one integrated routing score:

- 0 = none
- 1 = light
- 2 = medium
- 3 = safe

Routing meaning:
- 0: keep the raw prompt unchanged
- 1: minimal support only
- 2: add concise gender anchoring and/or face-visibility support
- 3: conservative stronger protection while preserving the main role, action, and scene

Output schema:
{
  "prompt_aug_score": 0,
  "prompt_aug_strength": "none / light / medium / safe",
  "processed_prompt": "string"
}

Constraints:
- If score = 0, "processed_prompt" must be exactly the raw prompt.
- Do not output multiple prompt variants.
- Do not add unnecessary scene, camera, or style details.
- Preserve the main subject, action, and scene.
'''


def _image_to_data_url(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(ext)
    if mime is None:
        raise ValueError(f"Unsupported image extension: {ext}")

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if text.lower().startswith("json"):
        text = text[4:].strip()
    return text


def _parse_json_response(text: str) -> Dict[str, Any]:
    cleaned = _clean_json_text(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON. Raw output:\n{cleaned}") from e


def _chat_completion_json(
    messages,
    model_name: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Empty response content from model.")
    return _parse_json_response(content)


def extract_reference_stable_id_summary(
    image_path: str,
    api_key: Optional[str] = None,
    model_name: str = "gpt-5.4",
    base_url: Optional[str] = None,
    extra_user_instruction: Optional[str] = None,
) -> Dict[str, Any]:
    image_data_url = _image_to_data_url(image_path)

    user_parts = [
        {"type": "text", "text": "Analyze this reference image and return the JSON object only."},
        {"type": "image_url", "image_url": {"url": image_data_url}},
    ]
    if extra_user_instruction:
        user_parts.append({"type": "text", "text": extra_user_instruction})

    messages = [
        {"role": "system", "content": REFERENCE_ID_SYSTEM_PROMPT},
        {"role": "user", "content": user_parts},
    ]
    return _chat_completion_json(
        messages=messages,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
    )


def adapt_prompt_with_reference_id(
    raw_prompt: str,
    stable_id_summary: Dict[str, Any],
    api_key: Optional[str] = None,
    model_name: str = "gpt-5.4",
    base_url: Optional[str] = None,
    extra_user_instruction: Optional[str] = None,
) -> Dict[str, Any]:
    user_text = (
        f"Raw prompt:\n{raw_prompt}\n\n"
        f"Reference stable ID summary:\n{json.dumps(stable_id_summary, ensure_ascii=False, indent=2)}\n"
    )
    if extra_user_instruction:
        user_text += f"\nAdditional instruction:\n{extra_user_instruction}\n"

    messages = [
        {"role": "system", "content": PROMPT_AUG_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    return _chat_completion_json(
        messages=messages,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
    )


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _write_text_file(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _find_prompt_txt_files(input_dir: str) -> List[str]:
    txt_files: List[str] = []
    for name in sorted(os.listdir(input_dir)):
        full_path = os.path.join(input_dir, name)
        if not os.path.isfile(full_path):
            continue
        if not name.lower().endswith(".txt"):
            continue
        if name.lower().endswith("_processed.txt"):
            continue
        txt_files.append(full_path)
    return txt_files


def process_directory(
    input_dir: str,
    api_key: Optional[str] = None,
    model_name: str = "gpt-5.4",
    base_url: Optional[str] = None,
    extract_instruction: Optional[str] = None,
    adapt_instruction: Optional[str] = None,
) -> Dict[str, Any]:
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")

    image_path = os.path.join(input_dir, "image.png")
    if not os.path.isfile(image_path):
        raise ValueError(f"Expected image file not found: {image_path}")

    prompt_txt_files = _find_prompt_txt_files(input_dir)
    if not prompt_txt_files:
        raise ValueError(
            f"No prompt .txt files found in directory: {input_dir} "
            f"(files ending with _processed.txt are ignored)"
        )

    stable_id_summary = extract_reference_stable_id_summary(
        image_path=image_path,
        api_key=api_key,
        model_name=model_name,
        base_url=base_url,
        extra_user_instruction=extract_instruction,
    )
    stable_id_json_path = os.path.join(input_dir, "stable_id_summary.json")
    save_json(stable_id_summary, stable_id_json_path)

    results: List[Dict[str, Any]] = []
    for txt_path in prompt_txt_files:
        raw_prompt = _read_text_file(txt_path)
        adapted = adapt_prompt_with_reference_id(
            raw_prompt=raw_prompt,
            stable_id_summary=stable_id_summary,
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            extra_user_instruction=adapt_instruction,
        )

        base_name = os.path.splitext(os.path.basename(txt_path))[0]
        prompt_aug_json_path = os.path.join(input_dir, f"{base_name}_prompt_aug.json")
        processed_txt_path = os.path.join(input_dir, f"{base_name}_processed.txt")

        save_json(adapted, prompt_aug_json_path)
        processed_prompt = str(adapted.get("processed_prompt", "")).strip()
        if not processed_prompt.strip():
            raise ValueError("The response contains an empty processed_prompt")
        _write_text_file(processed_txt_path, processed_prompt)

        results.append(
            {
                "input_txt": txt_path,
                "prompt_aug_json": prompt_aug_json_path,
                "processed_prompt_txt": processed_txt_path,
            }
        )

    return {
        "input_dir": input_dir,
        "image_path": image_path,
        "stable_id_json": stable_id_json_path,
        "num_prompts": len(prompt_txt_files),
        "results": results,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    process = subparsers.add_parser('process_dir', help='Optimize the prompt files in one case directory')
    process.add_argument('--input_dir', required=True)
    process.add_argument('--model_name', default=os.environ.get('OPENAI_CHAT_MODEL', 'gpt-5.4'))
    process.add_argument('--extract_instruction', default=None)
    process.add_argument('--adapt_instruction', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise SystemExit('Set OPENAI_API_KEY before running prompt optimization')
    try:
        result = process_directory(
            input_dir=args.input_dir, model_name=args.model_name, api_key=api_key,
            base_url=os.environ.get('OPENAI_BASE_URL'),
            extract_instruction=args.extract_instruction, adapt_instruction=args.adapt_instruction,
        )
    except (FileNotFoundError, NotADirectoryError, PermissionError) as exc:
        detail = os.path.basename(exc.filename) if isinstance(exc.filename, str) else ''
        raise SystemExit(f'{type(exc).__name__}: {detail}') from None
    except Exception as exc:
        status = getattr(exc, 'status_code', None)
        detail = f' (HTTP {status})' if isinstance(status, int) else ''
        raise SystemExit(f'Prompt optimization failed: {type(exc).__name__}{detail}') from None
    print(f"Processed {result['num_prompts']} prompt(s)")


if __name__ == '__main__':
    main()
