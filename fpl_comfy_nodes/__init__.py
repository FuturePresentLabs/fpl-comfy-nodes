BIFROST_BASE_URL = "http://192.168.1.40:4040/v1"
FPL_WORKFLOW_PATH = "/opt/fpl/comfyui/workflows/fpl-openrouter-image.json"

import json
import base64
import io
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from uuid import uuid4

from aiohttp import web
from server import PromptServer

import numpy
import torch
from PIL import Image
try:
    import folder_paths
except Exception:
    folder_paths = None


CHAT_MODELS = [
    "qwen3-8b",
    "granite-4.1-8b",
    "granite-4.1-8b-abliterated",
    "lmstudio-community/Qwen3.8-27B-MLX-4bit",
    "gpt-oss-20b",
    "or/poolside/laguna-s-2.1",
    "or/stealth/ox-alpha",
    "or/meta/muse-spark-1.2",
    "or/meta/muse-spark-1.2-contributor",
    "or/meta/muse-glimmer-30b",
    "or/meta/muse-spark-1.1",
    "or/deepseek/deepseek-v4-pro",
    "or/z-ai/glm-5.2",
    "qwen36-35b-a3b",
    "qwen3.8-27b-obliterated",
]

IMAGE_MODELS = [
    "fpl/image",
    "agora/image",
    "or-img/black-forest-labs/flux.2-klein-4b",
    "or-img/black-forest-labs/flux.2-pro",
    "or-img/black-forest-labs/flux.2-flex",
    "or-img/black-forest-labs/flux.2-max",
]

VIDEO_MODELS = [
    "fpl/video",
    "or-video/bytedance/seedance-1-5-pro",
    "or-video/bytedance/seedance-2.0-fast",
    "or-video/bytedance/seedance-2.0",
    "or-video/alibaba/wan-2.6",
    "or-video/alibaba/wan-2.7",
    "or-video/kwaivgi/kling-v3.0-std",
    "or-video/kwaivgi/kling-v3.0-pro",
    "or-video/kwaivgi/kling-video-o1",
    "or-video/google/veo-3.1",
    "or-video/openai/sora-2-pro",
]

MUSIC_MODELS = [
    "fpl/music",
    "ace-step-v1.5",
    "fal/ace-step",
]


@PromptServer.instance.routes.get("/fpl/workflows/openrouter-image")
async def get_openrouter_image_workflow(request):
    path = Path(os.environ.get("FPL_COMFYUI_WORKFLOW_PATH", FPL_WORKFLOW_PATH))
    if not path.exists():
        return web.json_response({"error": f"Workflow not found: {path}"}, status=404)
    try:
        return web.json_response(json.loads(path.read_text()))
    except json.JSONDecodeError as exc:
        return web.json_response({"error": f"Invalid workflow JSON: {exc}"}, status=500)


def api_key():
    api_key = os.environ.get("BIFROST_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("BIFROST_API_KEY or OPENAI_API_KEY is not set")
    return api_key


def base_url():
    return os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or BIFROST_BASE_URL


def output_dir():
    if folder_paths is not None:
        return Path(folder_paths.get_output_directory())
    return Path(os.environ.get("COMFYUI_OUTPUT_DIR", "/opt/comfyui/output"))


def read_json(request, timeout):
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", "replace")
        raise RuntimeError(f"Bifrost returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Bifrost request failed: {exc}") from exc


def post_json(path, payload, timeout=600):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url().rstrip("/") + path,
        data=body,
        headers={
            "Authorization": "Bearer " + api_key(),
            "Content-Type": "application/json",
        },
        method="POST",
    )
    return read_json(request, timeout)


def get_json(path, timeout=60):
    request = urllib.request.Request(
        base_url().rstrip("/") + path,
        headers={"Authorization": "Bearer " + api_key()},
        method="GET",
    )
    return read_json(request, timeout)


def configured_models(defaults, include):
    api_key_value = os.environ.get("BIFROST_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key_value:
        return defaults

    request = urllib.request.Request(
        base_url().rstrip("/") + "/models",
        headers={"Authorization": "Bearer " + api_key_value},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            data = json.load(response)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return defaults

    models = []
    for item in data.get("data") or []:
        model_id = item.get("id") if isinstance(item, dict) else item
        if isinstance(model_id, str) and include(model_id, item if isinstance(item, dict) else {}):
            models.append(model_id)

    if not models:
        return defaults

    preferred = [model for model in defaults if model in models]
    extras = sorted(model for model in models if model not in defaults)
    return preferred + extras


def model_modality(item):
    profile = item.get("bifrost_profile") if isinstance(item, dict) else None
    return profile.get("modality") if isinstance(profile, dict) else None


def configured_chat_models():
    return configured_models(
        CHAT_MODELS,
        lambda model_id, item: model_modality(item) == "llm"
        or model_id.startswith(("or/", "kimi/"))
        or model_id in CHAT_MODELS,
    )


def configured_image_models():
    return configured_models(
        IMAGE_MODELS,
        lambda model_id, item: model_modality(item) == "image"
        or model_id in ("fpl/image", "agora/image")
        or model_id.startswith("or-img/"),
    )


def configured_video_models():
    return configured_models(
        VIDEO_MODELS,
        lambda model_id, item: model_modality(item) == "video"
        or model_id == "fpl/video"
        or model_id.startswith("or-video/"),
    )


def configured_music_models():
    return configured_models(
        MUSIC_MODELS,
        lambda model_id, item: model_modality(item) == "music"
        or model_id in ("fpl/music", "ace-step-v1.5")
        or model_id.startswith("fal/"),
    )


def save_bytes(kind, ext, data):
    target_dir = output_dir() / "fpl_bifrost"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{kind}_{int(time.time())}_{uuid4().hex[:8]}.{ext.lstrip('.')}"
    path.write_bytes(data)
    return str(path)


def save_path(kind, ext):
    target_dir = output_dir() / kind
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{kind}_{int(time.time())}_{uuid4().hex[:8]}.{ext.lstrip('.')}"


def download_url(url, timeout=600, headers=None):
    request = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read(), response.headers.get("content-type", "")
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", "replace")
        raise RuntimeError(f"Download returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Download failed: {exc}") from exc


def extension_for_content_type(content_type, fallback):
    content_type = (content_type or "").split(";", 1)[0].strip().lower()
    return {
        "video/mp4": "mp4",
        "video/webm": "webm",
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/mpeg": "mp3",
        "audio/mp3": "mp3",
        "audio/flac": "flac",
        "audio/ogg": "ogg",
    }.get(content_type, fallback)


def image_tensor_to_data_url(image):
    if image is None:
        return None
    if not isinstance(image, torch.Tensor):
        raise RuntimeError("Expected ComfyUI IMAGE tensor input")

    frame = image[0].detach().cpu().numpy()
    frame = numpy.clip(frame * 255.0, 0, 255).astype(numpy.uint8)
    pil_image = Image.fromarray(frame).convert("RGB")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_chat_messages(system, prompt, image=None):
    messages = []
    if system and system.strip():
        messages.append({"role": "system", "content": system})

    image_url = image_tensor_to_data_url(image)
    if image_url:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        )
    else:
        messages.append({"role": "user", "content": prompt})
    return messages


def chat_completion(model, prompt, system, temperature, max_tokens, image=None):
    data = post_json(
        "/chat/completions",
        {
            "model": model,
            "messages": build_chat_messages(system, prompt, image),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        },
    )

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("Bifrost response did not include choices")
    message = choices[0].get("message") or {}
    text = message.get("content") or choices[0].get("text")
    if text is None:
        raise RuntimeError("Bifrost response did not include text")
    return text


class FPLBifrostModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (configured_chat_models(), {"default": "or/poolside/laguna-s-2.1"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model",)
    FUNCTION = "select"
    CATEGORY = "FPL/Bifrost"

    def select(self, model):
        return (model,)


class FPLBifrostChatCompletion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (configured_chat_models(), {"default": "or/poolside/laguna-s-2.1"}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Write a concise image prompt for a product photo.",
                    },
                ),
                "system": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "You are a concise assistant.",
                    },
                ),
                "temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "complete"
    CATEGORY = "FPL/Bifrost"
    OUTPUT_NODE = True

    def complete(self, model, prompt, system, temperature, max_tokens, image=None):
        return (chat_completion(model, prompt, system, temperature, max_tokens, image),)


class FPLBifrostTextGeneration(FPLBifrostChatCompletion):
    CATEGORY = "FPL/Bifrost"


class FPLBifrostCaptionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (configured_chat_models(), {"default": "or/poolside/laguna-s-2.1"}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Write a concise caption for this media.",
                    },
                ),
                "system": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "You write concise captions for product and social media assets.",
                    },
                ),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 256, "min": 1, "max": 4096, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "caption"
    CATEGORY = "FPL/Bifrost"
    OUTPUT_NODE = True

    def caption(self, model, prompt, system, temperature, max_tokens, image=None, video_path=""):
        if video_path and video_path.strip():
            prompt = f"{prompt}\n\nVideo file path for downstream context: {video_path.strip()}"
        return (chat_completion(model, prompt, system, temperature, max_tokens, image),)


class FPLBifrostImageGeneration:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (configured_image_models(), {"default": "fpl/image"}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A clean product photo on a neutral background",
                    },
                ),
                "aspect_ratio": (
                    ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9", "auto"],
                    {"default": "1:1"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "metadata")
    FUNCTION = "generate"
    CATEGORY = "FPL/Bifrost"
    OUTPUT_NODE = True

    def generate(self, model, prompt, aspect_ratio):
        data = post_json(
            "/images/generations",
            {
                "model": model,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "n": 1,
            },
        )

        images = data.get("data") or []
        if not images:
            raise RuntimeError("Bifrost response did not include image data")
        first = images[0]
        if first.get("b64_json"):
            raw = base64.b64decode(first["b64_json"])
        elif first.get("url"):
            with urllib.request.urlopen(first["url"], timeout=120) as response:
                raw = response.read()
        else:
            raise RuntimeError("Bifrost image response did not include b64_json or url")

        image = Image.open(io.BytesIO(raw)).convert("RGB")
        array = numpy.asarray(image).astype(numpy.float32) / 255.0
        tensor = torch.from_numpy(array)[None,]
        metadata = json.dumps(
            {
                "model": model,
                "aspect_ratio": aspect_ratio,
                "width": image.width,
                "height": image.height,
            }
        )
        return (tensor, metadata)


class FPLBifrostVideoGeneration:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (configured_video_models(), {"default": "fpl/video"}),
                "prompt": ("STRING", {"multiline": True, "default": "A short product video"}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4"], {"default": "16:9"}),
                "resolution": (["720p", "1080p", "480p"], {"default": "720p"}),
                "duration": ("INT", {"default": 6, "min": 1, "max": 30, "step": 1}),
                "image_mode": (["reference", "first_frame", "last_frame"], {"default": "reference"}),
                "poll_interval_seconds": ("INT", {"default": 10, "min": 2, "max": 60, "step": 1}),
                "timeout_seconds": ("INT", {"default": 900, "min": 60, "max": 3600, "step": 30}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "reference_image_url": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "metadata")
    FUNCTION = "generate"
    CATEGORY = "FPL/Bifrost"
    OUTPUT_NODE = True

    def generate(
        self,
        model,
        prompt,
        aspect_ratio,
        resolution,
        duration,
        image_mode,
        poll_interval_seconds,
        timeout_seconds,
        reference_image=None,
        reference_image_url="",
    ):
        payload = {
            "model": model,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "duration": int(duration),
        }
        reference_url = image_tensor_to_data_url(reference_image)
        if not reference_url and reference_image_url and reference_image_url.strip():
            reference_url = reference_image_url.strip()
        if reference_url:
            image_ref = {"type": "image_url", "image_url": {"url": reference_url}}
            if image_mode == "reference":
                payload["input_references"] = [image_ref]
            else:
                image_ref["frame_type"] = image_mode
                payload["frame_images"] = [image_ref]

        job = post_json("/videos", payload, timeout=60)
        job_id = job.get("id") or job.get("generation_id")
        if not job_id:
            raise RuntimeError("Bifrost video response did not include a job id")

        deadline = time.monotonic() + int(timeout_seconds)
        status = job
        terminal = {"completed", "failed", "cancelled", "expired"}
        while str(status.get("status", "")).lower() not in terminal:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Video generation timed out waiting for {job_id}")
            time.sleep(int(poll_interval_seconds))
            query = urllib.parse.urlencode({"model": model})
            status = get_json(f"/videos/{job_id}?{query}", timeout=60)

        if str(status.get("status", "")).lower() != "completed":
            raise RuntimeError(f"Video generation {job_id} ended with status {status.get('status')}: {status}")

        urls = status.get("unsigned_urls") or []
        if urls:
            raw, content_type = download_url(urls[0], timeout=600)
        else:
            query = urllib.parse.urlencode({"model": model})
            raw, content_type = download_url(
                base_url().rstrip("/") + f"/videos/{job_id}/content?{query}",
                timeout=600,
                headers={"Authorization": "Bearer " + api_key()},
            )
        ext = extension_for_content_type(content_type, "mp4")
        path = save_bytes("video", ext, raw)
        metadata = json.dumps({"model": model, "job": status, "path": path})
        return (path, metadata)


class FPLBifrostMusicGeneration:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (configured_music_models(), {"default": "fpl/music"}),
                "prompt": ("STRING", {"multiline": True, "default": "lofi house with warm keys"}),
                "lyrics": ("STRING", {"multiline": True, "default": "[inst]"}),
                "duration": ("INT", {"default": 10, "min": 1, "max": 300, "step": 1}),
                "instrumental": ("BOOLEAN", {"default": True}),
                "inference_steps": ("INT", {"default": 27, "min": 1, "max": 200, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 30.0, "step": 0.5}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647, "step": 1}),
            },
            "optional": {
                "tags": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("audio_path", "metadata")
    FUNCTION = "generate"
    CATEGORY = "FPL/Bifrost"
    OUTPUT_NODE = True

    def generate(
        self,
        model,
        prompt,
        lyrics,
        duration,
        instrumental,
        inference_steps,
        guidance_scale,
        seed,
        tags="",
    ):
        payload = {
            "model": model,
            "prompt": prompt,
            "lyrics": lyrics,
            "duration": int(duration),
            "response_format": "b64_json",
            "instrumental": bool(instrumental),
            "inference_steps": int(inference_steps),
            "guidance_scale": float(guidance_scale),
        }
        if int(seed) >= 0:
            payload["seed"] = int(seed)
        if tags and tags.strip():
            payload["tags"] = tags.strip()

        data = post_json("/audio/generations", payload, timeout=1200)
        outputs = data.get("data") or []
        if not outputs:
            raise RuntimeError("Bifrost music response did not include audio data")
        first = outputs[0]
        if first.get("b64_json"):
            raw = base64.b64decode(first["b64_json"])
            ext = payload.get("audio_format") or "wav"
        elif first.get("url"):
            raw, content_type = download_url(first["url"], timeout=600)
            ext = extension_for_content_type(content_type, "wav")
        else:
            raise RuntimeError("Bifrost music response did not include b64_json or url")
        path = save_bytes("music", ext, raw)
        metadata = json.dumps({"model": model, "response": data, "path": path})
        return (path, metadata)


def hyperspace_command(input_path, output_path, scene_path, fps, resolution):
    values = {
        "input": str(input_path),
        "output": str(output_path),
        "scene": str(scene_path),
        "fps": str(int(fps)),
        "resolution": str(resolution),
    }
    command_template = os.environ.get("HYPERSPACE_RENDER_COMMAND", "").strip()
    if command_template:
        if "{" in command_template:
            return shlex.split(command_template.format(**values)), None
        command = shlex.split(command_template)
        return command + [values["input"], values["output"], values["scene"], values["fps"], values["resolution"]], None

    render_bin = os.environ.get("HYPERSPACE_RENDER_BIN", "").strip()
    if render_bin:
        return [render_bin, values["input"], values["output"], values["scene"], values["fps"], values["resolution"]], None

    hyperspace_dir = os.environ.get("HYPERSPACE_DIR", "").strip()
    if hyperspace_dir:
        return [
            "cargo",
            "run",
            "--release",
            "--features",
            "render",
            "--example",
            "render",
            "--",
            values["input"],
            values["output"],
            values["scene"],
            values["fps"],
            values["resolution"],
        ], hyperspace_dir

    raise RuntimeError("Set HYPERSPACE_RENDER_BIN, HYPERSPACE_RENDER_COMMAND, or HYPERSPACE_DIR")


class FPLHyperspaceRender:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_path": ("STRING", {"default": ""}),
                "scene_path": ("STRING", {"default": "scenes/composed.toml"}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 120, "step": 1}),
                "resolution": (["9:16", "16:9", "1:1", "4:5", "1280x720", "1080x1920"], {"default": "9:16"}),
                "timeout_seconds": ("INT", {"default": 1800, "min": 60, "max": 14400, "step": 30}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "metadata")
    FUNCTION = "render"
    CATEGORY = "FPL/Media"
    OUTPUT_NODE = True

    def render(self, audio_path, scene_path, fps, resolution, timeout_seconds):
        input_path = Path(str(audio_path).strip()).expanduser()
        if not input_path.exists():
            raise RuntimeError(f"Audio file does not exist: {input_path}")

        output_path = save_path("fpl_hyperspace", "mp4")
        command, cwd = hyperspace_command(input_path, output_path, scene_path, fps, resolution)
        completed = subprocess.run(
            command,
            cwd=cwd,
            timeout=int(timeout_seconds),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(f"Hyperspace render failed with exit {completed.returncode}: {detail[-4000:]}")
        if not output_path.exists():
            raise RuntimeError(f"Hyperspace render completed but did not create {output_path}")

        metadata = json.dumps(
            {
                "audio_path": str(input_path),
                "scene_path": scene_path,
                "fps": int(fps),
                "resolution": resolution,
                "output_path": str(output_path),
                "cwd": cwd,
            }
        )
        return (str(output_path), metadata)


NODE_CLASS_MAPPINGS = {
    "FPLBifrostModel": FPLBifrostModel,
    "FPLBifrostChatCompletion": FPLBifrostChatCompletion,
    "FPLBifrostTextGeneration": FPLBifrostTextGeneration,
    "FPLBifrostCaptionNode": FPLBifrostCaptionNode,
    "FPLBifrostImageGeneration": FPLBifrostImageGeneration,
    "FPLBifrostVideoGeneration": FPLBifrostVideoGeneration,
    "FPLBifrostMusicGeneration": FPLBifrostMusicGeneration,
    "FPLHyperspaceRender": FPLHyperspaceRender,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FPLBifrostModel": "FPL Bifrost Model",
    "FPLBifrostChatCompletion": "FPL Bifrost Chat Completion",
    "FPLBifrostTextGeneration": "FPL Bifrost Text Generation",
    "FPLBifrostCaptionNode": "FPL Bifrost Caption",
    "FPLBifrostImageGeneration": "FPL Bifrost Image Generation",
    "FPLBifrostVideoGeneration": "FPL Bifrost Video Generation",
    "FPLBifrostMusicGeneration": "FPL Bifrost Music Generation",
    "FPLHyperspaceRender": "FPL Hyperspace Render",
}

WEB_DIRECTORY = "web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
