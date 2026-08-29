BIFROST_BASE_URL = "http://192.168.1.40:4040/v1"
FPL_WORKFLOW_PATH = "/opt/fpl/comfyui/workflows/fpl-openrouter-image.json"
PEXELS_BASE_URL = "https://api.pexels.com"

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

PEXELS_IMAGE_SIZES = ["large2x", "large", "original", "medium", "small", "portrait", "landscape", "tiny"]
PEXELS_VIDEO_QUALITIES = ["best", "uhd", "hd", "sd"]

ACTOR_HEADER = "X-FPL-Actor"
ACTOR_SIGNATURE_HEADER = "X-FPL-Actor-Signature"
DOWNLOAD_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)
FPL_DOWNLOAD_HOSTS = ("ai.fpl.dev", "bifrost.fpl.dev", "palisade.fpl.dev")
ACTOR_METADATA_KEYS = ("fpl_actor", "x-fpl-actor", "actor")
ACTOR_SIGNATURE_METADATA_KEYS = (
    "fpl_actor_signature",
    "x-fpl-actor-signature",
    "actor_signature",
)


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


def fpl_download_hosts():
    raw = os.environ.get("FPL_DOWNLOAD_REWRITE_HOSTS")
    if not raw:
        return set(FPL_DOWNLOAD_HOSTS)
    return {host.strip().lower() for host in raw.split(",") if host.strip()}


def direct_download_base_url():
    return os.environ.get("FPL_DOWNLOAD_BASE_URL") or os.environ.get("FPL_DIRECT_DOWNLOAD_BASE_URL") or base_url()


def rewrite_download_url(url):
    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return url

    host = (parsed.hostname or "").lower()
    if host not in fpl_download_hosts():
        return url

    direct = urllib.parse.urlparse(direct_download_base_url().rstrip("/"))
    if not direct.scheme or not direct.netloc:
        return url

    path = parsed.path or "/"
    return urllib.parse.urlunparse((direct.scheme, direct.netloc, path, "", parsed.query, parsed.fragment))


def hidden_actor_inputs():
    return {
        "fpl_actor": "fpl_actor",
        "fpl_actor_signature": "fpl_actor_signature",
        "extra_pnginfo": "EXTRA_PNGINFO",
    }


def actor_headers(extra_pnginfo=None, fpl_actor=None, fpl_actor_signature=None):
    actor = first_non_empty(fpl_actor)
    signature = first_non_empty(fpl_actor_signature)
    for source in metadata_sources(extra_pnginfo):
        actor = actor or first_named_value(source, ACTOR_METADATA_KEYS)
        signature = signature or first_named_value(source, ACTOR_SIGNATURE_METADATA_KEYS)

    if actor and signature:
        return {
            ACTOR_HEADER: actor,
            ACTOR_SIGNATURE_HEADER: signature,
        }
    if actor or signature:
        raise RuntimeError("Bifrost actor attribution metadata is incomplete")
    return {}


def first_non_empty(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def first_named_value(source, keys):
    if not isinstance(source, dict):
        return None
    for key in keys:
        value = first_non_empty(source.get(key))
        if value:
            return value
    return None


def metadata_sources(extra_pnginfo):
    if not isinstance(extra_pnginfo, dict):
        return []
    sources = [extra_pnginfo]
    fpl = extra_pnginfo.get("fpl")
    if isinstance(fpl, dict):
        sources.insert(0, fpl)
    actor = extra_pnginfo.get("actor")
    if isinstance(actor, dict):
        sources.insert(0, actor)
    return sources


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


def request_headers(extra_pnginfo=None, fpl_actor=None, fpl_actor_signature=None):
    return {
        "Authorization": "Bearer " + api_key(),
        **actor_headers(extra_pnginfo, fpl_actor, fpl_actor_signature),
    }


def post_json(path, payload, timeout=600, extra_pnginfo=None, fpl_actor=None, fpl_actor_signature=None):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url().rstrip("/") + path,
        data=body,
        headers={
            **request_headers(extra_pnginfo, fpl_actor, fpl_actor_signature),
            "Content-Type": "application/json",
        },
        method="POST",
    )
    return read_json(request, timeout)


def get_json(path, timeout=60, extra_pnginfo=None, fpl_actor=None, fpl_actor_signature=None):
    request = urllib.request.Request(
        base_url().rstrip("/") + path,
        headers=request_headers(extra_pnginfo, fpl_actor, fpl_actor_signature),
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
    url = rewrite_download_url(url)
    request_headers = {
        "User-Agent": DOWNLOAD_USER_AGENT,
        "Accept": "application/octet-stream, video/*, audio/*, image/*;q=0.9, */*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "close",
        **(headers or {}),
    }
    request = urllib.request.Request(url, headers=request_headers, method="GET")
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


def pexels_api_key():
    value = os.environ.get("PEXELS_API_KEY", "").strip()
    if not value:
        raise RuntimeError("PEXELS_API_KEY is not set")
    return value


def pexels_search_url(path, params):
    clean_params = {key: value for key, value in params.items() if value not in (None, "", "any")}
    return PEXELS_BASE_URL.rstrip("/") + path + "?" + urllib.parse.urlencode(clean_params)


def pexels_get_json(path, params, timeout=30):
    request = urllib.request.Request(
        pexels_search_url(path, params),
        headers={"Authorization": pexels_api_key()},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", "replace")
        raise RuntimeError(f"Pexels returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Pexels request failed: {exc}") from exc


def select_pexels_photo(response, result_index):
    photos = response.get("photos") or []
    if not photos:
        raise RuntimeError("Pexels returned no photos")
    index = int(result_index)
    if index < 0 or index >= len(photos):
        raise RuntimeError(f"Pexels photo result_index {index} is outside 0..{len(photos) - 1}")
    return photos[index]


def pexels_photo_src(photo, size):
    src = photo.get("src") or {}
    for key in [size, "large2x", "large", "original", "medium"]:
        url = src.get(key)
        if url:
            return url, key
    raise RuntimeError("Pexels photo did not include a downloadable src URL")


def select_pexels_video(response, result_index):
    videos = response.get("videos") or []
    if not videos:
        raise RuntimeError("Pexels returned no videos")
    index = int(result_index)
    if index < 0 or index >= len(videos):
        raise RuntimeError(f"Pexels video result_index {index} is outside 0..{len(videos) - 1}")
    return videos[index]


def pexels_video_file(video, quality):
    files = [
        item
        for item in video.get("video_files") or []
        if item.get("link") and str(item.get("file_type", "")).startswith("video/")
    ]
    if not files:
        raise RuntimeError("Pexels video did not include a downloadable video file")
    if quality != "best":
        exact = [item for item in files if item.get("quality") == quality]
        if exact:
            files = exact
    return max(files, key=lambda item: int(item.get("width") or 0) * int(item.get("height") or 0))


def pexels_license():
    return {
        "source": "Pexels",
        "license_url": "https://www.pexels.com/license/",
        "note": "Free for commercial use under the Pexels license; do not resell unmodified stock media.",
    }


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


def chat_completion(
    model,
    prompt,
    system,
    temperature,
    max_tokens,
    image=None,
    extra_pnginfo=None,
    fpl_actor=None,
    fpl_actor_signature=None,
):
    data = post_json(
        "/chat/completions",
        {
            "model": model,
            "messages": build_chat_messages(system, prompt, image),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        },
        extra_pnginfo=extra_pnginfo,
        fpl_actor=fpl_actor,
        fpl_actor_signature=fpl_actor_signature,
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
            "hidden": hidden_actor_inputs(),
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
            "hidden": hidden_actor_inputs(),
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "complete"
    CATEGORY = "FPL/Bifrost"
    OUTPUT_NODE = True

    def complete(
        self,
        model,
        prompt,
        system,
        temperature,
        max_tokens,
        image=None,
        extra_pnginfo=None,
        fpl_actor=None,
        fpl_actor_signature=None,
    ):
        return (
            chat_completion(
                model,
                prompt,
                system,
                temperature,
                max_tokens,
                image,
                extra_pnginfo,
                fpl_actor,
                fpl_actor_signature,
            ),
        )


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
            "hidden": hidden_actor_inputs(),
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "caption"
    CATEGORY = "FPL/Bifrost"
    OUTPUT_NODE = True

    def caption(
        self,
        model,
        prompt,
        system,
        temperature,
        max_tokens,
        image=None,
        video_path="",
        extra_pnginfo=None,
        fpl_actor=None,
        fpl_actor_signature=None,
    ):
        if video_path and video_path.strip():
            prompt = f"{prompt}\n\nVideo file path for downstream context: {video_path.strip()}"
        return (
            chat_completion(
                model,
                prompt,
                system,
                temperature,
                max_tokens,
                image,
                extra_pnginfo,
                fpl_actor,
                fpl_actor_signature,
            ),
        )


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

    def generate(
        self,
        model,
        prompt,
        aspect_ratio,
        extra_pnginfo=None,
        fpl_actor=None,
        fpl_actor_signature=None,
    ):
        data = post_json(
            "/images/generations",
            {
                "model": model,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "n": 1,
            },
            extra_pnginfo=extra_pnginfo,
            fpl_actor=fpl_actor,
            fpl_actor_signature=fpl_actor_signature,
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
            "hidden": hidden_actor_inputs(),
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
        extra_pnginfo=None,
        fpl_actor=None,
        fpl_actor_signature=None,
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

        job = post_json(
            "/videos",
            payload,
            timeout=60,
            extra_pnginfo=extra_pnginfo,
            fpl_actor=fpl_actor,
            fpl_actor_signature=fpl_actor_signature,
        )
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
            status = get_json(
                f"/videos/{job_id}?{query}",
                timeout=60,
                extra_pnginfo=extra_pnginfo,
                fpl_actor=fpl_actor,
                fpl_actor_signature=fpl_actor_signature,
            )

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
                headers=request_headers(extra_pnginfo, fpl_actor, fpl_actor_signature),
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
            "hidden": hidden_actor_inputs(),
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
        extra_pnginfo=None,
        fpl_actor=None,
        fpl_actor_signature=None,
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

        data = post_json(
            "/audio/generations",
            payload,
            timeout=1200,
            extra_pnginfo=extra_pnginfo,
            fpl_actor=fpl_actor,
            fpl_actor_signature=fpl_actor_signature,
        )
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


class FPLPexelsPhotoSearch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "query": ("STRING", {"default": "modular synthesizer", "multiline": False}),
                "orientation": (["any", "landscape", "portrait", "square"], {"default": "portrait"}),
                "size": (PEXELS_IMAGE_SIZES, {"default": "large2x"}),
                "per_page": ("INT", {"default": 10, "min": 1, "max": 80, "step": 1}),
                "page": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1}),
                "result_index": ("INT", {"default": 0, "min": 0, "max": 79, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_path", "metadata")
    FUNCTION = "search"
    CATEGORY = "FPL/Stock"
    OUTPUT_NODE = True

    def search(self, query, orientation, size, per_page, page, result_index):
        response = pexels_get_json(
            "/v1/search",
            {
                "query": query,
                "orientation": orientation,
                "per_page": int(per_page),
                "page": int(page),
            },
        )
        photo = select_pexels_photo(response, result_index)
        url, selected_size = pexels_photo_src(photo, size)
        raw, content_type = download_url(url, timeout=120)
        ext = extension_for_content_type(content_type, "jpg")
        path = save_bytes("pexels_photo", ext, raw)

        image = Image.open(io.BytesIO(raw)).convert("RGB")
        array = numpy.asarray(image).astype(numpy.float32) / 255.0
        tensor = torch.from_numpy(array)[None,]
        metadata = json.dumps(
            {
                "kind": "photo",
                "query": query,
                "orientation": orientation,
                "size": selected_size,
                "path": path,
                "pexels_id": photo.get("id"),
                "photographer": photo.get("photographer"),
                "photographer_url": photo.get("photographer_url"),
                "source_url": photo.get("url"),
                "download_url": url,
                "width": image.width,
                "height": image.height,
                "license": pexels_license(),
            }
        )
        return (tensor, path, metadata)


class FPLPexelsVideoSearch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "query": ("STRING", {"default": "synthesizer performance", "multiline": False}),
                "orientation": (["any", "landscape", "portrait", "square"], {"default": "portrait"}),
                "quality": (PEXELS_VIDEO_QUALITIES, {"default": "best"}),
                "per_page": ("INT", {"default": 10, "min": 1, "max": 80, "step": 1}),
                "page": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1}),
                "result_index": ("INT", {"default": 0, "min": 0, "max": 79, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "metadata")
    FUNCTION = "search"
    CATEGORY = "FPL/Stock"
    OUTPUT_NODE = True

    def search(self, query, orientation, quality, per_page, page, result_index):
        response = pexels_get_json(
            "/videos/search",
            {
                "query": query,
                "orientation": orientation,
                "per_page": int(per_page),
                "page": int(page),
            },
        )
        video = select_pexels_video(response, result_index)
        selected_file = pexels_video_file(video, quality)
        raw, content_type = download_url(selected_file["link"], timeout=600)
        ext = extension_for_content_type(content_type, "mp4")
        path = save_bytes("pexels_video", ext, raw)
        metadata = json.dumps(
            {
                "kind": "video",
                "query": query,
                "orientation": orientation,
                "quality": selected_file.get("quality"),
                "path": path,
                "pexels_id": video.get("id"),
                "user": (video.get("user") or {}).get("name"),
                "user_url": (video.get("user") or {}).get("url"),
                "source_url": video.get("url"),
                "download_url": selected_file.get("link"),
                "width": selected_file.get("width"),
                "height": selected_file.get("height"),
                "duration": video.get("duration"),
                "license": pexels_license(),
            }
        )
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
        cwd = os.environ.get("HYPERSPACE_DIR", "").strip() or None
        return [render_bin, values["input"], values["output"], values["scene"], values["fps"], values["resolution"]], cwd

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


def hyperspace_env():
    env = os.environ.copy()
    runtime_dir = env.get("XDG_RUNTIME_DIR", "").strip()
    if not runtime_dir or not Path(runtime_dir).is_dir():
        runtime_dir = "/tmp/fpl-hyperspace-runtime"
        path = Path(runtime_dir)
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o700)
        env["XDG_RUNTIME_DIR"] = runtime_dir
    return env


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
            env=hyperspace_env(),
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
    "FPLPexelsPhotoSearch": FPLPexelsPhotoSearch,
    "FPLPexelsVideoSearch": FPLPexelsVideoSearch,
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
    "FPLPexelsPhotoSearch": "FPL Pexels Photo Search",
    "FPLPexelsVideoSearch": "FPL Pexels Video Search",
    "FPLHyperspaceRender": "FPL Hyperspace Render",
}

WEB_DIRECTORY = "web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
