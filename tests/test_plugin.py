import importlib
import sys
import types


class _Routes:
    def get(self, _path):
        def decorator(func):
            return func

        return decorator


def install_comfy_stubs(monkeypatch, tmp_path):
    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=_Routes()))
    monkeypatch.setitem(sys.modules, "server", server)

    aiohttp = types.ModuleType("aiohttp")
    web = types.ModuleType("aiohttp.web")
    web.json_response = lambda *args, **kwargs: (args, kwargs)
    aiohttp.web = web
    monkeypatch.setitem(sys.modules, "aiohttp", aiohttp)
    monkeypatch.setitem(sys.modules, "aiohttp.web", web)

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_output_directory = lambda: str(tmp_path)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)

    torch = types.ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})
    monkeypatch.setitem(sys.modules, "torch", torch)

    numpy = types.ModuleType("numpy")
    numpy.uint8 = object()
    numpy.float32 = object()
    monkeypatch.setitem(sys.modules, "numpy", numpy)

    pil = types.ModuleType("PIL")
    image = types.ModuleType("PIL.Image")
    image.Image = type("Image", (), {})
    pil.Image = image
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", image)


def load_plugin(monkeypatch, tmp_path):
    install_comfy_stubs(monkeypatch, tmp_path)
    sys.modules.pop("fpl_comfy_nodes", None)
    return importlib.import_module("fpl_comfy_nodes")


def test_node_mappings_include_media_nodes(monkeypatch, tmp_path):
    plugin = load_plugin(monkeypatch, tmp_path)

    for node in [
        "FPLBifrostImageGeneration",
        "FPLBifrostVideoGeneration",
        "FPLBifrostTextGeneration",
        "FPLBifrostCaptionNode",
        "FPLBifrostMusicGeneration",
        "FPLPexelsPhotoSearch",
        "FPLPexelsVideoSearch",
        "FPLHyperspaceRender",
    ]:
        assert node in plugin.NODE_CLASS_MAPPINGS
        assert node in plugin.NODE_DISPLAY_NAME_MAPPINGS


def test_image_generation_declares_hidden_actor_inputs(monkeypatch, tmp_path):
    plugin = load_plugin(monkeypatch, tmp_path)

    hidden = plugin.FPLBifrostImageGeneration.INPUT_TYPES()["hidden"]

    assert hidden == {
        "fpl_actor": "fpl_actor",
        "fpl_actor_signature": "fpl_actor_signature",
        "extra_pnginfo": "EXTRA_PNGINFO",
    }


def test_hyperspace_render_bin_command(monkeypatch, tmp_path):
    plugin = load_plugin(monkeypatch, tmp_path)
    monkeypatch.setenv("HYPERSPACE_RENDER_BIN", "/opt/hyperspace/render")
    monkeypatch.setenv("HYPERSPACE_DIR", "/opt/hyperspace")

    command, cwd = plugin.hyperspace_command("in.wav", "out.mp4", "scenes/composed.toml", 30, "9:16")

    assert cwd == "/opt/hyperspace"
    assert command == ["/opt/hyperspace/render", "in.wav", "out.mp4", "scenes/composed.toml", "30", "9:16"]


def test_pexels_search_url_omits_any_orientation(monkeypatch, tmp_path):
    plugin = load_plugin(monkeypatch, tmp_path)

    url = plugin.pexels_search_url(
        "/v1/search",
        {"query": "modular synth", "orientation": "any", "per_page": 10, "page": 1},
    )

    assert url == "https://api.pexels.com/v1/search?query=modular+synth&per_page=10&page=1"


def test_pexels_photo_src_prefers_requested_size(monkeypatch, tmp_path):
    plugin = load_plugin(monkeypatch, tmp_path)

    url, size = plugin.pexels_photo_src(
        {"src": {"large": "https://example.test/large.jpg", "large2x": "https://example.test/large2x.jpg"}},
        "large",
    )

    assert url == "https://example.test/large.jpg"
    assert size == "large"


def test_pexels_video_file_picks_largest_quality_match(monkeypatch, tmp_path):
    plugin = load_plugin(monkeypatch, tmp_path)

    selected = plugin.pexels_video_file(
        {
            "video_files": [
                {
                    "quality": "hd",
                    "file_type": "video/mp4",
                    "width": 1280,
                    "height": 720,
                    "link": "https://example.test/720.mp4",
                },
                {
                    "quality": "hd",
                    "file_type": "video/mp4",
                    "width": 1920,
                    "height": 1080,
                    "link": "https://example.test/1080.mp4",
                },
                {
                    "quality": "sd",
                    "file_type": "video/mp4",
                    "width": 640,
                    "height": 360,
                    "link": "https://example.test/360.mp4",
                },
            ]
        },
        "hd",
    )

    assert selected["link"] == "https://example.test/1080.mp4"


def test_actor_headers_reads_prompt_bound_metadata(monkeypatch, tmp_path):
    plugin = load_plugin(monkeypatch, tmp_path)

    headers = plugin.actor_headers(
        {
            "fpl": {
                "fpl_actor": "payload",
                "fpl_actor_signature": "v1=signature",
            }
        }
    )

    assert headers == {
        "X-FPL-Actor": "payload",
        "X-FPL-Actor-Signature": "v1=signature",
    }


def test_actor_headers_rejects_incomplete_metadata(monkeypatch, tmp_path):
    plugin = load_plugin(monkeypatch, tmp_path)

    try:
        plugin.actor_headers({"fpl": {"fpl_actor": "payload"}})
    except RuntimeError as exc:
        assert "incomplete" in str(exc)
    else:
        raise AssertionError("expected incomplete actor metadata to fail")


def test_usage_context_headers_reads_signed_prompt_metadata(monkeypatch, tmp_path):
    plugin = load_plugin(monkeypatch, tmp_path)

    headers = plugin.usage_context_headers(
        {
            "fpl": {
                "fpl_usage_context": "context-payload",
                "fpl_usage_context_signature": "v1=context-signature",
            }
        }
    )

    assert headers == {
        "X-FPL-Usage-Context": "context-payload",
        "X-FPL-Usage-Context-Signature": "v1=context-signature",
    }


def test_usage_context_headers_rejects_incomplete_signed_metadata(monkeypatch, tmp_path):
    plugin = load_plugin(monkeypatch, tmp_path)

    try:
        plugin.usage_context_headers({"fpl": {"fpl_usage_context": "payload"}})
    except RuntimeError as exc:
        assert "incomplete" in str(exc)
    else:
        raise AssertionError("expected incomplete usage context metadata to fail")


def test_usage_context_headers_adds_direct_project_hints(monkeypatch, tmp_path):
    plugin = load_plugin(monkeypatch, tmp_path)
    monkeypatch.setenv("FPL_USAGE_PROJECT", "creative-render")

    headers = plugin.usage_context_headers(
        {
            "fpl": {
                "fpl_workflow": "workflow-42",
                "fpl_trace_id": "prompt-7",
                "fpl_parent_request_id": "conversation-3",
            }
        }
    )

    assert headers == {
        "X-FPL-Project": "creative-render",
        "X-FPL-Workflow": "workflow-42",
        "X-FPL-Trace-Id": "prompt-7",
        "X-FPL-Parent-Request-Id": "conversation-3",
    }


def test_download_url_rewrites_fpl_media_to_direct_bifrost(monkeypatch, tmp_path):
    plugin = load_plugin(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://192.168.1.40:4040/v1")

    assert (
        plugin.rewrite_download_url("https://ai.fpl.dev/v1/videos/job/content?model=fpl%2Fvideo")
        == "http://192.168.1.40:4040/v1/videos/job/content?model=fpl%2Fvideo"
    )


def test_download_url_does_not_rewrite_external_media(monkeypatch, tmp_path):
    plugin = load_plugin(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://192.168.1.40:4040/v1")

    assert plugin.rewrite_download_url("https://videos.pexels.com/video-files/test.mp4") == "https://videos.pexels.com/video-files/test.mp4"


def test_download_url_sends_browser_headers(monkeypatch, tmp_path):
    plugin = load_plugin(monkeypatch, tmp_path)
    captured = {}

    class Response:
        headers = {"content-type": "video/mp4"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b"video"

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(plugin.urllib.request, "urlopen", fake_urlopen)

    data, content_type = plugin.download_url("https://ai.fpl.dev/v1/media/test", headers={"Authorization": "Bearer token"})

    assert data == b"video"
    assert content_type == "video/mp4"
    assert captured["timeout"] == 600
    assert captured["request"].get_full_url() == "http://192.168.1.40:4040/v1/media/test"
    assert captured["request"].get_header("User-agent").startswith("Mozilla/5.0")
    assert captured["request"].get_header("Accept") == "application/octet-stream, video/*, audio/*, image/*;q=0.9, */*;q=0.8"
    assert captured["request"].get_header("Accept-language") == "en-US,en;q=0.9"
    assert captured["request"].get_header("Authorization") == "Bearer token"


def test_download_url_allows_caller_user_agent(monkeypatch, tmp_path):
    plugin = load_plugin(monkeypatch, tmp_path)
    captured = {}

    class Response:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b"ok"

    def fake_urlopen(request, timeout):
        captured["request"] = request
        return Response()

    monkeypatch.setattr(plugin.urllib.request, "urlopen", fake_urlopen)

    plugin.download_url("https://ai.fpl.dev/v1/media/test", headers={"User-Agent": "Custom/2.0"})

    assert captured["request"].get_header("User-agent") == "Custom/2.0"


def test_hyperspace_render_bin_command_without_hyperspace_dir(monkeypatch, tmp_path):
    plugin = load_plugin(monkeypatch, tmp_path)
    monkeypatch.setenv("HYPERSPACE_RENDER_BIN", "/opt/hyperspace/render")
    monkeypatch.delenv("HYPERSPACE_DIR", raising=False)

    command, cwd = plugin.hyperspace_command("in.wav", "out.mp4", "scenes/composed.toml", 30, "9:16")

    assert cwd is None
    assert command == ["/opt/hyperspace/render", "in.wav", "out.mp4", "scenes/composed.toml", "30", "9:16"]


def test_hyperspace_env_creates_runtime_dir(monkeypatch, tmp_path):
    plugin = load_plugin(monkeypatch, tmp_path)
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))

    env = plugin.hyperspace_env()

    assert env["XDG_RUNTIME_DIR"] == "/tmp/fpl-hyperspace-runtime"
    assert plugin.Path(env["XDG_RUNTIME_DIR"]).is_dir()


def test_hyperspace_command_template(monkeypatch, tmp_path):
    plugin = load_plugin(monkeypatch, tmp_path)
    monkeypatch.setenv(
        "HYPERSPACE_RENDER_COMMAND",
        "/bin/render --audio {input} --out {output} --scene {scene} --fps {fps} --res {resolution}",
    )

    command, cwd = plugin.hyperspace_command("a.wav", "b.mp4", "scene.toml", 24, "1:1")

    assert cwd is None
    assert command == [
        "/bin/render",
        "--audio",
        "a.wav",
        "--out",
        "b.mp4",
        "--scene",
        "scene.toml",
        "--fps",
        "24",
        "--res",
        "1:1",
    ]


def test_hyperspace_dir_falls_back_to_cargo(monkeypatch, tmp_path):
    plugin = load_plugin(monkeypatch, tmp_path)
    monkeypatch.delenv("HYPERSPACE_RENDER_BIN", raising=False)
    monkeypatch.delenv("HYPERSPACE_RENDER_COMMAND", raising=False)
    monkeypatch.setenv("HYPERSPACE_DIR", "/src/hyperspace")

    command, cwd = plugin.hyperspace_command("in.wav", "out.mp4", "scene.toml", 60, "16:9")

    assert cwd == "/src/hyperspace"
    assert command[:7] == ["cargo", "run", "--release", "--features", "render", "--example", "render"]
    assert command[-5:] == ["in.wav", "out.mp4", "scene.toml", "60", "16:9"]
