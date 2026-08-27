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
        "FPLHyperspaceRender",
    ]:
        assert node in plugin.NODE_CLASS_MAPPINGS
        assert node in plugin.NODE_DISPLAY_NAME_MAPPINGS


def test_hyperspace_render_bin_command(monkeypatch, tmp_path):
    plugin = load_plugin(monkeypatch, tmp_path)
    monkeypatch.setenv("HYPERSPACE_RENDER_BIN", "/opt/hyperspace/render")
    monkeypatch.setenv("HYPERSPACE_DIR", "/opt/hyperspace")

    command, cwd = plugin.hyperspace_command("in.wav", "out.mp4", "scenes/composed.toml", 30, "9:16")

    assert cwd == "/opt/hyperspace"
    assert command == ["/opt/hyperspace/render", "in.wav", "out.mp4", "scenes/composed.toml", "30", "9:16"]


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
