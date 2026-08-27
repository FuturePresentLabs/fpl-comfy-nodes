# FPL ComfyUI Nodes

Custom ComfyUI nodes for Future Present Labs creative and asset workflows.

This repository owns ComfyUI plugin code only. Deployment config belongs in
`shared-infra`; durable orchestration, approvals, CRM updates, and publishing
belong in `elmers`.

## Nodes

- `FPLBifrostImageGeneration`
- `FPLBifrostVideoGeneration`
- `FPLBifrostTextGeneration`
- `FPLBifrostCaptionNode`
- `FPLBifrostMusicGeneration`
- `FPLHyperspaceRender`

## Runtime Configuration

Bifrost nodes use an OpenAI-compatible Bifrost endpoint:

```bash
OPENAI_BASE_URL=http://192.168.1.40:4040/v1
BIFROST_API_KEY=sk-bifrost-...
```

The Hyperspace node renders `audio_path -> mp4` and accepts one of:

```bash
HYPERSPACE_RENDER_BIN=/opt/fpl/hyperspace/render
HYPERSPACE_RENDER_COMMAND='/opt/fpl/hyperspace/render {input} {output} {scene} {fps} {resolution}'
HYPERSPACE_DIR=/opt/fpl/hyperspace
```

`HYPERSPACE_RENDER_BIN` is preferred for production. `HYPERSPACE_DIR` falls
back to `cargo run --release --features render --example render`.

## Development

```bash
python -m pytest
python -m py_compile fpl_comfy_nodes/__init__.py
```
