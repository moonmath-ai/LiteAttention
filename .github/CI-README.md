# CI requirements

This document describes the requirements for self-hosted runners used by the GitHub Actions workflows in this repository.

## Workflows overview

| Workflow        | Purpose                                      | Runner labels used      |
|----------------|----------------------------------------------|--------------------------|
| **Release**    | Tag-based release: videos, draft release, wheels, PyPI | `self-hosted`, `Linux`, `gpu` |
| **_build**     | Build CUDA wheels in Docker (called by Release / Manual Build) | `self-hosted`            |
| **Manual Build** | Manually trigger a single wheel build       | Uses _build              |

## Runner requirements

### 1. Self-hosted runners

- Register at least **two** runner types (or one runner with multiple labels):
  - **`self-hosted` + `Linux`** — for jobs that need `gh`, Docker, and general Linux (e.g. `get_release_tag`, `create_draft_release`, `build_wheels`).
  - **`self-hosted` + `gpu`** — for jobs that run GPU workloads in Docker (e.g. video generation in Release).
- All runners must be **Linux** (x86_64 for wheel builds).

### 2. Docker

- **Docker** must be installed and the runner user must be able to run `docker` (e.g. in the `docker` group).
- Required for:
  - **Wheel builds** (`_build.yml`): jobs run inside `nvidia/cuda:*` containers.
  - **Release**: building and running the video-generation image with `docker build` and `docker run --gpus all`.

### 3. NVIDIA Container Toolkit (Docker + GPU)

- **NVIDIA Container Toolkit** must be installed so that containers can use the host GPUs.
- After installation, `docker run --gpus all` (or `--runtime=nvidia`) must work and see the same GPUs as the host.
- Used by:
  - Wheel build jobs: `container: image: nvidia/cuda:...` with `options: --gpus all`.
  - Video generation: `docker run --gpus all ...`.

### 4. GitHub CLI (`gh`)

- **GitHub CLI (`gh`)** must be installed and authenticated on the runner for:
  - **Release workflow**: `gh release create`, `gh release upload` (create draft release, upload assets and wheels).
- The `_build.yml` workflow installs `gh` **inside** the build container for uploading wheels to the release; the **host** runner still needs `gh` for the Release job steps that run on the host (e.g. `create_draft_release`).
- Ensure `gh auth status` succeeds (e.g. via `GITHUB_TOKEN` in the job env or a configured token on the runner).

### 5. Multi-GPU / GPU

- **Video generation** (Release) runs with `--gpus all` and may use multiple GPUs (NCCL env vars are set for multi-GPU).
- **Wheel builds** run in a single CUDA container with `--gpus all`; one GPU is enough for compilation, but the host must expose at least one GPU to the container.
- Recommended: at least **one** GPU for wheel builds; for video generation, multiple GPUs can improve throughput.

### 6. Optional but recommended

- **Disk space**: wheel builds and Docker images (CUDA images, video model images) are large; ensure sufficient free space (e.g. tens of GB).
- **Network**: access to GitHub, PyTorch index, and (for video generation) Hugging Face or other model/data sources.
- **Secrets / vars**: workflows expect repo secrets (e.g. `PYPI_API_TOKEN`, `CI_PULL_ID`, `CI_PULL_SECRET`) and optionally repo variables (e.g. `VIDEOS_PER_PROMPT`) as configured in the workflow files.

## Quick checklist

- [ ] Self-hosted runner(s) registered with labels `Linux` and/or `gpu` as required by the workflows.
- [ ] Docker installed; runner user in `docker` group.
- [ ] NVIDIA Container Toolkit installed; `docker run --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi` succeeds.
- [ ] GitHub CLI (`gh`) installed on the host; authentication works for the repo/org.
- [ ] At least one GPU available to Docker for wheel builds and video generation.
