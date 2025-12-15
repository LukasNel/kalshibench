import os
from typing import Protocol, cast

import modal  # pyright: ignore[reportMissingImports]
kalshibench_volume = modal.Volume.from_name("kalshibench-volume", create_if_missing=True)

class _ModalRemoteFn(Protocol):
    def remote(self, *args: object, **kwargs: object) -> object: ...
OUTPUT_DIR = "/prediction-market-grpo"
app = modal.App(name="prediction-market-grpo")
volume = modal.Volume.from_name("prediction-market-grpo-volume", create_if_missing=True)
image = modal.Image.debian_slim().uv_pip_install("unsloth", "vllm", "datasets", "evaluate", "trl").uv_pip_install("wandb").add_local_python_source('training')
HOURS = 3600
@app.function(gpu="A100", image=image, secrets=[modal.Secret.from_dotenv(".env")], volumes={OUTPUT_DIR: volume}, timeout=23*HOURS)
def train_model():
    from training import main
    _result = cast(object, main(OUTPUT_DIR))

# --- Benchmark/test runner (runs uv sync + run_test.sh) ---
REPO_DIR = "/root/predictionmarket"


def _write_dotenv_from_env(path: str) -> None:
    """Materialize a .env file from injected env vars so run_test.sh can load it."""
    import os

    keys = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "TOGETHER_API_KEY",
        "GEMINI_API_KEY",
    ]
    lines: list[str] = []
    for k in keys:
        v = os.environ.get(k)
        if v:
            lines.append(f"{k}={v}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


test_image = (
    modal.Image.debian_slim()
    .apt_install("bash")
    .uv_sync()
    .add_local_file("kalshibench.py", os.path.join(REPO_DIR, "kalshibench.py"))
    .add_local_file("run_test.sh", os.path.join(REPO_DIR, "run_test.sh"))
    .add_local_file("README.md", os.path.join(REPO_DIR, "README.md"))
    .add_local_file("run_neurips_benchmark.py", os.path.join(REPO_DIR, "run_neurips_benchmark.py"))
)

OUTPUT_RESULTS_DIR = "/neurips_results"


@app.function(
    image=test_image,
    secrets=[modal.Secret.from_dotenv(".env")],
    timeout=2 * HOURS,
    volumes={OUTPUT_RESULTS_DIR: kalshibench_volume},
)
def run_neurips_test() -> None:
    import subprocess
    _write_dotenv_from_env(os.path.join(REPO_DIR, ".env"))
    _ = subprocess.run(["bash", "run_test.sh", OUTPUT_RESULTS_DIR], cwd=REPO_DIR, check=True)



@app.local_entrypoint()
def test():
    _ = cast(_ModalRemoteFn, run_neurips_test).remote()