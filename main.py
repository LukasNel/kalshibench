from datetime import UTC, datetime
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
        "HF_TOKEN",  # For HuggingFace dataset upload
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
    .add_local_file("create_kalshibench.py", os.path.join(REPO_DIR, "create_kalshibench.py"))
    .add_local_file("run_test.sh", os.path.join(REPO_DIR, "run_test.sh"))
    .add_local_file("run_full.sh", os.path.join(REPO_DIR, "run_full.sh"))
    .add_local_file("README.md", os.path.join(REPO_DIR, "README.md"))
    .add_local_file("run_neurips_benchmark.py", os.path.join(REPO_DIR, "run_neurips_benchmark.py"))
)

DATASET_OUTPUT_DIR = "/neurips_results"
DATASET_FOLDER_NAME = "kalshibench_v2_" + datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
DATASET_OUTPUT_PATH = os.path.join(DATASET_OUTPUT_DIR, DATASET_FOLDER_NAME)


@app.function(
    image=test_image,
    secrets=[modal.Secret.from_dotenv(".env")],
    timeout=2 * HOURS,
    volumes={DATASET_OUTPUT_DIR: kalshibench_volume},
)
def run_neurips_test() -> None:
    import subprocess
    _write_dotenv_from_env(os.path.join(REPO_DIR, ".env"))
    _ = subprocess.run(["bash", "run_test.sh", DATASET_OUTPUT_PATH], cwd=REPO_DIR, check=True)


@app.function(
    image=test_image,
    secrets=[modal.Secret.from_dotenv(".env")],
    timeout=12 * HOURS,  # Full benchmark takes longer
    volumes={DATASET_OUTPUT_DIR: kalshibench_volume},
)
def run_neurips_full() -> None:
    """Run the full NeurIPS benchmark across all model tiers."""
    import subprocess
    _write_dotenv_from_env(os.path.join(REPO_DIR, ".env"))
    _ = subprocess.run(["bash", "run_full.sh", "--output-dir", DATASET_OUTPUT_PATH], cwd=REPO_DIR, check=True)


@app.function(
    image=test_image,
    secrets=[modal.Secret.from_dotenv(".env")],
    timeout=2 * HOURS,
    volumes={DATASET_OUTPUT_DIR: kalshibench_volume},
)
def create_kalshibench_dataset(
    output_path: str = DATASET_OUTPUT_DIR,
    upload_repo: str | None = "2084Collective/kalshibench-v2",
) -> None:
    """Create and optionally upload the KalshiBench dataset.
    
    Args:
        output_path: Local path to save the dataset
        upload_repo: HuggingFace repo ID to upload to (e.g., '2084Collective/kalshibench-v2')
    """
    import subprocess
    
    # Write .env for HuggingFace token if uploading
    _write_dotenv_from_env(os.path.join(REPO_DIR, ".env"))
    cmd = ["uv", "run", "python", "create_kalshibench.py", "--output", os.path.join(DATASET_OUTPUT_PATH, "dataset_creation")]
    
    if upload_repo:
        cmd.extend(["--upload", upload_repo])
    
    _ = subprocess.run(cmd, cwd=REPO_DIR, check=True)
    print(f"Dataset created at {output_path}")
    if upload_repo:
        print(f"Dataset uploaded to {upload_repo}")


# @app.local_entrypoint()
# def test():
    """Run test benchmark (quick validation)."""
    # _ = cast(_ModalRemoteFn, create_kalshibench_dataset).remote(
    #     output_path=DATASET_OUTPUT_DIR,
    # )
    # _ = cast(_ModalRemoteFn, run_neurips_test).remote()


@app.local_entrypoint()
def full():
    """Run full NeurIPS benchmark across all model tiers."""
    _ = cast(_ModalRemoteFn, create_kalshibench_dataset).remote(
        output_path=DATASET_OUTPUT_DIR,
    )
    _ = cast(_ModalRemoteFn, run_neurips_full).remote()

