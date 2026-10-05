import os
import sys
import yaml

_CURRENT_RUN = None
_WANDB_AVAILABLE = False
_WANDB_MODULE = None

try:
    _orig_path = list(sys.path)
    sys.path = [p for p in sys.path if p not in ("", ".")]
    import wandb as _wb  # type: ignore
    if hasattr(_wb, "init") and callable(getattr(_wb, "init")):
        _WANDB_AVAILABLE = True
        _WANDB_MODULE = _wb
    sys.path = _orig_path
except Exception:
    _WANDB_AVAILABLE = False
    _WANDB_MODULE = None


def _load_project_name():
    config_path = "configs/experiment.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
            return cfg.get("logging", {}).get("wandb_project", "cs747-final-23dcs007")
    return "cs747-final-23dcs007"


def init_run(name, config=None, group=None, seed=None, mode="offline"):
    """
    Initialize a W&B logging run.
    Args:
        name (str): Explicit name for the run.
        config (dict, optional): Hyperparameters / configuration dict.
        group (str, optional): Group name for grouping runs.
        seed (int, optional): Random seed metadata.
        mode (str, optional): W&B mode ("online", "offline", "disabled"). Defaults to "offline".
    """
    global _CURRENT_RUN
    if not name:
        raise ValueError("Run name must be explicitly provided.")

    project_name = _load_project_name()

    run_config = config.copy() if config is not None else {}
    if seed is not None:
        run_config["seed"] = seed

    if _WANDB_AVAILABLE and _WANDB_MODULE is not None:
        _CURRENT_RUN = _WANDB_MODULE.init(
            project=project_name,
            name=name,
            config=run_config,
            group=group,
            mode=mode,
            reinit=True
        )
    else:
        _CURRENT_RUN = {
            "project": project_name,
            "name": name,
            "config": run_config,
            "group": group,
            "seed": seed,
            "mode": mode,
            "logged_metrics": []
        }
        print(f"[W&B Logger] Initialized run '{name}' (project='{project_name}', group='{group}', seed={seed}, mode='{mode}')")

    return _CURRENT_RUN


def log_metrics(metrics, step=None):
    """
    Log a dictionary of metrics.
    Args:
        metrics (dict): Key-value pairs of metrics.
        step (int, optional): Step number.
    """
    global _CURRENT_RUN
    if _CURRENT_RUN is None:
        raise RuntimeError("No active W&B run. Call init_run() before logging metrics.")

    if _WANDB_AVAILABLE and _WANDB_MODULE is not None and getattr(_WANDB_MODULE, "run", None) is not None:
        _WANDB_MODULE.log(metrics, step=step)
    else:
        entry = {"metrics": metrics, "step": step}
        _CURRENT_RUN["logged_metrics"].append(entry)
        print(f"[W&B Logger] Logged metrics at step {step}: {metrics}")


def finish_run():
    """
    Finish the current active W&B run.
    """
    global _CURRENT_RUN
    if _WANDB_AVAILABLE and _WANDB_MODULE is not None and getattr(_WANDB_MODULE, "run", None) is not None:
        _WANDB_MODULE.finish()
    elif _CURRENT_RUN is not None:
        print(f"[W&B Logger] Finished run '{_CURRENT_RUN['name']}'")

    _CURRENT_RUN = None


if __name__ == "__main__":
    # Minimal offline smoke test
    print("--- Running W&B Logger Offline Smoke Test ---")
    run = init_run(name="smoke_test_run", config={"lr": 0.001}, group="smoke_tests", seed=42, mode="offline")
    log_metrics({"loss": 0.123, "accuracy": 0.95}, step=1)
    finish_run()
    print("--- Smoke Test Completed ---")
