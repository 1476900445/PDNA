from pathlib import Path

import torch

from pdna import PDNA, load_config
from pdna.data import CloudDialogueDataset, NegotiationTransitionDataset
from pdna.utils import resolve_device, seed_everything


def validate_dataset_files(
    llm_data_path: Path,
    negmas_data_path: Path,
) -> None:
    required_files = (
        llm_data_path / "train.jsonl",
        llm_data_path / "validation.jsonl",
        llm_data_path / "test.jsonl",
        negmas_data_path / "offline.jsonl",
        negmas_data_path / "online.jsonl",
    )

    missing_files = [
        path
        for path in required_files
        if not path.is_file()
    ]
    if missing_files:
        missing_text = "\n".join(
            f"  - {path}"
            for path in missing_files
        )
        raise FileNotFoundError(
            "PDNA cannot start because required dataset files are missing:\n"
            f"{missing_text}"
        )

    empty_files = [
        path
        for path in required_files
        if path.stat().st_size == 0
    ]
    if empty_files:
        empty_text = "\n".join(
            f"  - {path}"
            for path in empty_files
        )
        raise RuntimeError(
            "PDNA cannot start because required dataset files are empty:\n"
            f"{empty_text}"
        )


def main() -> None:
    project_root = Path(__file__).resolve().parent
    config_path = project_root / "configs" / "pdna.yaml"
    config = load_config(config_path)

    seed_everything(config.raw.seed)
    device = resolve_device(config.raw.device)

    llm_data_path = config.resolve_path("llm_data")
    negmas_data_path = config.resolve_path("negmas_data")

    validate_dataset_files(
        llm_data_path,
        negmas_data_path,
    )

    dialogue_datasets = {
        split: CloudDialogueDataset(llm_data_path, split, required=True)
        for split in ("train", "validation", "test")
    }
    offline_dataset = NegotiationTransitionDataset(
        negmas_data_path,
        "offline",
        state_dim=config.raw.fql.state_dim,
        action_dim=config.raw.fql.action_dim,
        required=True,
    )
    online_dataset = NegotiationTransitionDataset(
        negmas_data_path,
        "online",
        state_dim=config.raw.fql.state_dim,
        action_dim=config.raw.fql.action_dim,
        required=True,
    )

    empty_datasets = {
        "LLM_DATA/train.jsonl": len(dialogue_datasets["train"]),
        "LLM_DATA/validation.jsonl": len(dialogue_datasets["validation"]),
        "LLM_DATA/test.jsonl": len(dialogue_datasets["test"]),
        "NEGMAS_DATA/offline.jsonl": len(offline_dataset),
        "NEGMAS_DATA/online.jsonl": len(online_dataset),
    }
    empty_datasets = {
        name: size
        for name, size in empty_datasets.items()
        if size == 0
    }
    if empty_datasets:
        names = "\n".join(
            f"  - {name}"
            for name in empty_datasets
        )
        raise RuntimeError(
            "PDNA cannot start because no valid samples were found in:\n"
            f"{names}"
        )

    model = PDNA(config).to(device)
    model.eval()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("PDNA model initialized successfully.")
    print(f"Device: {device}")
    print(f"Model parameters: {parameter_count:,}")
    print(f"Trainable parameters: {trainable_parameter_count:,}")
    print(f"LLM_DATA: {llm_data_path}")
    print(f"NEGMAS_DATA: {negmas_data_path}")
    print(
        "Dialogue samples: "
        f"train={len(dialogue_datasets['train'])}, "
        f"validation={len(dialogue_datasets['validation'])}, "
        f"test={len(dialogue_datasets['test'])}"
    )
    print(
        "Reinforcement-learning transitions samples: "
        f"offline={len(offline_dataset)}, "
        f"online={len(online_dataset)}"
    )


if __name__ == "__main__":
    main()
