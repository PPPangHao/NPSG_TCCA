""" main.py
The main function of rPPG deep learning pipeline.

Usage examples:
    python main.py --config_file configs/train_configs/PURE_PURE_UBFC-rPPG_TSCAN_BASIC.yaml
"""
import argparse
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import get_config
from dataset import data_loader
from neural_methods import trainer
from unsupervised_methods.unsupervised_predictor import unsupervised_predict

# -------------------------
# Default deterministic seed
# -------------------------
RANDOM_SEED = 100


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic settings (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """
    Worker init function for DataLoader to improve reproducibility.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def add_args(parser):
    """Add CLI args (keeps original default config_file)."""
    parser.add_argument('--config_file', required=False,
                        default="configs/train_configs/PURE_PURE_UBFC-rPPG_TSCAN_BASIC.yaml",
                        type=str, help="Path to yaml config file.")
    return parser


def make_dataloader(loader_cls, name: str, data_path: str, config_data: dict, device: str,
                    batch_size: int, num_workers: int, shuffle: bool, generator: torch.Generator):
    """
    Create dataset instance and DataLoader.
      - loader_cls: dataset class (e.g., data_loader.UBFCrPPGLoader.UBFCrPPGLoader)
      - name: "train"/"valid"/"test"/"unsupervised"
    """
    if data_path is None or not str(data_path):
        raise ValueError(f"[{name}] data_path is not specified in config.")
    if not Path(data_path).exists():
        raise ValueError(f"[{name}] data_path does not exist: {data_path}")

    dataset = loader_cls(
        name=name,
        data_path=data_path,
        config_data=config_data,
        device=device
    )
    dl = DataLoader(
        dataset=dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=torch.cuda.is_available()
    )
    logging.info(f"Created DataLoader for {name}: batch_size={batch_size}, num_workers={num_workers}, shuffle={shuffle}")
    return dl


def train_and_test(config, data_loader_dict):
    """Trains and then tests the model using the trainer API."""
    if config.MODEL.NAME == "Tscan":
        trainer_cls = trainer.TscanTrainer.TscanTrainer
    else:
        raise ValueError('Your Model is Not Supported  Yet!')

    model_trainer = trainer_cls(config, data_loader_dict)
    logging.info("Start training...")
    model_trainer.train(data_loader_dict)
    logging.info("Training finished — running test...")
    model_trainer.test(data_loader_dict)


def test(config, data_loader_dict):
    """Runs test only."""
    if config.MODEL.NAME == "Tscan":
        trainer_cls = trainer.TscanTrainer.TscanTrainer
    else:
        raise ValueError('Your Model is Not Supported  Yet!')
    model_trainer = trainer_cls(config, data_loader_dict)
    logging.info("Start testing...")
    model_trainer.test(data_loader_dict)


def unsupervised_method_inference(config, data_loader):
    """Run unsupervised predictors listed in config."""
    if not config.UNSUPERVISED.METHOD:
        raise ValueError("Please set unsupervised method in yaml!")
    for unsupervised_method in config.UNSUPERVISED.METHOD:
        method = unsupervised_method.upper()
        logging.info(f"Running unsupervised method: {method}")
        if method in {"POS", "CHROM", "ICA", "GREEN", "LGI", "PBV", "OMIT"}:
            unsupervised_predict(config, data_loader, method)
        else:
            raise ValueError(f"Not supported unsupervised method: {unsupervised_method}")


def prepare_dataloaders(config):
    """
    Based on config.TOOLBOX_MODE prepare the needed dataloaders and return dict.
    Uses general_generator for val/test/unsupervised and train_generator for train.
    """
    # global generators to improve reproducibility across worker processes
    general_generator = torch.Generator()
    general_generator.manual_seed(RANDOM_SEED)
    train_generator = torch.Generator()
    train_generator.manual_seed(RANDOM_SEED)

    data_loader_dict = {}

    num_workers = getattr(config, "DATALOADER_NUM_WORKERS", None) or getattr(config, "NUM_WORKERS", 8)
    # ensure integer
    try:
        num_workers = int(num_workers)
    except Exception:
        num_workers = 8

    # Helper to map dataset name -> loader class. Extend here if you add more datasets.
    dataset_map = {
        "UBFC-rPPG": data_loader.UBFCrPPGLoader.UBFCrPPGLoader,
        # add other supported datasets map entries here if needed
    }

    # TRAIN & VALID creation for train_and_test
    if config.TOOLBOX_MODE == "train_and_test":
        # Train
        if config.TRAIN.DATA.DATASET not in dataset_map:
            raise ValueError(f"Unsupported TRAIN dataset: {config.TRAIN.DATA.DATASET}. Supported: {list(dataset_map.keys())}")
        train_loader_cls = dataset_map[config.TRAIN.DATA.DATASET]
        if config.TRAIN.DATA.DATA_PATH:
            data_loader_dict['train'] = make_dataloader(
                loader_cls=train_loader_cls,
                name="train",
                data_path=config.TRAIN.DATA.DATA_PATH,
                config_data=config.TRAIN.DATA,
                device=config.DEVICE,
                batch_size=config.TRAIN.BATCH_SIZE,
                num_workers=num_workers,
                shuffle=True,
                generator=train_generator
            )
        else:
            data_loader_dict['train'] = None

        # Valid
        if config.VALID.DATA.DATASET:
            if config.VALID.DATA.DATASET not in dataset_map:
                raise ValueError(f"Unsupported VALID dataset: {config.VALID.DATA.DATASET}. Supported: {list(dataset_map.keys())}")
            if config.TEST.USE_LAST_EPOCH:
                logging.info("TEST.USE_LAST_EPOCH is True: Skip creating validation dataloader.")
                data_loader_dict['valid'] = None
            else:
                if config.VALID.DATA.DATA_PATH:
                    valid_loader_cls = dataset_map[config.VALID.DATA.DATASET]
                    data_loader_dict["valid"] = make_dataloader(
                        loader_cls=valid_loader_cls,
                        name="valid",
                        data_path=config.VALID.DATA.DATA_PATH,
                        config_data=config.VALID.DATA,
                        device=config.DEVICE,
                        batch_size=config.TRAIN.BATCH_SIZE,
                        num_workers=num_workers,
                        shuffle=False,
                        generator=general_generator
                    )
                else:
                    data_loader_dict['valid'] = None
        else:
            data_loader_dict['valid'] = None

    # TEST dataloader (used in train_and_test or only_test)
    if config.TOOLBOX_MODE in {"train_and_test", "only_test"}:
        if config.TEST.DATA.DATASET not in dataset_map:
            raise ValueError(f"Unsupported TEST dataset: {config.TEST.DATA.DATASET}. Supported: {list(dataset_map.keys())}")
        test_loader_cls = dataset_map[config.TEST.DATA.DATASET]
        if config.TEST.DATA.DATA_PATH:
            data_loader_dict["test"] = make_dataloader(
                loader_cls=test_loader_cls,
                name="test",
                data_path=config.TEST.DATA.DATA_PATH,
                config_data=config.TEST.DATA,
                device=config.DEVICE,
                batch_size=getattr(config, "INFERENCE", {}).get("BATCH_SIZE", config.TRAIN.BATCH_SIZE),
                num_workers=num_workers,
                shuffle=False,
                generator=general_generator
            )
        else:
            data_loader_dict['test'] = None

    # Unsupervised mode
    if config.TOOLBOX_MODE == "unsupervised_method":
        if config.UNSUPERVISED.DATA.DATASET not in dataset_map:
            raise ValueError(f"Unsupported UNSUPERVISED dataset: {config.UNSUPERVISED.DATA.DATASET}. Supported: {list(dataset_map.keys())}")
        unsup_loader_cls = dataset_map[config.UNSUPERVISED.DATA.DATASET]
        if config.UNSUPERVISED.DATA.DATA_PATH:
            data_loader_dict["unsupervised"] = make_dataloader(
                loader_cls=unsup_loader_cls,
                name="unsupervised",
                data_path=config.UNSUPERVISED.DATA.DATA_PATH,
                config_data=config.UNSUPERVISED.DATA,
                device=config.DEVICE,
                batch_size=1,
                num_workers=num_workers,
                shuffle=False,
                generator=general_generator
            )
        else:
            data_loader_dict["unsupervised"] = None

    return data_loader_dict


def configure_logging(log_dir: str = None):
    """Configure logging to console and optionally to a file."""
    log_format = "%(asctime)s | %(levelname)5s | %(message)s"
    handlers = [logging.StreamHandler()]
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_file = Path(log_dir) / f"run_{int(time.time())}.log"
        handlers.append(logging.FileHandler(str(log_file)))
    logging.basicConfig(level=logging.INFO, format=log_format, handlers=handlers)


def main():
    # -----------------------
    # Parse args and config
    # -----------------------
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    # add options from trainer and data loader so CLI is compatible with original project
    parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
    parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()

    # Load configuration object (from project get_config)
    config = get_config(args)

    # Logging
    log_dir = getattr(config, "LOG_DIR", None) or getattr(config, "OUTPUT_DIR", None) or "./logs"
    configure_logging(log_dir)
    logging.info("Loaded configuration:")
    logging.info(config)

    # Seed & device
    set_seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() and getattr(config, "USE_CUDA", True) else "cpu"
    logging.info(f"Using device: {device}")

    # Prepare dataloaders based on toolbox mode
    try:
        data_loader_dict = prepare_dataloaders(config)
    except Exception as e:
        logging.exception("Failed to prepare dataloaders.")
        raise

    # Dispatch to requested mode
    try:
        if config.TOOLBOX_MODE == "train_and_test":
            train_and_test(config, data_loader_dict)
        elif config.TOOLBOX_MODE == "only_test":
            test(config, data_loader_dict)
        elif config.TOOLBOX_MODE == "unsupervised_method":
            # unsupervised_method expects an unsupervised dataloader dict key
            if "unsupervised" not in data_loader_dict or data_loader_dict["unsupervised"] is None:
                raise ValueError("Unsupervised dataloader is not prepared.")
            unsupervised_method_inference(config, data_loader_dict["unsupervised"])
        else:
            raise ValueError("TOOLBOX_MODE only support train_and_test, only_test or unsupervised_method.")
    except Exception:
        logging.exception("Error during execution.")
        raise


if __name__ == "__main__":
    main()
