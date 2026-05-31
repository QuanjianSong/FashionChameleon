import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from trainer import SFTICTrainer
from trainer import TeacherForcingICTrainer
from trainer import GradientReweightedScoreDistillationTrainer

from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="outputs", help="Path to the directory to save logs")
    args = parser.parse_args()

    config = OmegaConf.merge(
        OmegaConf.load("configs/default_config.yaml"),
        OmegaConf.load(args.config_path),
        OmegaConf.create(vars(args))
    )

    return config


def main():
    os.umask(0o000)
    config = parse_args()

    if config.trainer == 'sft_ic':
        trainer = SFTICTrainer(config)
    elif config.trainer == 'teacher_forcing_ic':
        trainer = TeacherForcingICTrainer(config)
    elif config.trainer == 'gradient_reweighted_score_distillation':
        trainer = GradientReweightedScoreDistillationTrainer(config)
    else:
        raise NotImplementedError

    trainer.train()
        

if __name__ == "__main__":
    main()
