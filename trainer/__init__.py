
from .sft_ic import Trainer as SFTICTrainer
from .teacher_forcing_ic import Trainer as TeacherForcingICTrainer
# from .distillation import Trainer as ScoreDistillationTrainer


__all__ = [
    "SFTICTrainer",
    "TeacherForcingICTrainer",
    # "ScoreDistillationTrainer",
]
