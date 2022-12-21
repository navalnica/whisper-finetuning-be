import logging
import sys

import torch

import transformers
from transformers import Seq2SeqTrainer


logger = logging.getLogger('custom_trainer')
logger.setLevel(logging.INFO)


class Seq2SeqTrainerCustomLinearScheduler(Seq2SeqTrainer):

    """
    Custom trainer to initialize Learning Rate Scheduler
    and define the learning rate in the end of a training.
    """

    @staticmethod
    def scheduler_n_steps_for_fixed_lr_in_end(lr_max, lr_end, num_train_steps, num_warmup_steps) -> int:
        assert lr_end < lr_max
        return num_warmup_steps + (num_train_steps - num_warmup_steps) * lr_max / (lr_max - lr_end)

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        use_custom_scheduler = False
        try:
            # if learning_rate_end was passed as an argument
            learning_rate_end = self.args.learning_rate_end
            use_custom_scheduler = True
            logger.info('TrainerCustomLinearScheduler.create_scheduler(). '
                        f'initializing custom linear scheduler using learning_rate_end={learning_rate_end}')
        except:
            logger.info('TrainerCustomLinearScheduler.create_scheduler(). '
                        'learning_rate_end was not set. fallback to a default behavior')

        if use_custom_scheduler is True:
            scheduler_num_steps = self.scheduler_n_steps_for_fixed_lr_in_end(
                lr_max=self.args.learning_rate,
                lr_end=learning_rate_end,
                num_train_steps=num_training_steps,
                num_warmup_steps=self.args.warmup_steps
            )

            self.lr_scheduler = transformers.get_scheduler(
                'linear', optimizer=optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=scheduler_num_steps
            )
            return self.lr_scheduler
        else:
            return super().create_scheduler(num_training_steps, optimizer)
