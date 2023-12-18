import logging
import os
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class SimpleCallback(BaseCallback):
    def __init__(
            self,
            verbose: int = 0,
            model_save_path: str = None,
            model_save_freq: int = 100,
            rollout=0,
    ):
        super().__init__(verbose)
        self.model_save_freq = model_save_freq
        self.model_save_path = model_save_path

        # Create folder if needed
        if self.model_save_path is not None:
            os.makedirs(self.model_save_path, exist_ok=True)
        else:
            assert (
                    self.model_save_freq == 0
            ), "to use the `model_save_freq` you have to set the `model_save_path` parameter"

        self.roll_out = rollout

    def _on_rollout_end(self) -> None:
        if self.model_save_freq > 0:
            if self.model_save_path is not None:
                if self.roll_out % self.model_save_freq == 0:
                    self.save_model()
        self.roll_out += 1


    def _on_training_end(self) -> None:
        if self.model_save_path is not None:
            self.save_model()

    def save_model(self) -> None:
        path = os.path.join(self.model_save_path, f"model_{self.roll_out}")
        self.model.save(path)
        print(f"Saving model checkpoint to {path}")
        # if self.verbose > 1:
        #     logger.info("Saving model checkpoint to " + path)

    def _on_step(self) -> bool:
        """
        :return: If the callback returns False, training is aborted early.
        """
        return True

