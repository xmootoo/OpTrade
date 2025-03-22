import os
import torch


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.

    Github:
        https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

    """

    def __init__(self, patience=7, verbose=True, delta=0, path="checkpoint.pt") -> None:
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbos = verbose
        self.counter = 0
        self.best_val_loss = float("inf")
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:

        if val_loss > self.best_val_loss:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module) -> None:

        if self.verbos:
            print(
                f"Validation loss decreased ({self.best_val_loss:.6f} --> {val_loss:.6f})."
            )
        path_dir = os.path.abspath(os.path.dirname(self.path))

        if not os.path.isdir(path_dir):
            os.makedirs(path_dir)

        print(f"Saving Model Weights in {self.path}...")
        torch.save(model.state_dict(), self.path)
        self.best_val_loss = val_loss
