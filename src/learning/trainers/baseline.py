import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm

class BaselineTrainer:
    def __init__(self, model, train_loader, val_loader, training_config, run_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run_dir = run_dir

        self.resolved_config = {
            "device": training_config.get("device", "cuda") if torch.cuda.is_available() else "cpu",
            "num_epochs": training_config.get("num_epochs", 100),
            "learning_rate": training_config.get("learning_rate", 1e-3),
            "checkpoint_freq": training_config.get("checkpoint_freq", 10),
            "clip_grad_norm": training_config.get("clip_grad_norm", 1.0)
        }
        
        self.device = torch.device(self.resolved_config["device"])
        self.epochs = self.resolved_config["num_epochs"]
        self.lr = self.resolved_config["learning_rate"]
        self.save_freq = self.resolved_config["checkpoint_freq"]
        self.clip_grad = self.resolved_config["clip_grad_norm"]
        
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train(self):
        train_log = {"epoch": [], "train_loss": [], "val_loss": []}
        best_val_loss = float('inf')
        
        logging.info(f"Starting training for {self.epochs} epochs on {self.device}...")
        
        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_loss = self._validate_epoch(epoch)
            
            train_log["epoch"].append(epoch)
            train_log["train_loss"].append(train_loss)
            train_log["val_loss"].append(val_loss)
            
            if epoch == 1 or epoch == self.epochs or epoch % 10 == 0:
                logging.info(f"Epoch [{epoch}/{self.epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, "best_model.pth")
                
            if epoch % self.save_freq == 0:
                self._save_checkpoint(epoch, f"checkpoint_epoch_{epoch}.pth")
                
        self._save_checkpoint(self.epochs, "final_model.pth")
        
        return {
            "execution_config": self.resolved_config,
            "final_metrics": {
                "best_val_loss": best_val_loss,
                "final_train_loss": train_log["train_loss"][-1],
                "final_val_loss": train_log["val_loss"][-1]
            },
            "history": train_log
        }

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.epochs} [Train]",
            leave=False,
            dynamic_ncols=True
        )

        for X_history, U_future, Y_future in progress:
            X = X_history.to(self.device)
            Y = Y_future.to(self.device)

            targets = Y[:, 0, :]

            self.optimizer.zero_grad(set_to_none=True)
            predictions = self.model(X)
            loss = self.criterion(predictions, targets)
            loss.backward()

            if self.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

            self.optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss

            progress.set_postfix(loss=f"{batch_loss:.6f}")

        return total_loss / len(self.train_loader)

    def _validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0

        progress = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch}/{self.epochs} [Val]",
            leave=False,
            dynamic_ncols=True
        )

        with torch.no_grad():
            for X_history, U_future, Y_future in progress:
                X = X_history.to(self.device)
                Y = Y_future.to(self.device)

                targets = Y[:, 0, :]
                predictions = self.model(X)
                loss = self.criterion(predictions, targets)

                batch_loss = loss.item()
                total_loss += batch_loss

                progress.set_postfix(loss=f"{batch_loss:.6f}")

        return total_loss / len(self.val_loader)

    def _save_checkpoint(self, epoch, filename):
        checkpoint_path = os.path.join((self.run_dir / "checkpoints"), filename)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)