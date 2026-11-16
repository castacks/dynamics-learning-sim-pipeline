import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader

from data.datasets.dynamics_dataset import VehicleDynamicsDataset
from learning.models.baseline import BaselineSequenceModel
from learning.trainers.baseline import BaselineTrainer

class TrainingPipeline:
    def __init__(self, config: dict):
        self.config = config
        self._setup_workspace()
        self._setup_logging()

    def _setup_workspace(self):
        """Creates the unique runs/exp_name_vX directory."""
        exp_name = self.config.get("exp_name", "experiment").replace(" ", "_").lower()
        version = str(self.config.get("version", "1.0")).replace(".", "_")
        
        project_root = Path(__file__).resolve().parent.parent.parent
        self.run_dir = project_root / "runs" / f"{exp_name}_v{version}"
        
        # Prevent accidental overwrites of existing experiments
        if self.run_dir.exists():
            raise FileExistsError(
                f"Experiment directory already exists: {self.run_dir}\n"
                f"Please increment the 'version' in your config file to avoid overwriting weights."
            )
            
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)

    def _setup_logging(self):
        """Routes logging to both the terminal and a persistent file in the run directory."""
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers if pipeline is run sequentially in notebooks
        if logger.hasHandlers():
            logger.handlers.clear()
            
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File Handler
        file_handler = logging.FileHandler(self.run_dir / "training.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logging.info(f"Initialized training workspace at {self.run_dir}")

    def _get_requested_data_subset(self):
        """Loads the preprocessed dataset and applies training subset filters."""
        logging.info("Loading and slicing dataset...")
        dataset_path = self.config["dataset"]["path"]
        raw_data = dict(np.load(dataset_path))
        
        # Guardrail: Ensure subset logic is currently set to "all"
        subset_config = self.config["dataset"].get("training_subset", {})
        for key, value in subset_config.items():
            if value != "all":
                raise NotImplementedError(
                    f"Dynamic subset filtering for '{key}' is not yet implemented. "
                    f"Expected 'all', but got: {value}. Update your config."
                )

        # Since everything is 'all', just slice by the preprocessor's splits
        train_mask = raw_data['split'] == 'train'
        val_mask = raw_data['split'] == 'val'
        
        train_data = {k: v[train_mask] for k, v in raw_data.items()}
        val_data = {k: v[val_mask] for k, v in raw_data.items()}
        
        logging.info(f"Training subset loaded: {len(train_data['rollout_id'])} timesteps available.")
        logging.info(f"Validation subset loaded: {len(val_data['rollout_id'])} timesteps available.")
        
        return train_data, val_data

    def _calculate_norm_stats(self, train_data):
        """Calculates mean/std on the masked training set for the model to use."""
        logging.info("Calculating dynamic normalization statistics...")
        norm_stats = {}
        skip_keys = {'rollout_id', 'timestep', 'vehicle_dist_id', 'vehicle_id', 'world_dist_id', 'world_id', 'split'}
        
        for key, array in train_data.items():
            if key not in skip_keys and np.issubdtype(array.dtype, np.number):
                if len(array) > 0:
                    norm_stats[key] = {
                        "mean": float(np.mean(array)),
                        "std": float(np.std(array))
                    }
        return norm_stats
        
    def _save_norm_stats(self, norm_stats: dict):
        """Saves the computed normalization statistics to the experiment directory."""
        norm_file = self.run_dir / "norm_stats.json"
        with open(norm_file, "w") as f:
            json.dump(norm_stats, f, indent=4)
        logging.info(f"Saved normalization statistics to {norm_file}")  

    def _normalize(self, data_dict: dict, norm_stats: dict):
        """Applies zero-mean, unit-variance normalization using computed stats.
        TODO: Move to a utils file"""
        logging.info("Applying normalization to dataset...")
        norm_data = {}
        
        for key, array in data_dict.items():
            if key in norm_stats:
                mean = norm_stats[key]["mean"]
                std = norm_stats[key]["std"]
                std = std if std > 1e-6 else 1e-6 # Prevent div by 0
                norm_data[key] = (array - mean) / std
            else:
                # Leave metadata keys (timestep, rollout_id, etc.) untouched
                norm_data[key] = array.copy()
                
        return norm_data
    
    def _document_pipeline(self, log: dict):
        """Aggregates config, execution truths, and plots into the final model card."""
        logging.info("Generating learning curves and saving model card...")
        
        # 1. Build and save the complete model card using self.config
        model_card = self.config.copy()
        model_card["creation_timestamp"] = datetime.now().strftime("%m_%d_%Y__%I_%M_%p_EST")
        model_card["training_execution"] = log["execution_config"]
        model_card["final_metrics"] = log["final_metrics"]
        
        card_path = self.run_dir / "model_card.json"
        with open(card_path, "w") as f:
            json.dump(model_card, f, indent=4)
            
        # 2. Plot Learning Curves
        epochs = log["history"]["epoch"]
        train_loss = log["history"]["train_loss"]
        val_loss = log["history"]["val_loss"]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, label='Train Loss', color='blue')
        plt.plot(epochs, val_loss, label='Validation Loss', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.title('Training and Validation Learning Curves')
        plt.legend()
        plt.grid(True)
        
        plot_path = self.run_dir / "learning_curves.png"
        plt.savefig(plot_path)
        plt.close()
        
        logging.info(f"Saved model card and learning curves to {self.run_dir}")

    def run(self):
        """Executes the pipeline orchestration and hands off to the Trainer."""
        train_data, val_data = self._get_requested_data_subset()
        norm_stats = self._calculate_norm_stats(train_data)
        norm_train_data = self._normalize(train_data, norm_stats) # TODO: move this to a util
        norm_val_data = self._normalize(val_data, norm_stats)
        self._save_norm_stats(norm_stats)

        logging.info("Initializing PyTorch Datasets and DataLoaders...")
        train_dataset = VehicleDynamicsDataset(
            data_dict=norm_train_data, 
            len_history=self.config["dataset"]["history_steps"],
            num_future_steps=self.config["dataset"]["future_steps"],
            features=self.config["dataset"]["features_schema"]
        )
        val_dataset = VehicleDynamicsDataset(
            data_dict=norm_val_data, 
            len_history=self.config["dataset"]["history_steps"],
            num_future_steps=self.config["dataset"]["future_steps"],
            features=self.config["dataset"]["features_schema"]
        )
        
        batch_size = self.config["training_params"].get("batch_size", 256)
        num_workers = self.config["training_params"].get("num_workers", 4)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        )
        val_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True if num_workers > 0 else False,
                prefetch_factor=2 if num_workers > 0 else None
            )
        logging.info("Initializing Model Architecture...")
        
        # TODO: In future create a model factory w/create_model() method in models/ dir
        model = BaselineSequenceModel(
            architecture_params=self.config["model"],
            features_schema=self.config["dataset"]["features_schema"], 
            norm_stats=norm_stats
        )
        logging.info("Handing off to Trainer...")
        # TODO: In the future instantiate this as a base class
        trainer = BaselineTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            training_config=self.config["training_params"],
            run_dir=self.run_dir
        )
        
        log = trainer.train()
        self._document_pipeline(log)
        logging.info("Pipeline complete")
