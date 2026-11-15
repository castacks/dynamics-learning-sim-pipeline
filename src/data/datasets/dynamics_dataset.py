import torch
import numpy as np
from torch.utils.data import Dataset

class VehicleDynamicsDataset(Dataset):
    """IMPORTANT: Expects data_dict to include a timestep column that begins at 0 for the start of each rollout"""
    def __init__(self, data_dict, len_history, num_future_steps, features):
        """Data Dict should contain only numeric keys.
        Remove any keys that never need to get passed to neural network (except for timestep & rollout_id)
        """
        self.len_history = len_history
        self.num_future_steps = num_future_steps
        
        state_keys = list(features.get("observable_state", []))
        act_keys = list(features.get("actions", []))
        priv_terrain_keys = list(features.get("privileged_terrain_params", []))
        feature_keys = state_keys + act_keys + priv_terrain_keys
        target_keys = list(features.get("targets", []))

        self.X = np.column_stack([data_dict[k] for k in feature_keys])
        self.U = np.column_stack([data_dict[k] for k in act_keys])
        self.Y = np.column_stack([data_dict[k] for k in target_keys])
        
        self.timesteps = np.array(data_dict['timestep'])
        rollout_ids = np.array(data_dict['rollout_id'])
        
        valid_mask = rollout_ids[:-self.num_future_steps] == rollout_ids[self.num_future_steps:]
        self.valid_indices = np.where(valid_mask)[0]
        
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        idx = self.valid_indices[idx]
        timestep = self.timesteps[idx]
        
        if timestep < self.len_history:
            X = self.X[(idx - timestep):(idx + 1), :]
            padding_needed = (self.len_history + 1) - len(X)
            X = np.pad(X, pad_width=((padding_needed, 0), (0, 0)), mode='edge')
        else:
            X = self.X[(idx - self.len_history):(idx + 1), :]
            
        U_future = self.U[(idx + 1):(idx + 1 + self.num_future_steps), :]
        Y = self.Y[(idx + 1):(idx + 1 + self.num_future_steps), :]

        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(U_future, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.float32)
        )