import numpy as np
import pandas as pd
import random
import json
import logging
from pathlib import Path
from typing import Tuple, Set, List, Dict

logging.basicConfig(level=logging.INFO, format='%(message)s')

class DatasetPreprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.fractions = config["holdout_fractions"]
        random.seed(config["seed"])
        
        safe_name = config["dataset_name"].replace(" ", "_").lower()
        safe_version = config["version"].replace(".", "_")
        processed_folder_name = f"{safe_name}_v{safe_version}"
        
        self.raw_dir = Path(config["raw_data_dir"])
        self.processed_dir = Path(config["processed_data_dir"]) / processed_folder_name
        
        self._validate_directories()

    def _validate_directories(self):
        if self.processed_dir.exists():
            raise FileExistsError(
                f"Processed dataset directory already exists at: {self.processed_dir}\n"
                f"Please update the 'version' or 'dataset_name' in your config file."
            )
            
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Raw dataset directory not found: {self.raw_dir}")
            
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _warn_if_zero(self, intended_fraction: float, actual_count: int, name: str):
        if intended_fraction > 0 and actual_count == 0:
            logging.warning(f"[WARNING] Intended to hold out {intended_fraction*100}% of {name}, but math floored to 0.")

    def _extract_metadata(self, raw_data: dict) -> pd.DataFrame:
        df = pd.DataFrame({
            'rollout_id': raw_data['rollout_id'],
            'vehicle_dist_id': raw_data['vehicle_dist_id'],
            'vehicle_id': raw_data['vehicle_id'],
            'world_dist_id': raw_data['world_dist_id'],
            'world_id': raw_data['world_id']
        })
        meta_df = df.drop_duplicates().copy()
        meta_df['split'] = 'test'
        return meta_df

    def _apply_distribution_holdouts(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Set[int], Set[int]]:
        v_dists = df['vehicle_dist_id'].unique()
        num_v_holdout = int(len(v_dists) * self.fractions.get("vehicle_dists", 0.0))
        self._warn_if_zero(self.fractions.get("vehicle_dists", 0.0), num_v_holdout, "Vehicle Distributions")
        held_out_v_dists = set(random.sample(list(v_dists), num_v_holdout))

        w_dists = df['world_dist_id'].unique()
        num_w_holdout = int(len(w_dists) * self.fractions.get("world_dists", 0.0))
        self._warn_if_zero(self.fractions.get("world_dists", 0.0), num_w_holdout, "World Distributions")
        held_out_w_dists = set(random.sample(list(w_dists), num_w_holdout))

        train_eligible = df[
            (~df['vehicle_dist_id'].isin(held_out_v_dists)) & 
            (~df['world_dist_id'].isin(held_out_w_dists))
        ]
        return train_eligible, held_out_v_dists, held_out_w_dists

    def _apply_instance_holdouts(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Set[int], Set[int]]:
        vehicles = df['vehicle_id'].unique()
        num_v_holdout = int(len(vehicles) * self.fractions.get("in_dist_vehicles", 0.0))
        self._warn_if_zero(self.fractions.get("in_dist_vehicles", 0.0), num_v_holdout, "Specific Vehicles")
        held_out_vehicles = set(random.sample(list(vehicles), num_v_holdout))

        worlds = df['world_id'].unique()
        num_w_holdout = int(len(worlds) * self.fractions.get("in_dist_worlds", 0.0))
        self._warn_if_zero(self.fractions.get("in_dist_worlds", 0.0), num_w_holdout, "Specific Worlds")
        held_out_worlds = set(random.sample(list(worlds), num_w_holdout))

        train_eligible = df[
            (~df['vehicle_id'].isin(held_out_vehicles)) & 
            (~df['world_id'].isin(held_out_worlds))
        ]
        return train_eligible, held_out_vehicles, held_out_worlds

    def _apply_combo_holdouts(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[List[int]]]:
        unique_combos = df[['vehicle_id', 'world_id']].drop_duplicates()
        num_combos_holdout = int(len(unique_combos) * self.fractions.get("in_domain_vw_combos", 0.0))
        self._warn_if_zero(self.fractions.get("in_domain_vw_combos", 0.0), num_combos_holdout, "Vehicle-World Combos")
        
        held_out_combos_list = []
        if num_combos_holdout > 0:
            sampled_combos = unique_combos.sample(n=num_combos_holdout, random_state=self.config["seed"])
            merged = df.merge(sampled_combos, on=['vehicle_id', 'world_id'], how='left', indicator=True)
            df = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
            held_out_combos_list = sampled_combos.values.tolist()
            
        return df, held_out_combos_list

    def _apply_rollout_splits(self, meta_df: pd.DataFrame, eligible_df: pd.DataFrame) -> Tuple[pd.DataFrame, Set[int], Set[int], Set[int]]:
        eligible_rollout_ids = eligible_df['rollout_id'].tolist()
        random.shuffle(eligible_rollout_ids)

        total_eligible = len(eligible_rollout_ids)
        val_count = int(total_eligible * self.fractions.get("in_domain_rollouts_val", 0.0))
        test_count = int(total_eligible * self.fractions.get("in_domain_rollouts_test", 0.0))

        val_ids = set(eligible_rollout_ids[:val_count])
        test_ids = set(eligible_rollout_ids[val_count : val_count + test_count])
        train_ids = set(eligible_rollout_ids[val_count + test_count:])

        meta_df.loc[meta_df['rollout_id'].isin(val_ids), 'split'] = 'val'
        meta_df.loc[meta_df['rollout_id'].isin(train_ids), 'split'] = 'train'
        
        return meta_df, train_ids, val_ids, test_ids

    def _broadcast_and_save(self, raw_data: dict, meta_df: pd.DataFrame):
        logging.info("Broadcasting splits to timestep level...")
        split_mapping = meta_df.set_index('rollout_id')['split'].to_dict()
        raw_data['split'] = np.array([split_mapping[r_id] for r_id in raw_data['rollout_id']])
        
        out_file = self.processed_dir / "dataset_split.npz"
        np.savez_compressed(out_file, **raw_data)
        logging.info(f"Successfully saved processed dataset to {out_file}")

    def _calculate_and_save_norm_stats(self, raw_data: dict):
        logging.info("Calculating normalization statistics on the training set...")
        train_mask = raw_data['split'] == 'train'
        norm_stats = {}
        
        skip_keys = {'rollout_id', 'timestep', 'vehicle_dist_id', 'vehicle_id', 'world_dist_id', 'world_id', 'split'}
        
        for key, array in raw_data.items():
            if key not in skip_keys and np.issubdtype(array.dtype, np.number):
                train_slice = array[train_mask]
                if len(train_slice) > 0:
                    std_val = float(np.std(train_slice))
                    norm_stats[key] = {
                        "mean": float(np.mean(train_slice)),
                        "std": std_val if std_val > 1e-6 else 1.0 
                    }
    
        norm_file = self.processed_dir / "norm_stats.json"
        with open(norm_file, "w") as f:
            json.dump(norm_stats, f, indent=4)
        logging.info(f"Successfully saved normalization statistics to {norm_file}")

    def _generate_report(self, total_meta: int, train_ids: Set[int], val_ids: Set[int], test_ids: Set[int], 
                         held_out_v_dists: Set[int], held_out_w_dists: Set[int], 
                         held_out_vehicles: Set[int], held_out_worlds: Set[int], 
                         held_out_combos: List[List[int]]):
        
        report = {
            "description": self.config.get("description", ""),
            "raw_dataset_source": str(self.raw_dir),
            "configuration_seed": self.config["seed"],
            "total_rollouts": total_meta,
            "train_rollouts": len(train_ids),
            "holdout_fractions": self.fractions,
            "holdout_counts": {
                "vehicle_dists": len(held_out_v_dists),
                "world_dists": len(held_out_w_dists),
                "vehicles": len(held_out_vehicles),
                "worlds": len(held_out_worlds),
                "vw_combos": len(held_out_combos),
                "in_domain_val": len(val_ids),
                "in_domain_test": len(test_ids)
            },
            "held_out_entities": {
                "vehicle_distributions": list(map(int, held_out_v_dists)),
                "world_distributions": list(map(int, held_out_w_dists)),
                "specific_vehicles": list(map(int, held_out_vehicles)),
                "specific_worlds": list(map(int, held_out_worlds)),
                "specific_vw_combos": [[int(v), int(w)] for v, w in held_out_combos]
            },
        }
        
        with open(self.processed_dir / "split_report.json", "w") as f:
            json.dump(report, f, indent=4)
            
        logging.info("\n--- Preprocessing Report ---")
        logging.info(json.dumps(report, indent=4))

    def run(self):
        logging.info(f"Loading raw dataset from {self.raw_dir}...")
        raw_data = dict(np.load(self.raw_dir / "dataset.npz"))
        
        meta_df = self._extract_metadata(raw_data)
        
        train_eligible, h_v_dists, h_w_dists = self._apply_distribution_holdouts(meta_df)
        train_eligible, h_vehicles, h_worlds = self._apply_instance_holdouts(train_eligible)
        train_eligible, h_combos = self._apply_combo_holdouts(train_eligible)
        
        meta_df, train_ids, val_ids, test_ids = self._apply_rollout_splits(meta_df, train_eligible)

        self._broadcast_and_save(raw_data, meta_df)
        self._calculate_and_save_norm_stats(raw_data)
        
        self._generate_report(
            total_meta=len(meta_df),
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            held_out_v_dists=h_v_dists,
            held_out_w_dists=h_w_dists,
            held_out_vehicles=h_vehicles,
            held_out_worlds=h_worlds,
            held_out_combos=h_combos
        )