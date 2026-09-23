import torch
from torch.utils.data import Dataset
import pickle
import numpy as np

# class MetaWorldExpertDataset(Dataset):
#     def __init__(self, pkl_path):
#         with open(pkl_path, "rb") as f:
#             self.images_list, self.observations_list, self.actions_list, self.rewards_list = pickle.load(f)

#         # 将所有 episode 拼接在一起
#         self.images = np.concatenate(self.images_list, axis=0)         # [N, 9, 224, 224]
#         self.observations = np.concatenate(self.observations_list, 0)  # [N, obs_dim]
#         self.actions = np.concatenate(self.actions_list, 0)            # [N, act_dim]

#         print(f"Loaded dataset: {self.images.shape}, {self.observations.shape}, {self.actions.shape}") #(4375, 9, 224, 224), (4375, 39), (4375, 4)  exp: 25 × 175 = 4375

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         img = torch.tensor(self.images[idx], dtype=torch.float32) / 255.0
#         obs = torch.tensor(self.observations[idx], dtype=torch.float32)
#         act = torch.tensor(self.actions[idx], dtype=torch.float32)
#         return img, obs, act
class MetaWorldExpertDataset(Dataset):
    def __init__(self, pkl_path, max_episodes=5):
        with open(pkl_path, "rb") as f:
            self.images_list, self.observations_list, self.actions_list, self.rewards_list = pickle.load(f)

        # 只保留前 max_episodes 个 episode
        self.images_list = self.images_list[:max_episodes]
        self.observations_list = self.observations_list[:max_episodes]
        self.actions_list = self.actions_list[:max_episodes]
        self.rewards_list = self.rewards_list[:max_episodes]

        # 将选定的 episode 拼接
        self.images = np.concatenate(self.images_list, axis=0)
        self.observations = np.concatenate(self.observations_list, axis=0)
        self.actions = np.concatenate(self.actions_list, axis=0)
        state_t  = self.observations[:,:18]
        state_t1 = self.observations[:,18:36]
        proprio_t  = state_t[:,:4]
        proprio_t1 = state_t1[:,:4]
        self.observations=np.concatenate([proprio_t, proprio_t1], axis=1)# numpy

        print(f"Loaded dataset (first {max_episodes} episodes): {self.images.shape}, {self.observations.shape}, {self.actions.shape}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx], dtype=torch.float32) / 255.0
        obs = torch.tensor(self.observations[idx], dtype=torch.float32)
        act = torch.tensor(self.actions[idx], dtype=torch.float32)
        return img, obs, act
