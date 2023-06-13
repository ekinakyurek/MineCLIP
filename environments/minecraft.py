from collections import deque
import math
from typing import Any, Callable, List, Optional, Tuple, Union

import gym
from minedojo.sim import spaces as spaces
from minedojo.sim.wrappers import ARNNWrapper
import numpy as np
from tianshou.env.venvs import SubprocVectorEnv
from tianshou.data import ReplayBuffer, Batch
from mineclip.utils import any_concat

import torch


class FlattenedMotionCamera(gym.ActionWrapper):
    def __init__(
        self,
        sim: ARNNWrapper,
        cam_interval: Union[int, float] = 15,
        cam_yaw_range=(-60, 60),
        cam_pitch_range=(-60, 60),
    ):
        assert isinstance(
            sim, ARNNWrapper
        ), f"Please mount this wrapper on the top of `ARNNWrapper`"
        super().__init__(sim)
        self._cam_yaw_range = cam_yaw_range
        self._cam_pitch_range = cam_pitch_range
        cam_yaw_range = abs(cam_yaw_range[0] - cam_yaw_range[1])
        cam_pitch_range = abs(cam_pitch_range[0] - cam_pitch_range[1])
        n_pitch_bins = math.ceil(cam_pitch_range / cam_interval) + 1
        n_yaw_bins = math.ceil(cam_yaw_range / cam_interval) + 1

        self.action_space = spaces.MultiDiscrete(
            [
                # ------ flattened motion + camera ------
                # last 6 indices are for motions
                # use "no-op" in the cartesian product of yaw and pitch as the "no-op" for this head
                # in first n_pitch_bins * n_yaw_bins indices, first increase yaw then increase pitch
                n_pitch_bins * n_yaw_bins + 6,
                # ------ functional actions, same as ARNNWrapper ------
                self.env.action_space.nvec[5],
                # ------ craft argument, same as ARNNWrapper ------
                self.env.action_space.nvec[6],
                # ------arg for "equip", "place", and "destroy" ------
                self.env.action_space.nvec[7],
            ],
            noop_vec=[
                n_yaw_bins * (n_pitch_bins - 1) // 2 + (n_yaw_bins - 1) // 2,
                0,
                0,
                0,
            ],
        )
        self._cam_interval = cam_interval
        self._arnn_cam_interval = self.env.cam_interval
        self._n_cam_cartesian = n_pitch_bins * n_yaw_bins
        self._n_yaw_bins = n_yaw_bins

    def action(self, action: list[int]):
        """
        Trimmed action to ARNN action
        """
        # print("action: ", action)
        assert self.action_space.contains(action)
        noop = self.env.action_space.no_op()

        # ------ parse flattened motions and camera ------
        # camera
        if action[0] < self._n_cam_cartesian:
            pitch_idx, yaw_idx = (
                action[0] // self._n_yaw_bins,
                action[0] % self._n_yaw_bins,
            )
            pitch_value = pitch_idx * self._cam_interval + min(self._cam_pitch_range)
            yaw_value = yaw_idx * self._cam_interval + min(self._cam_yaw_range)
            noop[3] = math.ceil((pitch_value - (-180)) / self._arnn_cam_interval)
            noop[4] = math.ceil((yaw_value - (-180)) / self._arnn_cam_interval)
        elif action[0] - self._n_cam_cartesian == 0:
            # forward
            noop[0] = 1
        elif action[0] - self._n_cam_cartesian == 1:
            # forward + jump
            noop[0] = 1
            noop[2] = 1
        elif action[0] - self._n_cam_cartesian == 2:
            # back
            noop[0] = 2
        elif action[0] - self._n_cam_cartesian == 3:
            # strafe left
            noop[1] = 1
        elif action[0] - self._n_cam_cartesian == 4:
            # strafe right
            noop[1] = 2
        elif action[0] - self._n_cam_cartesian == 5:
            # jump
            noop[2] = 1
        # other actions are identical with ARNNWrapper
        noop[5] = action[1]
        noop[6] = action[2]
        noop[7] = action[3]
        return noop

    def reverse_action(self, action):
        """
        ARNN action to trimmed action
        """
        assert len(action) == 8, (
            "actions from ARNNWrapper are supposed to have len = 8 but got"
            f" {len(action)} instead"
        )
        assert self.env.action_space.contains(action)

        noop = self.action_space.no_op()
        if action[0] == 0 and action[1] == 0 and action[2] != 1:
            # when bott F/B head and L/R head are no-op and jump head is not jump, let's keep the camera movement
            pitch = float(action[3]) * self._arnn_cam_interval + (-180)
            pitch = np.clip(pitch, *self._cam_pitch_range)
            yaw = float(action[4]) * self._arnn_cam_interval + (-180)
            yaw = np.clip(yaw, *self._cam_yaw_range)

            pitch_idx = math.ceil(
                (pitch - min(self._cam_pitch_range)) / self._cam_interval
            )
            yaw_idx = math.ceil((yaw - min(self._cam_yaw_range)) / self._cam_interval)
            # first increase yaw then increase pitch
            noop[0] = pitch_idx * self._n_yaw_bins + yaw_idx
        else:
            # trim cartesian product (noop, forward, back) x (noop, left, right) x (noop, jump, sneak, sprint)
            maps = {
                (0, 0, 1): self._n_cam_cartesian + 5,  # jump -> jump
                (0, 1, 0): self._n_cam_cartesian + 3,  # strafe left
                (0, 1, 1): self._n_cam_cartesian + 3,  # left + jump -> left
                (0, 2, 0): self._n_cam_cartesian + 4,  # strafe right
                (0, 2, 1): self._n_cam_cartesian + 4,  # right + jump -> right
                (1, 0, 0): self._n_cam_cartesian + 0,  # forward
                (1, 0, 1): self._n_cam_cartesian + 1,  # forward + jump
                (1, 1, 0): self._n_cam_cartesian + 0,  # forward + left -> forward
                (1, 1, 1): self._n_cam_cartesian
                + 1,  # forward + left + jump -> forward + jump
                (1, 2, 0): self._n_cam_cartesian + 0,  # forward + right -> forward
                (1, 2, 1): self._n_cam_cartesian
                + 1,  # forward + right + jump -> forward + jump
                (2, 0, 0): self._n_cam_cartesian + 2,  # back
                (2, 0, 1): self._n_cam_cartesian + 2,  # back + jump -> back
                (2, 1, 0): self._n_cam_cartesian + 2,  # back + left -> back
                (2, 1, 1): self._n_cam_cartesian + 2,  # back + left + jump -> back
                (2, 2, 0): self._n_cam_cartesian + 2,  # back + right -> back
                (2, 2, 1): self._n_cam_cartesian + 2,  # back + right + jump -> back
                (0, 1, 2): self._n_cam_cartesian + 3,  # sneak + left -> left
                (0, 1, 3): self._n_cam_cartesian + 3,  # sprint + left -> left
                (0, 2, 2): self._n_cam_cartesian + 4,  # sneak + right -> right
                (0, 2, 3): self._n_cam_cartesian + 4,  # sprint + right -> right
                # "forward" dominates
                (1, 0, 2): self._n_cam_cartesian + 0,
                (1, 0, 3): self._n_cam_cartesian + 0,
                (1, 1, 2): self._n_cam_cartesian + 0,
                (1, 1, 3): self._n_cam_cartesian + 0,
                (1, 2, 2): self._n_cam_cartesian + 0,
                (1, 2, 3): self._n_cam_cartesian + 0,
                # "back" dominates
                (2, 0, 2): self._n_cam_cartesian + 2,
                (2, 0, 3): self._n_cam_cartesian + 2,
                (2, 1, 2): self._n_cam_cartesian + 2,
                (2, 1, 3): self._n_cam_cartesian + 2,
                (2, 2, 2): self._n_cam_cartesian + 2,
                (2, 2, 3): self._n_cam_cartesian + 2,
            }

            noop[0] = maps[(action[0], action[1], action[2])]

        # ------ other actions are identical with ARNNWrapper ------
        noop[1] = action[5]
        noop[2] = action[6]
        noop[3] = action[7]
        return noop


class PreprocessedObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def observation(self, observation):
        gps = observation["location_stats"]["pos"]
        voxels = observation["voxels"]["block_meta"]
        rgb = observation["rgb"]
        pitch = observation["location_stats"]["pitch"]
        yaw = observation["location_stats"]["yaw"]
        # scalar to 1d array
        biome_id = observation["location_stats"]["biome_id"][None].copy()

        compass = np.concatenate(
            [np.sin(pitch), np.cos(pitch), np.sin(yaw), np.cos(yaw)]
        )

        obs = {
            "compass": compass.astype("float32"),
            "gps": gps,
            "voxels": voxels.reshape(-1).copy(),
            "biome_id": biome_id,
            "prev_action": self.env.prev_action[None].copy(),
            "rgb": rgb,
            "prompt": self.env.prompt_features,
        }

        return obs

class MotionActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        action_nvec = env.action_space.nvec  # value: [87, 8, 244, 36]
        noop_vec = env.action_space._noop_vec  # value: [40, 0, 0, 0]
        # motion actions
        new_action_nvec = list(action_nvec[:-3])  # [87,]
        new_noop_vec = noop_vec[:-3]  # [40,]
        self._motion_noop_idx = new_noop_vec[0]
        # attack and use actions
        new_action_nvec[0] += 2  # [89,]
        self._use_action_idx, self._attack_action_idx = (
            new_action_nvec[0] - 2,
            new_action_nvec[0] - 1,
        )
        self.action_space = spaces.MultiDiscrete(new_action_nvec, noop_vec=new_noop_vec)

    def action(self, action):
        # self._prev_action = np.int64(action[0])
        if action == self._use_action_idx or action == self._attack_action_idx:
            action_idx = 1 if action == self._use_action_idx else 3
            return np.array([self._motion_noop_idx, action_idx, 0, 0])
        else:
            return any_concat([action, np.array([0, 0, 0])])


class CachedActionWrapper(gym.Wrapper):
    def __init__(self, env, prompt: str, prompt_features: np.ndarray):
        super().__init__(env)
        self.prompt = prompt
        self._prompt_features = prompt_features
        self._prev_action = self.env.action_space.no_op()[0]

    @property
    def prompt_features(self):
        return self._prompt_features

    @property
    def prev_action(self):
        return self._prev_action

    def step(self, action):
        self._prev_action = action[0]
        return self.env.step(action)

    def reset(self):
        self._prev_action = self.env.action_space.no_op()[0]
        return self.env.reset()


class MineCLIPEnv(SubprocVectorEnv):
    def __init__(
        self,
        env_fns: List[Callable[[], gym.Env]],
        mineclip_model=None,
        max_img_buffer_len: int = 16,
        device: str = "cuda",
        **kwargs: Any,
    ) -> None:
        super().__init__(env_fns, **kwargs)
        assert mineclip_model is not None

        self.mineclip_model = mineclip_model

        self.device = device

        self.image_feature_buffer = [
            deque(maxlen=max_img_buffer_len) for _ in range(len(env_fns))
        ]

        self.max_img_buffer_len = max_img_buffer_len

        obs_space = self.get_env_attr("observation_space")[0]

        new_space = {k: v for k, v in obs_space.items()}

        new_space["img_feats"] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(512,),
            dtype=np.float32,
        )

        new_space = gym.spaces.Dict(new_space)

        self.set_env_attr("observation_space", new_space)

    @torch.no_grad()
    def _reset_feature_buffer(self, env_id):
        pad = torch.zeros(512, device=self.device)
        buffer = self.image_feature_buffer[env_id]
        buffer.clear()
        for _ in range(self.max_img_buffer_len):
            buffer.append(pad)


    @torch.no_grad()
    def _video_reward_fn(
        self, next_image: List[np.ndarray], env_ids: List[int]
    ) -> np.ndarray:
        """Reward function from mineclip."""
        image_tensor = torch.tensor(np.array(next_image), device=self.device)
        image_feats = self.mineclip_model.forward_image_features(image_tensor)
        video_feats = []

        for index, env_id in enumerate(env_ids):
            image_feat = image_feats[index]
            buffer = self.image_feature_buffer[env_id]
            buffer.append(image_feat)
            current_env_frames = list(buffer)
            env_video = torch.stack(current_env_frames)
            video_feats.append(env_video)


        video_feats = torch.stack(video_feats)

        # print("video_feats.shape before", video_feats.shape)

        video_feats = self.mineclip_model.forward_video_features(video_feats)

        # print("video_feats.shape after", video_feats.shape)

        text_feats = self.get_env_attr("prompt_features", env_ids)

        text_feats = torch.tensor(np.array(text_feats), device=self.device)

        # print("text_feats.shape", text_feats.shape)

        logits_per_video, logits_per_prompt = self.mineclip_model.forward_reward_head(
            video_feats,
            text_tokens=text_feats,
        )

        # print("logits_per_prompt.shape", logits_per_prompt.shape)
        # print("logits_per_video.shape", logits_per_video.shape)

        prob_per_prompt = logits_per_prompt.softmax(dim=-1)

        video_reward = (
            torch.relu(torch.diag(prob_per_prompt) - (1 / prob_per_prompt.shape[-1]))
            .cpu()
            .numpy()
        )

        return video_reward

    def step(
        self, action: np.ndarray, id: Optional[Union[int, List[int], np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run one timestep of the batched environment's dynamics."""
        # print("action: ", action)
        ids = self._wrap_id(id)

        obs, native_reward, done, info = super().step(action, ids)

        highres_rgbs = [ob["rgb"] for ob in obs]

        video_reward = self._video_reward_fn(highres_rgbs, ids)

        is_successful = self.get_env_attr("is_successful", ids)

        for index, env_id in enumerate(ids):
            obs[index]["img_feats"] = self.image_feature_buffer[env_id][-1].cpu().numpy()
            info[index]["native_reward"] = native_reward[index]
            info[index]["is_successful"] = is_successful[index]

        reward = native_reward + video_reward

        return obs, reward, done, info

    def reset(
        self, id: Optional[Union[int, List[int], np.ndarray]] = None
    ) -> np.ndarray:
        """Reset all the environments and return an array of observations, or a
        tuple of arrays of observations if ``self.num_envs > 1``.
        """
        ids = self._wrap_id(id)
        obs = super().reset(ids)

        for env_id in ids:
            self._reset_feature_buffer(env_id)

        highres_rgbs = [ob["rgb"] for ob in obs]

        self._video_reward_fn(highres_rgbs, ids)

        for index, env_id in enumerate(ids):
            obs[index]["img_feats"] = self.image_feature_buffer[env_id][-1].cpu().numpy()

        return obs

class SelfBCEnv(MineCLIPEnv):

    def __init__(self, env_fns: List[Callable[[], gym.Env]], *args, si_frequency: int = 100000, **kwargs):
        super().__init__(env_fns, *args, **kwargs)

        self.si_buffer = [deque(maxlen=10) for _ in range(len(env_fns))]
        self.si_counter = 0
        self.si_frequency = si_frequency
        self.si_needed = False
        self.current_buffer = [[] for _ in range(len(env_fns))]

    def step(
        self, action: np.ndarray, id: Optional[Union[int, List[int], np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run one timestep of the batched environment's dynamics."""
        env_ids = self._wrap_id(id)

        output = super().step(action, env_ids)

        obs, reward, done, info = output

        for index, env_id in enumerate(env_ids):
            env_buffer = self.current_buffer[env_id]
            data = Batch(obs_next=obs[index], act=action[index], rew=reward[index])
            env_buffer.append(data)

            if done[index]:
                if info[index]["is_successful"]:
                    print("successful episode will added to si_buffer")
                    print("env id: ", env_id)
                    episode = Batch.stack(env_buffer)
                    obs = episode.obs_next[:-1]
                    act = episode.act[1:]
                    rew = episode.rew[1:] * 5
                    episode = Batch(obs=obs, act=act, rew=rew, info=obs)
                    self.si_buffer[env_id].append(episode)
                    self.si_counter += len(env_buffer)
                    print("si counter: ", self.si_counter)
                else:
                    episode = Batch.stack(env_buffer)
                    obs = episode.obs_next[:-1]
                    act = episode.act[1:]
                    rew = episode.rew[1:]
                    episode = Batch(obs=obs, act=act, rew=rew, info=obs)
                    sum_reward = rew.sum()
                    si_rews = [ep.rew.sum() for ep in self.si_buffer[env_id]]
                    si_mean = np.mean(si_rews) if len(si_rews) > 0 else 0.0
                    si_std = np.std(si_rews) if len(si_rews) > 0 else 0.0
                    if sum_reward > si_mean + 2 * si_std:
                        print("high reward episode will added to si_buffer")
                        print("env id: ", env_id)
                        self.si_buffer[env_id].append(episode)
                        self.si_counter += len(env_buffer)
                        print("si counter: ", self.si_counter)

                self.current_buffer[env_id] = []

        obs, reward, done, info = output
        return output


    def reset(self, id: Optional[Union[int, List[int], np.ndarray]] = None) -> np.ndarray:
        env_ids = self._wrap_id(id)

        obs = super().reset(env_ids)

        for index, env_id in enumerate(env_ids):
            self.current_buffer[env_id] = []
            noop = self.get_env_attr("action_space")[0].no_op()[0]
            data = Batch(obs_next=obs[index], act=np.array([noop]), rew=np.array([0.0]))
            self.current_buffer[env_id].append(data)
        return obs



# def transform_action(action):
#     """
#     Map agent action to env action.
#     """
#     assert action.ndim == 2
#     action = action[0]
#     action = action.cpu().numpy()
#     if action[-1] != 0 or action[-1] != 1 or action[-1] != 3:
#         action[-1] = 0
#     action = np.concatenate([action, np.array([0, 0])])
#     return action


# @hydra.main(config_name="conf", config_path=".", version_base="1.1")
# def main(cfg):
#     random.seed(42)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Init mineclip model
#     mineclip_kwargs = cfg.mineclip_kwargs
#     # ProgressMeter just a wrapper for mineclip model
#     model = ProgressMeter(
#         checkpoint="/home/akyurek/git/MineCLIP/mineclip/checkpoints/attn.pth",
#         arch=mineclip_kwargs.arch,
#         hidden_dim=mineclip_kwargs.hidden_dim,
#         resolution=mineclip_kwargs.resolution,
#         image_feature_dim=mineclip_kwargs.image_feature_dim,
#         mlp_adapter_spec=mineclip_kwargs.mlp_adapter_spec,
#         pool_type=mineclip_kwargs.pool_type,
#     ).to(device)

#     # Init actor's feature net
#     feature_net_kwargs = cfg.feature_net_kwargs
#     feature_net = {}
#     for k, v in feature_net_kwargs.items():
#         v = dict(v)
#         cls = v.pop("cls")
#         cls = getattr(F, cls)
#         feature_net[k] = cls(**v, device=device)
#     feature_fusion_kwargs = cfg.feature_fusion
#     feature_net = SimpleFeatureFusion(
#         feature_net, **feature_fusion_kwargs, device=device
#     )

#     # init actor
#     actor = MultiCategoricalActor(
#         feature_net,
#         action_dim=[87, 8, 244, 36],  # [3, 3, 4, 25, 25, 8],
#         device=device,
#         **cfg.actor,
#     )

#     critic = MineCritic(copy.deepcopy(feature_net), device=device, **cfg.actor).to(
#         device
#     )

#     # init agent
#     mine_agent = MineAgent(
#         actor=actor,
#     ).to(device)

#     task_ids = list(minedojo.tasks.ALL_PROGRAMMATIC_TASK_INSTRUCTIONS.keys())
#     random.shuffle(task_ids)
#     for task_id in task_ids[:1]:
#         instruction = minedojo.tasks.ALL_PROGRAMMATIC_TASK_INSTRUCTIONS[task_id]
#         prompt = instruction[1]
#         with torch.no_grad():
#             prompt_feature = (
#                 model.clip_model.encode_text([prompt]).flatten().cpu().numpy()
#             )

#         prompt_info = {task_id: {"features": prompt_feature, "text": prompt}}

#         def wrapped_env_fn(task_id: str, prompt_info):
#             # init env
#             env = minedojo.make(
#                 task_id=task_id, image_size=(160, 256), world_seed=123, seed=42
#             )
#             # get the prompt
#             prompt_text = prompt_info["text"]
#             # random prompt features for placeholder
#             prompt_features = prompt_info["features"]
#             # cache previous action
#             env = FlattenedMotionCamera(env)
#             env = CachedActionWrapper(env, prompt_text, prompt_features)
#             env = PreprocessedObservation(env)
#             return env

#         env_fn = functools.partial(
#             wrapped_env_fn, task_id=task_id, prompt_info=prompt_info[task_id]
#         )

#         # # Test unbatched env
#         # env = env_fn()
#         # print("action space", env.action_space)
#         # obs = env.reset()
#         # print("obs", obs)
#         # # mine_agent model expects batched obs, add extra batch dimension
#         # for k, v in obs.items():
#         #     if k == "rgb":
#         #         v = np.expand_dims(v, 0)
#         #         v = torch.tensor(v).to(device)
#         #         v = model.clip_model.forward_image_features(v)
#         #     else:
#         #         v = np.expand_dims(v, 0)
#         #         v = torch.tensor(v).to(device)
#         #     print(k, v.shape, v.dtype)
#         #     obs[k] = v

#         # output = mine_agent.forward(Batch(obs=obs))
#         # print("act: ", output.act)
#         # # action = transform_action(output.act)
#         # # print("act: ", output.act)
#         # action = output.act.cpu().numpy().flatten()

#         # Test batched env
#         env = MineCLIPEnv(
#             [env_fn, env_fn], mineclip_model=model.clip_model, device=device
#         )
#         obs = env.reset()

#         batched_obs = Batch(obs=obs)

#         # print("batched obs", batched_obs.obs.rgb.shape)

#         output = mine_agent.forward(batched_obs)
#         # print("act: ", output.act)
#         # action = transform_action(output.act)
#         # print("act: ", output.act)
#         action = output.act.cpu().numpy()

#         obs, *_ = env.step(action)

#         batched_obs = Batch(obs=obs)

#         output = mine_agent.forward(batched_obs)
#         # print("act: ", output.act)

#         value = critic.forward(batched_obs.obs)

#         # print("value: ", value)

#         # print("done")


# if __name__ == "__main__":
#     main()  # pylint: disable=no-value-for-parameter
