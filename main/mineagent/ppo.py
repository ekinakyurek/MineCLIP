import copy
import datetime
import functools
import os
import pprint
import gym
import hydra
import minedojo
import numpy as np
from tianshou.data import Collector
from tianshou.data import VectorReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils import WandbLogger
from tianshou.utils.net.common import ActorCritic
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from environments.minecraft import CachedActionWrapper
from environments.minecraft import FlattenedMotionCamera
from environments.minecraft import MotionActionWrapper
from environments.minecraft import MineCLIPEnv
from environments.minecraft import PreprocessedObservation
from mineclip import MineCritic
from mineclip import MultiCategoricalActor
from mineclip import SimpleFeatureFusion
from mineclip.mineagent import features as F
from mineclip import MineCLIP


def make_minecraft_envs(cfg, device="cpu"):
    rng = np.random.default_rng(seed=cfg.ppo_args.seed)
    # Init mineclip model
    mineclip_kwargs = cfg.mineclip_kwargs
    # ProgressMeter just a wrapper for mineclip model
    clip_model = MineCLIP(
        arch=mineclip_kwargs.arch,
        hidden_dim=mineclip_kwargs.hidden_dim,
        resolution=mineclip_kwargs.resolution,
        image_feature_dim=mineclip_kwargs.image_feature_dim,
        mlp_adapter_spec=mineclip_kwargs.mlp_adapter_spec,
        pool_type=mineclip_kwargs.pool_type,
    )
    clip_model.load_ckpt("/home/akyurek/git/MineCLIP/mineclip/checkpoints/attn.pth",
                         strict=True)
    clip_model.to(device)
    clip_model.device = device

    task_ids = [key for key in minedojo.tasks.ALL_PROGRAMMATIC_TASK_INSTRUCTIONS.keys()
                if key.startswith("harvest")]

    rng.shuffle(task_ids)

    train_task_ids = task_ids[:20]
    test_task_ids = task_ids[:20]

    def wrapped_env_fn(task_id: str, prompt_info, time_limit: int = 0):
        # init env
        env = minedojo.make(
            task_id=task_id, image_size=(160, 256), world_seed=123, seed=42
        )
        # get the prompt
        prompt_text = prompt_info["text"]
        # random prompt features for placeholder
        prompt_features = prompt_info["features"]
        # cache previous action
        env = FlattenedMotionCamera(env)
        env = MotionActionWrapper(env)
        env = CachedActionWrapper(env, prompt_text, prompt_features)
        env = PreprocessedObservation(env)
        if time_limit > 0:
            env = gym.wrappers.TimeLimit(env, time_limit)
        return env

    prompt_info = {}

    for task_id in set(train_task_ids + test_task_ids):
        instruction = minedojo.tasks.ALL_PROGRAMMATIC_TASK_INSTRUCTIONS[task_id]
        prompt = instruction[1]
        with torch.no_grad():
            prompt_feature = (
                clip_model.encode_text([prompt]).flatten().cpu().numpy()
            )

        prompt_info[task_id] = {"features": prompt_feature, "text": prompt}

    def get_env_fn(task_id: str, time_limit: int = 0):
        env_fn = functools.partial(
            wrapped_env_fn,
            task_id=task_id,
            prompt_info=prompt_info[task_id],
            time_limit=time_limit
        )
        return env_fn

    train_envs = MineCLIPEnv(
        [get_env_fn(task_id) for task_id in train_task_ids],
        mineclip_model=clip_model,
        device=clip_model.device,
    )
    test_envs = MineCLIPEnv(
        [get_env_fn(task_id, time_limit=500) for task_id in test_task_ids],
        mineclip_model=clip_model,
        device=clip_model.device,
    )
    envs = train_envs
    return envs, train_envs, test_envs


@hydra.main(config_name="ppo", config_path=".", version_base="1.1")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env, train_envs, test_envs = make_minecraft_envs(cfg)

    observation_space = env.get_env_attr("observation_space")[0]
    action_space = env.get_env_attr("action_space")[0]
    cfg.state_shape = observation_space.shape
    cfg.action_shape = action_space.shape
    # should be N_FRAMES x H x W
    print("Observations shape:", cfg.state_shape)
    print("Actions shape:", cfg.action_shape)
    # seed
    np.random.seed(cfg.ppo_args.seed)
    torch.manual_seed(cfg.ppo_args.seed)

    # Init actor's feature net
    feature_net_kwargs = cfg.feature_net_kwargs
    feature_net = {}
    for k, v in feature_net_kwargs.items():
        v = dict(v)
        cls = v.pop("cls")
        cls = getattr(F, cls)
        feature_net[k] = cls(**v, device=device)
    feature_fusion_kwargs = cfg.feature_fusion
    feature_net = SimpleFeatureFusion(
        feature_net, **feature_fusion_kwargs, device=device
    )

    # init actor
    actor = MultiCategoricalActor(
        feature_net,
        action_dim=action_space.nvec.tolist(),  # [3, 3, 4, 25, 25, 8],
        device=device,
        **cfg.actor,
    ).to(device)

    critic = MineCritic(copy.deepcopy(feature_net), device=device, **cfg.actor).to(
        device
    )

    optim = torch.optim.Adam(
        ActorCritic(actor, critic).parameters(), lr=cfg.ppo_args.lr, eps=1e-5
    )

    lr_scheduler = None
    if cfg.ppo_args.lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = (
            np.ceil(cfg.ppo_args.step_per_epoch / cfg.ppo_args.step_per_collect)
            * cfg.ppo_args.epoch
        )
        lr_scheduler = CosineAnnealingLR(optim, max_update_num, eta_min=cfg.ppo_args.min_lr)

    policy = PPOPolicy(
        actor,
        critic,
        optim,
        actor.dist_fn,
        discount_factor=cfg.ppo_args.gamma,
        gae_lambda=cfg.ppo_args.gae_lambda,
        max_grad_norm=cfg.ppo_args.max_grad_norm,
        vf_coef=cfg.ppo_args.vf_coef,
        ent_coef=cfg.ppo_args.ent_coef,
        reward_normalization=cfg.ppo_args.rew_norm,
        action_scaling=False,
        lr_scheduler=lr_scheduler,
        action_space=env.action_space,
        eps_clip=cfg.ppo_args.eps_clip,
        value_clip=cfg.ppo_args.value_clip,
        dual_clip=cfg.ppo_args.dual_clip,
        advantage_normalization=cfg.ppo_args.norm_adv,
        recompute_advantage=cfg.ppo_args.recompute_adv,
    ).to(device)

    # load a previous policy
    if cfg.ppo_args.resume_path:
        policy.load_state_dict(
            torch.load(cfg.ppo_args.resume_path, map_location=device)["model"]
        )
        print("Loaded agent from: ", cfg.ppo_args.resume_path)
    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
    buffer = VectorReplayBuffer(
        cfg.ppo_args.buffer_size,
        buffer_num=len(train_envs),
        ignore_obs_next=True,
    )
    # collector
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    # if cfg.ppo_args.si_frequency > -1:
    #     si_buffer = PrioritizedVectorReplayBuffer(
    #         total_size=cfg.ppo_args.buffer_size,
    #         alpha=cfg.ppo_args.si_alpha,
    #         beta=cfg.ppo_args.si_beta,
    #         buffer_num=len(train_envs),
    #         ignore_obs_next=True,
    #     )
    #     si_collector = Collector(policy, train_envs, si_buffer)
    # else:
    #     si_collector = None


    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    cfg.algo_name = "ppo"
    log_name = os.path.join(
        cfg.ppo_args.task, cfg.algo_name, str(cfg.ppo_args.seed), now
    )
    log_path = os.path.join(cfg.ppo_args.logdir, log_name)

    # logger
    if cfg.ppo_args.logger == "wandb":
        logger = WandbLogger(
            save_interval=1,
            name=log_name.replace(os.path.sep, "__"),
            run_id=cfg.ppo_args.resume_id,
            config=cfg,
            project=cfg.wandb_kwargs.wandb_project,
        )
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(cfg))
    if cfg.ppo_args.logger == "tensorboard":
        logger = TensorboardLogger(writer)
    else:  # wandb
        logger.load(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards):
        try:
            if env.spec.reward_threshold:
                return mean_rewards >= env.spec.reward_threshold
            else:
                return False
        except AttributeError:
            return False

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(log_path,
                                 f"checkpoint_{epoch}_{env_step}_{gradient_step}.pth")
        torch.save({"model": policy.state_dict()}, ckpt_path)
        return ckpt_path

    # watch agent's performance
    def watch():
        print("Setup test envs ...")
        policy.eval()
        test_envs.seed(cfg.ppo_args.seed)
        if cfg.ppo_args.save_buffer_name:
            print(f"Generate buffer with size {cfg.ppo_args.buffer_size}")
            buffer = VectorReplayBuffer(
                cfg.ppo_args.buffer_size,
                buffer_num=len(test_envs),
                ignore_obs_next=True,
            )
            collector = Collector(policy, test_envs, buffer, exploration_noise=True)
            result = collector.collect(n_step=cfg.ppo_args.buffer_size)
            print(f"Save buffer into {cfg.ppo_args.save_buffer_name}")
            # Unfortunately, pickle will cause oom with 1M buffer size
            buffer.save_hdf5(cfg.ppo_args.save_buffer_name)
        else:
            print("Testing agent ...")
            test_collector.reset()
            result = test_collector.collect(
                n_episode=cfg.ppo_args.test_num, render=cfg.ppo_args.render
            )
        rew = result["rews"].mean()
        print(f"Mean reward (over {result['n/ep']} episodes): {rew}")

    if cfg.ppo_args.watch:
        watch()
        exit(0)

    # test train_collector and start filling replay buffer
    train_collector.collect(n_step=cfg.ppo_args.batch_size * cfg.ppo_args.training_num)
    # trainer
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        cfg.ppo_args.epoch,
        cfg.ppo_args.step_per_epoch,
        cfg.ppo_args.repeat_per_collect,
        cfg.ppo_args.test_num,
        cfg.ppo_args.batch_size,
        step_per_collect=cfg.ppo_args.step_per_collect,
        stop_fn=stop_fn,
        save_fn=save_best_fn,
        logger=logger,
        test_in_train=False,
        resume_from_log=cfg.ppo_args.resume_id is not None,
        save_checkpoint_fn=save_checkpoint_fn,
        # si_collector=si_collector,
    )

    pprint.pprint(result)
    watch()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
