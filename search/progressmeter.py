from typing import Tuple, List, Optional
from mineclip import MineCLIP
import torch
from torch import nn
import logging


class ProgressMeter(nn.Module):
    def __init__(
        self,
        arch: str = "vit_base_p16_fz.v2.t2",
        hidden_dim: int = 512,
        resolution: List[int] = [160, 256],
        image_feature_dim: int = 512,
        mlp_adapter_spec: str = "v0-2.t0",
        pool_type: str = "attn.d2.nh8.glusw",
        checkpoint: Optional[str] = None,
    ):
        super(ProgressMeter, self).__init__()

        self.clip_model = MineCLIP(
            arch=arch,
            hidden_dim=hidden_dim,
            image_feature_dim=image_feature_dim,
            mlp_adapter_spec=mlp_adapter_spec,
            resolution=resolution,
            pool_type=pool_type,
        )

        self.checkpoint = checkpoint

        if self.checkpoint:
            print(f"Loading MineCLIP chcekpoint {checkpoint}")
            self.clip_model.load_ckpt(checkpoint, strict=True)

    def forward(self, video: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        # video.shape: (vide_batch, frame_count, 3, H, W)
        # len(prompts): text_batch
        video_feats = self.clip_model.encode_video(video)
        text_feats = self.clip_model.encode_text(prompts)
        return self.clip_model.forward_reward_head(
            video_feats, text_tokens=text_feats,
        )


if __name__ == "__main__":
    import minedojo
    import os
    import asyncio
    import pickle
    import random
    from search.planner import SearchPlanner
    from search.searcher import Searcher



    device = "cuda"

    model = ProgressMeter(
        checkpoint="/home/akyurek/git/MineCLIP/mineclip/checkpoints/attn.pth",
        arch="vit_base_p16_fz.v2.t2",
        hidden_dim=512,
        resolution=[160, 256],
        image_feature_dim=512,
        mlp_adapter_spec="v0-2.t0",
        pool_type="attn.d2.nh8.glusw",
        device=device,
    )

    api_keys = os.environ.get("OPENAI_API_KEY_POOL").split(",")
    planner = SearchPlanner(Searcher(), api_keys=api_keys)

    random.seed(42)
    task_ids = list(minedojo.tasks.ALL_PROGRAMMATIC_TASK_INSTRUCTIONS.keys())
    random.shuffle(task_ids)

    for task_id in task_ids[:10]:
        instruction = minedojo.tasks.ALL_PROGRAMMATIC_TASK_INSTRUCTIONS[task_id]
        instruction = {
            "task_id": task_id,
            "instruction": instruction[0],
            "gpt_plan": instruction[1],
        }

        env = minedojo.make(
            task_id=task_id,
            image_size=(160, 256),
            world_seed=123,
            seed=42,
        )

        obs = env.reset()

        loop = asyncio.get_event_loop()

        plan = loop.run_until_complete(planner.plan(instruction, obs))

        plan_steps = plan["plans"][0]["text"].strip().split("\n")

        print(plan_steps)

        cumulated_instructions = plan_steps

        # ["\n".join(plan_steps[:i]) for i in range(len(plan_steps))]

        video_array = obs["rgb"].reshape(1, 1, 3, 160, 256).copy()
        video_tensor = torch.from_numpy(video_array).to(device)

        breakpoint()
        scores = model(video_tensor, cumulated_instructions)

        logits_per_video, logits_per_text = scores

        with open(f"search/chat/{task_id}.pkl", "wb") as handle:
            pickle.dump(
                {
                    "obs": obs,
                    "scores": scores,
                    "prompts": cumulated_instructions,
                    "plan": plan,
                    "instruction": instruction,
                },
                handle,
            )
