from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from tianshou.data import Batch, to_torch
from tianshou.policy import BasePolicy


class ImitationPolicy(BasePolicy):
    """Implementation of vanilla imitation learning.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> a)
    :param torch.optim.Optimizer optim: for optimizing the model.
    :param gym.Space action_space: env's action space.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.action_type = "discrete"

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        logits, hidden = self.model(batch.obs, state=state, info=batch.info)
        if self.action_type == "discrete":
            act = logits.max(dim=1)[1]
        else:
            act = logits
        return Batch(logits=logits, act=act, state=hidden)

    def learn(self, batch: Batch,  batch_size: int, repeat: int, **kwargs: Any) -> Dict[str, float]:
        losses = []
        for _ in range(repeat):
            for minibatch in batch.split(batch_size, merge_last=True):
                self.optim.zero_grad()
                if self.action_type == "continuous":  # regression
                    act = self(minibatch).act
                    act_target = to_torch(minibatch.act, dtype=torch.float32, device=act.device)
                    loss = F.mse_loss(act, act_target)  # type: ignore
                elif self.action_type == "discrete":  # classification
                    logits = self(minibatch).logits
                    act_target = to_torch(minibatch.act, dtype=torch.long, device=logits.device)
                    logits = logits.reshape(-1, logits.shape[-1])
                    act_target = act_target.reshape(-1)
                    loss = F.cross_entropy(logits, act_target, reduction='mean')  # type: ignore
                loss.backward()
                self.optim.step()
                losses.append(loss.item())
                    # update learning rate if lr_scheduler is given
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        return {"loss": losses}
