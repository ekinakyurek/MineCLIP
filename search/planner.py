from typing import Tuple, Optional, Any, Mapping
from gpt.src.specification import EndPointSpec, OpenAIEndPointArgs, APIArgs
from gpt.src.api import run_pipeline
from search.searcher import Searcher
import minedojo


def postprocesser(inps, outs):
    arr = [
        {
            "text": out["text"],
            "instruction": inp["instruction"],
            "observation": inp["observation"],
        }
        for inp, out in zip(inps, outs)
    ]

    return arr


def merge_plans(inps, outs):
    arr = [
        {
            "initial_plan": inp["text"],
            **out,
        }
        for inp, out in zip(inps, outs)
    ]

    return arr


def preprocesser(arr):
    arr = [
        {
            "instructions": "\n====\n".join(
                [instruction["text"] for instruction in arr]
            ),
            "instruction": arr[0]["instruction"],
            "observation": arr[0]["observation"],
        }
    ]
    return arr


def stringify_obs(obs):
    id2biome = {v: k for k, v in minedojo.sim.BIOMES_MAP.items()}

    def clean(inventory, quantity):
        return [f"{count} {item}" for item, count in zip(inventory, quantity) if count > 0]

    inventory = clean(obs["inventory"]["name"], obs["inventory"]["quantity"])
    inventory = ", ".join(inventory)

    equipment = clean(obs["equipment"]["name"], obs["equipment"]["quantity"])
    equipment = ", ".join(equipment)

    output = (
        "The agent is in the"
        f" {id2biome[int(obs['location_stats']['biome_id'])]} biome\nThe agent has the"
        f" following items in its inventory: {inventory}\nThe agent has the following"
        f" equipments: {equipment}\n"
    )
    return output


class SearchPlanner:
    def __init__(
        self,
        searcher: Searcher,
        pipeline: Optional[Tuple[EndPointSpec]] = None,
        api_keys: Optional[Tuple[str]] = None,
        debug: bool = False,
    ):
        self.searcher = searcher
        self.api_keys = api_keys
        self.debug = debug
        if pipeline is None:
            self.pipeline = self.make_pipeline()

    def make_pipeline(
        self,
        model: str = "gpt-4",
        temperature: float = 0.25,
        top_p: float = 1.0,  # 0.9,
        max_input_tokens: int = 3512,
        max_tokens: int = 512,
        frequency_penalty: float = 0.25,  # 0.25,
        presence_penalty: float = 0.25,  # 0.25,
    ):
        endpoint_args = OpenAIEndPointArgs(
            model=model,
            max_tokens=max_tokens,
            n=1,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

        args = APIArgs(
            endpoint_args,
            max_input_tokens=max_input_tokens,
            truncate_input=True,
        )

        pipeline = [
            EndPointSpec(
                name="summarize_a_page",
                template=(
                    "I would like to get step-by-step instructions for the goal"
                    ' "{instruction}" in MineCraft (Java Edition).\nI found the'
                    " following texts on the internet that might be"
                    ' helpful:\n\n"""{texts}"""\n\nCan you please generate step-by-step'
                    " plan according to above information. Stop when the goal is"
                    " achieved. Please write your plan as concise as possible and"
                    " specify how to craft each material required as a step.\nPLAN:\n"
                ),
                args=args,
                postprocesser=postprocesser,
            ),
            EndPointSpec(
                name="summarize_multiple_instructions",
                template=(
                    'I have found following instructions for the task "{instruction}"'
                    " in MineCraft (Java Edition) from multiple pages.\nI extracted the"
                    " following instructions\n====\n{instructions}\n==END==\nCan you"
                    " please generate step-by-step plan according to above information."
                    " Stop when the goal is achieved. Please write your plan as concise"
                    " as possible and specify how to craft each material required as a"
                    " step.\nFINAL PLAN:\n"
                ),
                args=args,
                preprocesser=preprocesser,
                postprocesser=postprocesser,
            ),
            EndPointSpec(
                name="update_plan_with_observation",
                template=(
                    "Initial Plan:\n{text}\n\nThe observation:\n{observation}\n\nPlese"
                    " update the plan based on the observation, remove any steps that"
                    " are already completed.\nUpdated Plan:"
                ),
                args=args,
                postprocesser=merge_plans,
            ),
        ]

        return pipeline

    async def plan(
        self, instruction: Mapping[str, Any], obs: Optional[Mapping[str, Any]] = None
    ):
        query_text = instruction["instruction"]
        search_outputs = self.searcher.search(query_text)

        if obs is not None:
            obs = stringify_obs(obs)
            pipeline = self.pipeline
        else:
            pipeline = self.pipeline[:-1]

        data = [
            {
                "instruction": query_text,
                "texts": search_outputs[i]["texts"],
                "observation": obs,
            }
            for i in range(len(search_outputs))
        ]

        output = await run_pipeline(data, pipeline, api_keys=self.api_keys)
        return {"plans": output, "search_outputs": search_outputs}


if __name__ == "__main__":
    import asyncio
    import os

    searcher = Searcher()
    api_keys = os.environ.get("OPENAI_API_KEY_POOL").split(",")
    planner = SearchPlanner(searcher, api_keys=api_keys, debug=True)
    loop = asyncio.get_event_loop()
    plan = loop.run_until_complete(
        planner.plan(
            {"instruction": "starting from iron tools, craft and use a golden shovel"}
        )
    )
    plan_text = plan["plans"][0]["text"].strip()
    print(plan_text)
