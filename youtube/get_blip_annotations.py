import os
import torch
import itertools
import json
from lavis.models import load_model_and_preprocess
from PIL import Image
from torchvision.io import VideoReader
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

class GPTFramesDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        annotations,
        frame_transform=None,
        video_transform=None,
        rate: int = 5,
        seed: int = 42,
    ):
        super(GPTFramesDataset).__init__()

        self.data = {}
        self.annotations = annotations
        self.rng = np.random.default_rng(seed)

        total_frames = 0.0
        for _, row in annotations.iterrows():
            video_path = "youtube/dgx_videos/" + row["file"]
            summary_path = "youtube/samples/" + row["file"].split(".")[0] + ".en.vtt.gpt.srt"
            if os.path.isfile(video_path) and os.path.isfile(summary_path):
                self.data[video_path] = {"frames": [], "info": row}
                with open(summary_path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        if " - " in line:
                            try:
                                start, end = line.strip().split(" - ")
                                end = end.split(":")[0].strip()
                                start = int(start)
                                end = int(end)
                                if start == 0:
                                    continue
                                self.data[video_path]["frames"].append((start, end))
                                total_frames += end - start
                            except Exception as err:
                                print("Error in file: ", summary_path)
                                print("line: ", line)
                                print("error", str(err))
                                continue

        self.epoch_size = int(np.round(total_frames / rate))
        print(self.epoch_size)
        self.frame_transform = frame_transform
        self.video_transform = video_transform
        self.seed = seed


    def __iter__(self):
        print("Iteration starts")
        for file, info in self.data.items():
            path = file
            row = info["info"]
            frames = info["frames"]
            if not os.path.exists(path):
                print("path does not exist: ", path)
                continue
            nbframes = float(row["nbframes"])
            duration = float(row["duration"])
            fps = int(nbframes / duration)
            # max_seek = duration - (self.clip_len / fps)
            print("path: ", path)
            print("frames: ", frames)
            print("fps: ", fps)
            print("file", file)
            for frame in frames:
                start_frame, end_frame = frame
                for frame in range(start_frame, end_frame, 10):
                    vid = VideoReader(path, "video")
                    time = frame / fps
                    image = next(vid.seek(time))['data']

                    if self.frame_transform is not None:
                        image = self.frame_transform(image)

                    output = {"path": path, "image": image, "info": row.to_dict(), "frame": frame}
                    yield output

if __name__ == "__main__":
    import pandas as pd
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    df = pd.read_csv("youtube/dgx_videos.csv")
    frame_transform = transforms.Compose(
        [
            transforms.Resize((256, 512), interpolation=transforms.InterpolationMode.BICUBIC),
        ]
    )
    to_pil_image = transforms.ToPILImage()

    dataset = GPTFramesDataset(df, frame_transform=frame_transform)

    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5_instruct",
        model_type="flant5xxl",
        is_eval=True,
        device=
        # model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=
        device,
    )
    dataloader = DataLoader(dataset, batch_size=1)
    results = {}
    for batch in dataloader:
        path = str(batch["path"][0])
        image = batch["image"][0]
        image = to_pil_image(image)
        frame = int(batch["frame"][0])
        image = vis_processors["eval"](image).unsqueeze(0).to(device)
        output = model.generate(
            {
                "image": image,
                "prompt": (
                    "Write a detailed description of the screenshot from a minecraft recording and then tell what is the intent of the gamer behind the camera."
                ),
            },
            use_nucleus_sampling=True,
            top_p=0.9,
            temperature=1,
        )

        if path not in results:
            results[path] = {}
        results[path][frame] = output
        with open("youtube/blip_annotations.json", "w") as handle:
            json.dump(results, handle)





    # raw_image = Image.open("youtube/samples/ss2.png").convert("RGB")
    # image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    # print(
    #     model.generate(
    #         {
    #             "image": image,
    #             "prompt": (
    #                 "Write a detailed description of the image by using minecraft"
    #                 " specific names."
    #             ),
    #         },
    #         use_nucleus_sampling=True,
    #         top_p=0.9,
    #         temperature=1,
    #     )
    # )
