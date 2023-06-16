import os
import torch
import numpy as np
import glob
import sys
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
import pandas as pd

import decord
from decord import VideoReader
from decord import cpu
import time

decord.bridge.set_bridge("torch")


class VideoFramesIterator(torch.utils.data.IterableDataset):
    def __init__(
        self,
        annotations: pd.DataFrame,
        frame_transform=None,
        nframes_per_iteration: int = 29,
        seed: int = 42,
    ):
        super(VideoFramesIterator).__init__()

        self.annotations = annotations
        self.nframes_per_iteration = nframes_per_iteration
        self.frame_transform = frame_transform
        self.epoch_size = int(self.annotations.nbframes.sum() / nframes_per_iteration)
        self.seed = seed
        self.nframes_per_iteration = nframes_per_iteration
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        for i in range(self.epoch_size):
            # Get random sample
            sample = self.annotations.iloc[i]
            path = "youtube/samples/" + sample["file"]

            if not os.path.exists(path):
                print("path does not exist: ", path)
                continue

            vr = VideoReader(path, ctx=cpu(0), width=224, height=224)

            start_frame = 0
            video_frames = []

            for frame, data in enumerate(vr):
                data = data.permute(2, 0, 1)

                if self.frame_transform is not None:
                    data = self.frame_transform(data)

                video_frames.append(data)

                if frame == start_frame + self.nframes_per_iteration:
                    out = {
                        "path": path,
                        "video": video_frames,
                        "start": start_frame,
                        "end": frame,
                    }
                    yield out
                    start_frame = frame + 1
                    video_frames = []


def video_transform(video, num_frames=100):
    # assuming batch size of 1
    video = [to_pil_image(frame[0]) for frame in video]
    # get 100 frames for chatgpt
    frame_idxs = get_seq_frames(len(video), num_frames)
    video = [video[i] for i in frame_idxs]
    return video


def run(projection_path: str, model_name: str, question: str):
    df = pd.read_csv("youtube/dgx_videos.csv")

    # only include videos in sample directory for testing
    samples = glob.glob("youtube/samples/*.mp4")
    samples = [s.split("/")[-1] for s in samples]
    df = df[df["file"].isin(samples)]
    ###

    dataset = VideoFramesIterator(df)
    dataloader = DataLoader(dataset, batch_size=1)

    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(
        model_name, projection_path
    )

    timer = 0.0
    for batch in dataloader:
        path = str(batch["path"][0])
        video_frames = video_transform(batch["video"])
        # print("time to load video: ", time.time() - timer)

        start_frame = int(np.round(batch["start"][0]))
        end_frame = int(np.round(batch["end"][0]))

        # timer = time.time()
        output = video_chatgpt_infer(
            video_frames,
            question,
            "video-chatgpt_v1",  # conv_mode (not sure if there is any mode)
            model,
            vision_tower,
            tokenizer,
            image_processor,
            video_token_len,
        )
        # print("time to run chatgpt: ", time.time() - timer)
        # append to related annotation file
        with open(path.replace("mp4", "chatgpt"), "a+", encoding="utf-8") as handle:
            annotation = f"{start_frame} - {end_frame}\n{output}\n"
            handle.write(annotation)

        # timer = time.time()


if __name__ == "__main__":
    sys.path.append("vgpt")
    from video_chatgpt.eval.model_utils import initialize_model, get_seq_frames
    from video_chatgpt.inference import video_chatgpt_infer

    # FLAGS
    PROJECTION_PATH = "vgpt/weights/video_chatgpt-7B.bin"
    MODEL_NAME = "vgpt/LLaVA-Lightning-7B-v1-1"
    QUESTION = "What is the gamer doing in this minecraft scene?"

    run(PROJECTION_PATH, MODEL_NAME, QUESTION)
