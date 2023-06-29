import glob
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

import pandas as pd


import decord
from decord import cpu
from decord import VideoReader
decord.bridge.set_bridge("torch")


class VideoFramesIterator(torch.utils.data.IterableDataset):
    def __init__(
        self,
        annotations: pd.DataFrame,
        frame_transform=None,
        nframes_per_iteration: int = 29,
        nframes_per_video: int = 100,
        seed: int = 42,
    ):
        super(VideoFramesIterator).__init__()

        self.annotations = annotations
        self.nframes_per_iteration = nframes_per_iteration
        self.nframes_per_video = nframes_per_video
        self.frame_transform = frame_transform
        self.epoch_size = int(self.annotations.nbframes.sum() / nframes_per_iteration)
        # self.file_boundaries = self.annotations.nbframes.cumsum().values
        self.seed = seed
        self.nframes_per_iteration = nframes_per_iteration
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        for i in range(len(self.annotations)):
            # Get random sample
            # find which file we are at
            # file_idx = np.searchsorted(self.file_boundaries, i)
            sample = self.annotations.iloc[i]
            path = "youtube/samples/" + sample["file"]

            if not os.path.exists(path):
                print("path does not exist: ", path)
                continue

            vr = VideoReader(path, ctx=cpu(0), width=224, height=224)
            current_frame = 0
            start_frame = 0
            video_frames = []
            skip_frames = (self.nframes_per_iteration // self.nframes_per_video) - 1
            if skip_frames > 0:
                while True:
                    try:
                        frame = vr.next()
                    except StopIteration:
                        break

                    current_frame += 1

                    frame = frame.permute(2, 0, 1)
                    if self.frame_transform is not None:
                        frame = self.frame_transform(frame)
                    video_frames.append(frame)

                    vr.skip_frames(skip_frames)
                    current_frame += skip_frames

                    if len(video_frames) == self.nframes_per_video:
                        out = {
                            "path": path,
                            "video": video_frames,
                            "start": start_frame,
                            "end": current_frame,
                        }
                        yield out
                        start_frame = current_frame
                        video_frames = []
            else:
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


def run(
    projection_path: str,
    model_name: str,
    question: str,
    nframes_per_iteration: int,
    exp_folder: str,
    temp: float = 0.2,
    max_new_tokens: int = 256,
):
    df = pd.read_csv("youtube/dgx_videos.csv")

    # only include videos in sample directory for testing
    samples = glob.glob("youtube/samples/*.mp4")
    samples = [s.split("/")[-1] for s in samples]
    df = df[df["file"].isin(samples)]
    ###

    print("Dataset loaded")

    dataset = VideoFramesIterator(df, nframes_per_iteration=nframes_per_iteration)
    dataloader = DataLoader(dataset, batch_size=1)

    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(
        model_name, projection_path
    )

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
            temp=temp,
            max_new_tokens=max_new_tokens,
        )
        # print("time to run chatgpt: ", time.time() - timer)
        # append to related annotation file
        filename = path.split("/")[-1].replace(
            "mp4", f"chatgpt_{nframes_per_iteration}"
        )
        file = os.path.join(exp_folder, filename) + ".txt"
        with open(file, "a+", encoding="utf-8") as handle:
            annotation = f"{start_frame} - {end_frame}\n{output}\n"
            handle.write(annotation)

        # timer = time.time()


if __name__ == "__main__":
    print("Starting the script")
    sys.path.append("vgpt")
    import argparse

    from video_chatgpt.eval.model_utils import get_seq_frames
    from video_chatgpt.eval.model_utils import initialize_model
    from video_chatgpt.inference import video_chatgpt_infer

    print("Adding parser")
    parser = argparse.ArgumentParser(description="Get GPT annotations for videos")
    parser.add_argument(
        "--projection_path",
        type=str,
        default="vgpt/weights/video_chatgpt-7B.bin",
        help="video chatgpt weights",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="vgpt/LLaVA-Lightning-7B-v1-1",
        help="video chatgpt model name",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="What is the gamer doing in this minecraft scene?",
        help="question to ask chatgpt",
    )
    parser.add_argument(
        "--nframes_per_iteration",
        type=int,
        default=1000,
        help="number of frames to process per iteration",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.2,
        help="temperature for chatgpt",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="max new tokens for chatgpt",
    )
    args = parser.parse_args()
    print("Args parsed")

    question_hash = args.question.replace(" ", "_").replace("?", "")
    question_hast = question_hash[:64]
    exp_folder = f"youtube/samples/{question_hash}/"
    os.makedirs(exp_folder, exist_ok=True)
    print("folder created")

    run(
        args.projection_path,
        args.model_name,
        args.question,
        args.nframes_per_iteration,
        exp_folder,
        temp=args.temp,
        max_new_tokens=args.max_new_tokens,
    )
