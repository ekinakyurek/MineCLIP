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
import pandas as pd


class VideoFramesIterator(torch.utils.data.IterableDataset):
    def __init__(
        self,
        annotations: pd.DataFrame,
        frame_transform=None,
        video_transform=None,
        clip_len: int = 16,
        seed: int = 42,
    ):
        super(VideoFramesIterator).__init__()

        self.annotations = annotations
        self.clip_len = clip_len
        self.frame_transform = frame_transform
        self.video_transform = video_transform
        self.epoch_size = int(self.annotations.nbframes.sum())
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        for i in range(self.epoch_size):
            # Get random sample
            sample = self.annotations.iloc[i]
            path = "youtube/samples/" + sample["file"]
            if not os.path.exists(path):
                print("path does not exist: ", path)
                continue
            nbframes = float(sample["nbframes"])
            duration = float(sample["duration"])
            fps = int(nbframes / duration)
            print("fps: ", fps)
            max_seek = duration - 1.0
            print("max_seek: ", max_seek)
            nframes_per_iter = int(np.round(1.0 * fps))
            print("nframes_per_iter", nframes_per_iter)
            for start_pts in np.arange(0, max_seek, 1.0):
                video_frames = []  # video frame buffer
                vid = VideoReader(path, "video").seek(start_pts)
                end_pts = start_pts
                print("start_frame_pts: ", start_pts)
                for frame_info in itertools.islice(vid, nframes_per_iter):
                    data = frame_info["data"]
                    if self.frame_transform:
                        data = self.frame_transform(data)
                    video_frames.append(data)
                    end_pts = frame_info["pts"]
                print("end_frame_pts: ", end_pts)
                # Stack it into a tensor
                video = torch.stack(video_frames, 0)

                if self.video_transform:
                    video = self.video_transform(video)

                out = {
                    "path": path,
                    "video": video,
                    "start": start_pts,
                    "end": end_pts,
                }
                yield out


if __name__ == "__main__":
    import sys
    sys.path.append("vgpt")
    from video_chatgpt.eval.model_utils import initialize_model, get_seq_frames
    from video_chatgpt.inference import video_chatgpt_infer
    import glob

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    df = pd.read_csv("youtube/dgx_videos.csv")
    samples = glob.glob("youtube/samples/*.mp4")
    samples = [s.split("/")[-1] for s in samples]
    #fitler df
    print(df.head())
    df = df[df["file"].isin(samples)]
    print(df.head())


    frame_transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 244), interpolation=transforms.InterpolationMode.BICUBIC
            ),
        ]
    )

    to_pil_image = transforms.ToPILImage()

    def video_transform(video, num_frames=100):
        arr = [to_pil_image(frame) for frame in video]
        frame_idxs = get_seq_frames(len(arr), num_frames)
        video = [arr[i] for i in frame_idxs]
        return video

    dataset = VideoFramesIterator(
        df, frame_transform=frame_transform
    )

    PROJECTION_PATH = "vgpt/weights/video_chatgpt-7B.bin"
    MODEL_NAME = "vgpt/LLaVA-Lightning-7B-v1-1"

    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(
        MODEL_NAME, PROJECTION_PATH
    )
    dataloader = DataLoader(dataset, batch_size=1)
    results = {}
    question = "What is the gamer doing in this minecraft scene?"
    conv_mode = "video-chatgpt_v1"



    for batch in dataloader:
        path = str(batch["path"][0])
        video_frames = video_transform(batch["video"][0])
        start_frame = int(np.round(batch["start"][0] * 30))
        end_frame = int(np.round(batch["end"][0] * 30))

        output = video_chatgpt_infer(video_frames, question, conv_mode, model, vision_tower,
                                             tokenizer, image_processor, video_token_len)

        if path not in results:
            results[path] = {}

        results[path][f"{start_frame} - {end_frame}"] = output
        with open("youtube/vgpt_annotations.json", "w") as handle:
            json.dump(results, handle)