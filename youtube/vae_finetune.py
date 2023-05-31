import os
import torch
import random
import itertools

from torch import nn

from diffusers import AutoencoderKL
from torchvision import transforms as VT
from glob import glob
from tqdm import tqdm
from torchvision.io import VideoReader

import ffmpeg
import pandas as pd

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import cv2


def get_video_info(filname):
    probe = ffmpeg.probe(filname)
    video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
    info = {
        "nbframes": video_info["nb_frames"],
        "duration": video_info["duration"],
        "width": video_info["width"],
        "height": video_info["height"],
    }
    return info


def annotate_video_folder(folder):
    # create a csv file with video info
    # for each mp4 file in the folder
    # get all mp4 files in the folder
    data = []
    for file in tqdm(glob(folder + "/*.mp4")):
        # get video info
        try:
            info = get_video_info(file)
            # add to dataframe
            filename = file.split("/")[-1]
            info_tuple = (
                filename,
                int(info["nbframes"]),
                float(info["duration"]),
                info["width"],
                info["height"],
            )
            data.append(info_tuple)
        except Exception as e:
            print(e)
            print("Error with file: ", file)
    df = pd.DataFrame(data, columns=["file", "nbframes", "duration", "width", "height"])
    return df


class RandomDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, annotations, frame_transform=None, video_transform=None, clip_len=16
    ):
        super(RandomDataset).__init__()

        self.annotations = annotations
        self.clip_len = clip_len
        self.frame_transform = frame_transform
        self.video_transform = video_transform
        self.epoch_size = len(self.annotations)

    def __iter__(self):
        for i in range(self.epoch_size):
            # Get random sample
            sample = self.annotations.sample(1, weights="nbframes", replace=False)
            sample = sample.iloc[0]
            path = "youtube/dgx_videos/" + sample["file"]
            if not os.path.exists(path):
                print("path does not exist: ", path)
                continue
            nbframes = float(sample["nbframes"])
            duration = float(sample["duration"])
            fps = int(nbframes / duration)
            max_seek = duration - (self.clip_len / fps)
            vid = VideoReader(path, "video")
            video_frames = []  # video frame buffer
            # Seek and return frames
            start = random.uniform(0.0, max_seek)
            for frame in itertools.islice(vid.seek(start), self.clip_len):
                data = frame["data"]
                if self.frame_transform:
                    data = self.frame_transform(data)
                video_frames.append(data)
                current_pts = frame["pts"]
            # Stack it into a tensor
            video = torch.stack(video_frames, 0)

            if self.video_transform:
                video = self.video_transform(video)
            output = {"path": path, "video": video, "start": start, "end": current_pts}
            yield output


class VAELoss(nn.Module):
    def __init__(self, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0):
        super().__init__()
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        # output log variance
        #self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

    def forward(self, inputs, reconstructions, posteriors):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        # nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        nll_loss = rec_loss
        weighted_nll_loss = nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss = kl_loss * self.kl_weight + weighted_nll_loss * self.pixel_weight

        return loss


class AutoencoderKLWLoss(nn.Module):
    def __init__(self, vae: AutoencoderKL, kl_weight=1.0, pixelloss_weight=1.0):
        super().__init__()
        self.vae = vae
        self.loss = VAELoss(kl_weight=kl_weight, pixelloss_weight=pixelloss_weight)

    def forward(self, x):
        posteriors = self.encode(x)
        z = posteriors.sample()
        reconstructions = self.decode(z)
        return reconstructions, posteriors

    def encode(self, x):
        return self.vae.encode(x).latent_dist

    def decode(self, z):
        return self.vae.decode(z).sample


def train(
    model,
    df,
    batch_size: int = 8,
    gaccum: int = 8,
    total_iter: int = 100,
):
    train_transforms = VT.Compose(
        [
            VT.Resize((256, 512), interpolation=VT.InterpolationMode.BICUBIC),
            lambda x: 2 * (x / 255.0) - 1.0,
        ]
    )
    frame_transform = lambda x: x
    dataset = RandomDataset(
        df,
        clip_len=1,
        video_transform=train_transforms,
        frame_transform=frame_transform,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size)
    optim = AdamW(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optim, total_iter, eta_min=1e-6)
    n_iter = 0
    total_loss = 0.0
    n_forward = 0
    model.train()
    while n_iter < total_iter:
        for data in tqdm(dataloader):
            video = data["video"].squeeze(1)
            video = video.cuda()
            reconstructions, posteriors = model(video)
            loss = model.loss(video, reconstructions, posteriors)
            loss.backward()
            total_loss += loss.item()
            n_forward += 1
            if gaccum % batch_size == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optim.step()
                optim.zero_grad()
                scheduler.step()
                n_iter += 1
                if n_iter >= total_iter:
                    break
                if n_iter % 100 == 0:
                    print("Iter: ", n_iter, "Loss: ", total_loss / n_forward)
                    model.eval()
                    with torch.no_grad():
                        reconstructions, posteriors = model(video)
                        # save reconstructed image
                        reconstructions = reconstructions.cpu().numpy()
                        reconstructions = ((reconstructions * 0.5 + 0.5) * 255.0).astype("uint8")
                        original = ((video.cpu().numpy() * 0.5 + 0.5) * 255.0).astype("uint8")
                        for i in range(reconstructions.shape[0]):
                            img = reconstructions[i].transpose(1, 2, 0)
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(f"youtube/recon_{i}.png", img)
                            img = original[i].transpose(1, 2, 0)
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(f"youtube/origin_{i}.png", img)


                    total_loss = 0.0
                    n_forward = 0


if __name__ == "__main__":
    df = pd.read_csv("youtube/dgx_videos.csv")

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        subfolder="vae",
    )

    vae = AutoencoderKLWLoss(vae, kl_weight=1.0, pixelloss_weight=1.0).cuda()


    train(vae, df, batch_size=4, gaccum=16, total_iter=50000)

    torch.save(vae.state_dict(), "youtube/vae_state_dict.pt")


