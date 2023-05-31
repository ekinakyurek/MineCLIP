from concurrent.futures import ProcessPoolExecutor
from minedojo.data import YouTubeDataset
from tqdm import tqdm
import yt_dlp
import json

your_download_directory = "/home/data/minedojo/"
gameplay = YouTubeDataset(
    full=True,  # full=False for tutorial videos or
    # full=True for general gameplay videos
    download=True,  # download=True to automatically download data or
    # download=False to load data from download_dir
    download_dir=your_download_directory
    # default: "~/.minedojo". You can also manually download data from
    # https://doi.org/10.5281/zenodo.6641142 and put it in download_dir.
)

tutorial = YouTubeDataset(
    full=False, download=True, download_dir=your_download_directory
)

print("Number of gameplay videos in the dataset:", len(gameplay))
print("Number of tutorial videos in the dataset:", len(tutorial))

print("example of a gameplay video:", gameplay[0])
print("example of a tutorial video:", tutorial[0])


ydl_opts = {
    "format": "133+139/133+140",
    "outtmpl": {"default": "videos/%(id)s.%(ext)s"},
    "writesubtitles": True,
    "writeautomaticsub": True,
    "subtitleslangs": ["en", "zh-Hans"],
    "quiet": True,
    "retries": 10,
    "no_warnings": True,
}


def download_a_video_v2(info):
    print(info["link"])
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([info["link"]])
            return info["id"]
        except Exception as e:
            return str(e)
    return False


MAX_WORKERS = 1
executer = ProcessPoolExecutor(max_workers=MAX_WORKERS)

buffer = []
completed = []

for i in tqdm(range(3)):
    future = executer.submit(download_a_video_v2, tutorial[i])

    buffer.append(future)
    if len(buffer) >= 1 * MAX_WORKERS:
        for future in buffer:
            completed.append(future.result())
        buffer = []

if buffer:
    for future in buffer:
        completed.append(future.result())
    buffer = []


print("completed:")
print(completed)


# # Download a video
# def download_a_video(info):
#     print(info["link"])
#     video = YouTube(info["link"])
#     # captions = video.captions
#     # print("captions", captions)
#     # print("tracks", video.caption_tracks)
#     # if "en" in captions:
#     #     caption = captions["en"]
#     #     caption = str(caption.generate_srt_captions())
#     #     print("caption", caption[:100])
#     # print("streams", video.streams)
#     # print("downloading...")
#     print("captions: ", video.captions.all())
#     stream = video.streams.filter(resolution="360p", progressive=True).first()
#     type = stream.mime_type.split("/")[-1]
#     stream.download(
#         output_path="videos", filename=info["id"]+f".{type}"
#     )
#     saved_path = os.path.join("videos", info["id"]+f".{type}")
#     return saved_path

# import json
