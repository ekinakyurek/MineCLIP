from minedojo.data import YouTubeDataset
from pytube import YouTube

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
    full=False,  # full=False for tutorial videos or
    # full=True for general gameplay videos
    download=True,  # download=True to automatically download data or
    # download=False to load data from download_dir
    download_dir=your_download_directory
    # default: "~/.minedojo". You can also manually download data from
    # https://doi.org/10.5281/zenodo.6641142 and put it in download_dir.
)

print("Number of gameplay videos in the dataset:", len(gameplay))
print("Number of tutorial videos in the dataset:", len(tutorial))

print("example of a gameplay video:", gameplay[0])
print("example of a tutorial video:", tutorial[0])


# Download a video
def download_a_video(info):
    print(info["link"])
    video = YouTube(info["link"])
    # captions = video.captions
    # print("captions", captions)
    # print("tracks", video.caption_tracks)
    # if "en" in captions:
    #     caption = captions["en"]
    #     caption = str(caption.generate_srt_captions())
    #     print("caption", caption[:100])
    print("streams", video.streams)
    print("downloading...")
    stream = video.streams.filter(resolution="360p", progressive=True).first()
    type = stream.mime_type.split("/")[-1]
    stream.download(
        output_path="videos", filename=info["id"]+f".{type}"
    )


for i in range(10):
    try:
        download_a_video(tutorial[i])
    except Exception as e:
        print(e)
        print("failed to download", tutorial[i]["link"])
