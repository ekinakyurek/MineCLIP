from typing import List
from datetime import datetime
import os
import re

import numpy as np

from gpt.src.api import run_pipeline
from gpt.src.specification import APIArgs
from gpt.src.specification import EndPointSpec
from gpt.src.specification import OpenAIEndPointArgs


TIMEFINDER = re.compile(r"\d{2}:\d{2}:\d{2}.\d{3}")
SUBFRAMETITLEFINDER = re.compile(r"<c>.*?</c>")
SUBFRAMENUMFINDER = re.compile(r"<\d+>")


def compressed_format(subtitles: str, fps: int = 30, vtt: bool = True) -> str:
    if vtt:
        # remove headers
        subtitles = "\n".join(subtitles.split("\n")[3:])
    else:
        new_subtitles = []
        old_subtitles = subtitles.split("\n")
        for index, line in enumerate(old_subtitles):
            if index + 1 < len(old_subtitles):
                if "-->" in old_subtitles[index + 1]:
                    continue
                else:
                    new_subtitles.append(line)
            else:
                new_subtitles.append(line)

        subtitles = "\n".join(new_subtitles)

    for match in TIMEFINDER.finditer(subtitles):
        timestring = match.group(0)
        try:
            pt = datetime.strptime(timestring, "%H:%M:%S.%f")
        except:
            pt = datetime.strptime(timestring, "%H:%M:%S,%f")

        total_seconds = (
            pt.second + pt.minute * 60 + pt.hour * 3600
        ) + pt.microsecond / 999999

        frame = str(int(np.round(total_seconds * fps)))
        subtitles = subtitles.replace(timestring, frame)

    for match in SUBFRAMETITLEFINDER.finditer(subtitles):
        # remove the <c> and </c> tags
        timestring = match.group(0)
        subtitles = subtitles.replace(timestring, timestring[3:-4])

    subtitles = subtitles.replace("-->", "-").strip()
    subtitles = subtitles.replace("align:start position:0%", "").strip()

    for match in SUBFRAMENUMFINDER.finditer(subtitles):
        timestring = match.group(0)
        subtitles = subtitles.replace(timestring, "")

    output = []
    for index, line in enumerate(subtitles.split("\n")):
        if line.strip():
            output.append(line)

    if vtt:
        reduced_output = []
        for index, line in enumerate(output):
            if index < 2:
                reduced_output.append(line)
            else:
                if line.strip() in output[index - 2]:
                    continue
                else:
                    reduced_output.append(line)
        output = reduced_output

        reduced_output = []
        for index, line in enumerate(output):
            if index < 1:
                reduced_output.append(line)
            else:
                if " - " in output[index - 1] and " - " in line:
                    last_line = reduced_output.pop()
                    start_frame = last_line.split(" - ")[0]
                    end_frame = line.split(" - ")[1]
                    reduced_output.append(f"{start_frame} - {end_frame}")
                else:
                    reduced_output.append(line)
        output = reduced_output

    output = "\n".join(output)

    return output


def partition(input_subtitle: str, output_subtitle: str, parts: int = 3):
    inputs = np.array(input_subtitle.split("\n"))
    outputs = np.array(output_subtitle.split("\n"))

    input_frame_lines = np.arange(0, len(inputs), 2)
    output_frame_lines = np.arange(0, len(outputs), 2)

    num_frames = len(input_frame_lines) // parts

    response = []
    for i in range(parts):
        if i == parts - 1:
            input_part_frame_lines = input_frame_lines[i * num_frames :]
        else:
            input_part_frame_lines = input_frame_lines[
                i * num_frames : (i + 1) * num_frames
            ]

        input_part_start_frame, _ = inputs[input_part_frame_lines[0]].split(" - ")
        input_part_start_frame = int(input_part_start_frame)

        _, input_part_end_frame = inputs[input_part_frame_lines[-1]].split(" - ")
        input_part_end_frame = int(input_part_end_frame)

        output_part = []
        for line_no in output_frame_lines:
            output_part_start_frame, output_part_end_frame = outputs[line_no].split(
                " - "
            )
            output_part_start_frame = int(output_part_start_frame)
            output_part_end_frame = int(output_part_end_frame)

            if (
                output_part_end_frame > input_part_start_frame
                and output_part_start_frame < input_part_end_frame
            ):
                output_part.append(line_no)

        output_line_start = output_part[0]
        output_line_end = output_part[-1]

        output_part_str = "\n".join(outputs[output_line_start : output_line_end + 2])

        input_line_start = input_part_frame_lines[0]
        input_line_end = input_part_frame_lines[-1]
        input_part_str = "\n".join(inputs[input_line_start : input_line_end + 2])

        response.append((input_part_str, output_part_str))
    return response


def partition_input(input_subtitle: str, parts: int = 3):
    inputs = np.array(input_subtitle.split("\n"))
    input_frame_lines = np.arange(0, len(inputs), 2)
    num_frames = len(input_frame_lines) // parts
    response = []
    for i in range(parts):
        if i == parts - 1:
            input_part_frame_lines = input_frame_lines[i * num_frames :]
        else:
            input_part_frame_lines = input_frame_lines[
                i * num_frames : (i + 1) * num_frames
            ]

        input_line_start = input_part_frame_lines[0]
        input_line_end = input_part_frame_lines[-1]
        input_part_str = "\n".join(inputs[input_line_start : input_line_end + 2])
        response.append(input_part_str)
    return response


def get_pipeline(model: str = "gpt-3.5-turbo-16k", max_input_tokens: int = 3000):
    def postprocesser(inputs, outputs):
        response = []
        for inp, out in zip(inputs, outputs):
            del inp
            new_output = []
            for line in out["text"].split("\n"):
                if line.strip():
                    new_output.append(line)
            new_output = "\n".join(new_output)
            response.append(new_output)
        return response

    spec = EndPointSpec(
        name="subtitle_generator",
        template=(
            "You're a smart assistant. I want you to tell me what a gamer is"
            " performing in a minecraft video by reading its subtitles. To achieve"
            " this, you need to convert the subtitles to detailed descriptions by"
            " inferring the actions and the goals of the gamer. Please also mention the"
            " names of used or crafted items if there are. \nHere is an example"
            " conversion:\nExample:\n\n\n--Subtitle--\n{input_subtitle}\n--Formatted"
            " Description--\n{output_subtitle}\n--DONE--\n\n\nHere is the query"
            " subtitle:\nQuery:\n\\n\n--Subtitle--\n{query_subtitle}\n--Formatted"
            " Description--\n"
        ),
        args=APIArgs(
            OpenAIEndPointArgs(model=model, stop=["--DONE--"]),
            max_input_tokens=max_input_tokens,
            truncate_input=True,
        ),
        postprocesser=postprocesser,
    )

    return [spec]


def merge_parts(outputs: List[str], parts: int = 4):
    response = []
    for i in range(len(outputs) // parts):
        response.append("\n".join(outputs[i * parts : (i + 1) * parts]))
    return response


if __name__ == "__main__":
    import glob

    import openai

    api_keys = os.environ.get("OPENAI_API_KEY").split(",")
    openai.organization = None

    async def convert(
        vtt_folder: str = "samples",
        model: str = "gpt-3.5",
        max_input_tokens: int = 3000,
        parts: int = 4,
        max_samples: int = 5,
    ):
        example_subtitle = compressed_format(
            open("samples/-xtrtbYT8wA.en.vtt", "r", encoding="utf-8").read()
        )

        example_parse = compressed_format(
            open(
                "samples/-xtrtbYT8wA.en.vtt.annotated.srt", "r", encoding="utf-8"
            ).read(),
            vtt=False,
        )

        example_parts = partition(example_subtitle, example_parse, parts=parts)

        inputs = []
        files = []

        for file in glob.glob(f"{vtt_folder}/*.en.vtt"):
            if "xtrtbYT8wA" not in file:  # skip the example
                query_subtitle = compressed_format(
                    open(file, "r", encoding="utf-8").read()
                )
                query_parts = partition_input(query_subtitle, parts=parts)
                files.append(file)
                for i, (example_input, example_output) in enumerate(example_parts):
                    inputs.append(
                        {
                            "input_subtitle": example_input,
                            "output_subtitle": example_output,
                            "query_subtitle": query_parts[i],
                        }
                    )

                if len(files) >= max_samples:
                    break

        pipeline = get_pipeline(model=model, max_input_tokens=max_input_tokens)
        outputs = await run_pipeline(inputs, pipeline, api_keys=api_keys)
        input_strings = [inp["query_subtitle"] for inp in inputs]
        inputs = merge_parts(input_strings, parts=parts)
        outputs = merge_parts(outputs, parts=parts)

        return inputs, outputs

    import asyncio

    # run the script
    inputs, outputs = asyncio.run(
        convert(
            vtt_folder="samples", model="gpt-3.5-turbo", max_input_tokens=3000, parts=4
        )
    )

    breakpoint()
