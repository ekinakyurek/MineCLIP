import asyncio
from datetime import datetime
import os
import re
import glob
import openai
import numpy as np
import json

from gpt.src.api import run_pipeline
from gpt.src.specification import APIArgs
from gpt.src.specification import EndPointSpec
from gpt.src.specification import OpenAIEndPointArgs


OUTPUT_FRAME_FINDER = re.compile(r"\d+ - \d+\n")
INPUT_FRAME_FINDER = re.compile(r".*-->.*\n")
TIME_FINDER = re.compile(r"\d{2}:\d{2}:\d{2}.\d{3}")
CTAGS_FINDER = re.compile(r"<\/c>|<c>|<\d{2}:\d{2}:\d{2}.\d{3}>")
NUMBER_FINDER = re.compile(r"\d+")
SUBFRAMETITLEFINDER = re.compile(r"<c>.*?</c>")
SUBFRAMENUMFINDER = re.compile(r"<\d+>")


def parse(
    text: str,
    frame_finder=INPUT_FRAME_FINDER,
    frame_split=NUMBER_FINDER,
    frame_converter=None,
):
    keys = []
    values = []
    end = 0

    for match in frame_finder.finditer(text):
        current_start, current_end = match.span()
        star_frame, end_frame = frame_split.findall(match.group().strip())
        if frame_converter:
            start_frame = frame_converter(star_frame)
            end_frame = frame_converter(end_frame)
        keys.append((start_frame, end_frame))
        if end > 0:
            lines = text[end:current_start].split("\n")
            nonempty_lines = [line.strip() for line in lines if line.strip()]
            values.append(nonempty_lines)
        end = current_end

    lines = text[end:].split("\n")
    nonempty_lines = [line.strip() for line in lines if line.strip()]
    values.append(nonempty_lines)
    assert len(keys) == len(values), f"{len(keys)} != {len(values)}\n{text}"
    return keys, values


def convert_timestring_to_frame(timestring: str, fps=30):
    try:
        pt = datetime.strptime(timestring, "%H:%M:%S.%f")
    except:
        pt = datetime.strptime(timestring, "%H:%M:%S,%f")

    total_seconds = (
        pt.second + pt.minute * 60 + pt.hour * 3600
    ) + pt.microsecond / 999999

    frame = int(np.round(total_seconds * fps))
    return frame


def parse_input_subtitles(path: str):
    text = open(path, "r", encoding="utf-8").read()
    text = CTAGS_FINDER.sub("", text)
    keys, values = parse(
        text,
        frame_finder=INPUT_FRAME_FINDER,
        frame_split=TIME_FINDER,
        frame_converter=convert_timestring_to_frame,
    )
    assert len(keys) == len(values)
    prev_value_lines = None
    start_to_be_added = None
    new_keys = []
    new_values = []
    for key, value_lines in zip(keys, values):
        start, end = new_key = key
        if prev_value_lines is not None:
            new_value_lines = [
                line for line in value_lines if line not in prev_value_lines
            ]
        else:
            new_value_lines = value_lines

        if len(new_value_lines) == 0 and start_to_be_added is None:
            start_to_be_added = start
        elif len(new_value_lines) > 0:
            if start_to_be_added is not None:
                new_key = (start_to_be_added, new_key[1])
                start_to_be_added = None

            new_keys.append(new_key)
            new_values.append(new_value_lines)
            prev_value_lines = value_lines
    return new_keys, new_values


def parsed_to_string(keys, values):
    return "\n".join(
        [
            str(key[0]) + " - " + str(key[1]) + "\n" + "\n".join(value)
            for key, value in zip(keys, values)
        ]
    )


def parse_output_subtitles(path):
    text = open(path, "r", encoding="utf-8").read()
    keys, values = parse(
        text,
        frame_finder=OUTPUT_FRAME_FINDER,
        frame_split=NUMBER_FINDER,
        frame_converter=int,
    )
    return (keys, values)


def get_parts(parsed_input, parsed_output=None, parts=4):
    input_lengths = np.cumsum([len(value) for value in parsed_input[1]])

    break_positions = np.round(np.linspace(0, input_lengths[-1], parts+1, endpoint=True)).astype(int)

    response = []
    input_start_index = 0
    output_start_index = 0

    for poisition in break_positions[1:]:
        # find the index in input_lengths that is closest to position
        input_end_index = np.argmin(np.abs(input_lengths - poisition))

        inp = (
            parsed_input[0][input_start_index : input_end_index + 1],
            parsed_input[1][input_start_index : input_end_index + 1],
        )

        if parsed_output is not None:
            inp_end_frame_no = parsed_input[0][input_end_index][1]
            inp_start_frame_no = parsed_input[0][input_start_index][0]

            output_end_index = None

            for index, (start, end) in enumerate(parsed_output[0][output_start_index:]):
                index = index + output_start_index

                if end > inp_start_frame_no and start < inp_end_frame_no:
                    output_end_index = index
                elif start > inp_end_frame_no:
                    break

            if output_end_index is None:
                output_end_index = len(parsed_output[0]) - 1

            output_end_index += 1
            if output_end_index - output_start_index  == 0:
                out = (
                    parsed_output[0][output_end_index - 1 : output_end_index],
                    parsed_output[1][output_end_index - 1 : output_end_index],
                )
            else:
                out = (
                    parsed_output[0][output_start_index : output_end_index],
                    parsed_output[1][output_start_index : output_end_index],
                )

            output_start_index = output_end_index

            response.append((parsed_to_string(*inp), parsed_to_string(*out)))
        else:
            response.append(parsed_to_string(*inp))

        input_start_index = input_end_index + 1

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
            "You're a smart assistant. I want you to tell me what a gamer is performing"
            " in a minecraft video by reading its subtitles. To achieve this, you need"
            " to convert the subtitles to detailed annotations by inferring the actions"
            " and the goals of the gamer. Please also mention the names of used or"
            " crafted items if there are. \nHere is an example of annotation that I"
            " want:\nExample:\n\n\n-- Subtitle --\n{input_subtitle}\n-- Formatted"
            " Annotation --\n{output_subtitle}\n--DONE--\n\n\nHere is the query subtitle"
            " that you need to"
            " convert:\nQuery:\n\\n\n-- Subtitle --\n{query_subtitle}\n-- Formatted"
            " Annotation --\n"
        ),
        args=APIArgs(
            OpenAIEndPointArgs(model=model, stop=["--DONE--"]),
            max_input_tokens=max_input_tokens,
            truncate_input=True,
        ),
        postprocesser=postprocesser,
    )

    return [spec]


def merge_parts(inputs, outputs):
    input_strings = []
    output_strings = []
    annotation_files = []

    current_part_index = -1

    current_input_parts = []
    current_output_parts = []
    last_file = None

    for i in range(len(inputs)):
        inp = inputs[i]
        if inp['part_index'] <= current_part_index or i == len(inputs)-1:

            current_input = "\n".join(current_input_parts)
            current_output = "\n".join(current_output_parts)
            current_file = last_file

            input_strings.append(current_input)
            output_strings.append(current_output)
            annotation_files.append(current_file)

            current_input_parts = []
            current_output_parts = []


        current_part_index = inp['part_index']
        last_file = inp['id']
        current_input_parts.append(inp['query_subtitle'])
        current_output_parts.append(outputs[i])


    return input_strings, output_strings, annotation_files


if __name__ == "__main__":
    api_keys = os.environ.get("OPENAI_API_KEY").split(",")
    openai.organization = None
    from gpt.src.tokenizer import GPTTokenizer

    async def convert(
        vtt_folder: str = "samples",
        model: str = "gpt-3.5-turbo",
        max_input_tokens: int = 3000,
        parts: int = 4,
        dynamic: bool = False,
        max_samples: int = 100,
        cache_path: str = None,
    ):

        tokenizer = GPTTokenizer(model=model)

        example_subtitle = parse_input_subtitles("samples/-xtrtbYT8wA.en.vtt")

        example_parse = parse_input_subtitles(
            "samples/-xtrtbYT8wA.en.vtt.annotated.srt"
        )
        if dynamic:
            input_length = parsed_to_string(*example_subtitle) + parsed_to_string(*example_parse)
        else:
            example_parts = get_parts(example_subtitle, example_parse, parts=min(parts, 3))

        inputs = []
        files = []
        print("vtt_folder", vtt_folder)
        for file in glob.glob(f"{vtt_folder}/*.en.vtt")[:5]:
            if "xtrtbYT8wA" not in file:  # skip the example
                if len(open(file, "r", encoding="utf-8").read().split("\n")) < 10:
                    continue

                query_subtitle = parse_input_subtitles(file)
                if dynamic:
                    query_length = parsed_to_string(*query_subtitle)
                    token_count = tokenizer.token_count(input_length + query_length) + 127
                    parts = int(token_count // max_input_tokens) + 1
                    query_parts = get_parts(query_subtitle, parts=parts)
                    example_parts = get_parts(example_subtitle, example_parse, parts=min(parts, 3))
                else:
                    query_parts = get_parts(query_subtitle, parts=parts)


                files.append(file)
                for i, query_part in enumerate(query_parts):
                    example_input, example_output = example_parts[min(i, len(example_parts)-1)]
                    inputs.append(
                        {
                            "input_subtitle": example_input,
                            "output_subtitle": example_output,
                            "query_subtitle": query_part,
                            "id": file.split("/")[-1].replace(".en.vtt", ""),
                            "total_parts": parts,
                            "part_index": i,
                        }
                    )


                if len(files) >= max_samples:
                    break

        pipeline = get_pipeline(model=model, max_input_tokens=max_input_tokens)

        outputs = await run_pipeline(
            inputs, pipeline, api_keys=api_keys, cache_path=cache_path
        )

        inputs, outputs, files = merge_parts(inputs, outputs)

        return inputs, outputs, files

    # run the script

    MODEL = "gpt-3.5-turbo"
    MAX_INPUT_TOKENS = 2048
    CACHE_PATH = "cache.pkl"

    inputs, outputs, files = asyncio.run(
        convert(
            vtt_folder="dgx_videos/",
            model=MODEL,
            max_input_tokens=MAX_INPUT_TOKENS,
            dynamic=True,
            cache_path=CACHE_PATH,
        )
    )

    # save the results
    with open(
        f"10_annotations_v2_{MODEL}_{MAX_INPUT_TOKENS}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump({"inputs": inputs, "outputs": outputs, "files": files}, f, indent=4)

    import shutil

    for index, (file, output) in enumerate(zip(files, outputs)):
        with open(f"samplesv4/{file}.gpt35", "w", encoding="utf-8") as f:
            f.write(output)
        if os.path.exists(f"dgx_videos/{file}.mp4"):
            shutil.copy2(f"dgx_videos/{file}.mp4", f"samplesv4/{file}.mp4")
