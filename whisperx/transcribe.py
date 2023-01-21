import sys
import csv

# Whisper
import whisper
import torch


#import argparse
#import os
import warnings
from typing import List, Optional, Tuple, Union, Iterator, TYPE_CHECKING

import numpy as np
#import torch

# Alignment
import torchaudio
from transformers import AutoProcessor, Wav2Vec2ForCTC

#these are all whisper things
#from audio import SAMPLE_RATE, N_FRAMES, HOP_LENGTH, pad_or_trim, log_mel_spectrogram, load_audio
from alignment import get_trellis, backtrack, merge_repeats, merge_words
#from decoding import DecodingOptions, DecodingResult
#from tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer
#from utils import exact_div, format_timestamp, optional_int, optional_float, str2bool, write_txt, write_vtt, write_srt, write_ass, write_csv_words

# Constant variables
from whisper.utils import format_timestamp
from whisper.audio import SAMPLE_RATE



if TYPE_CHECKING:
    from model import Whisper

LANGUAGES_WITHOUT_SPACES = ["ja", "zh"]

DEFAULT_ALIGN_MODELS_TORCH = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "fr": "VOXPOPULI_ASR_BASE_10K_FR",
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    "es": "VOXPOPULI_ASR_BASE_10K_ES",
    "it": "VOXPOPULI_ASR_BASE_10K_IT",
}

DEFAULT_ALIGN_MODELS_HF = {
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "uk": "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
}

# Logs
import logging
logger = logging.getLogger("whisperx")

def align(
    transcript: Iterator[dict],
    model: torch.nn.Module,
    align_model_metadata: dict,
    audio: Union[str, np.ndarray, torch.Tensor],
    device: str,
    extend_duration: float = 0.0,
    start_from_previous: bool = True,
    drop_non_aligned_words: bool = False,
):
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = whisper.audio.load_audio(audio)
        audio = torch.from_numpy(audio)
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)

    MAX_DURATION = audio.shape[1] / SAMPLE_RATE

    model_dictionary = align_model_metadata['dictionary']
    model_lang = align_model_metadata['language']
    model_type = align_model_metadata['type']

    prev_t2 = 0
    word_segments_list = []
    for idx, segment in enumerate(transcript):
        # first we pad
        t1 = max(segment['start'] - extend_duration, 0)
        t2 = min(segment['end'] + extend_duration, MAX_DURATION)

        # use prev_t2 as current t1 if it's later
        if start_from_previous and t1 < prev_t2:
            t1 = prev_t2

        # check if timestamp range is still valid
        if t1 >= MAX_DURATION:
            print("Failed to align segment: original start time longer than audio duration, skipping...")
            continue
        if t2 - t1 < 0.02:
            print("Failed to align segment: duration smaller than 0.02s time precision")
            continue

        f1 = int(t1 * SAMPLE_RATE)
        f2 = int(t2 * SAMPLE_RATE)

        waveform_segment = audio[:, f1:f2]

        with torch.inference_mode():
            if model_type == "torchaudio":
                emissions, _ = model(waveform_segment.to(device))
            elif model_type == "huggingface":
                emissions = model(waveform_segment.to(device)).logits
            else:
                raise NotImplementedError(f"Align model of type {model_type} not supported.")
            emissions = torch.log_softmax(emissions, dim=-1)

        emission = emissions[0].cpu().detach()
        transcription = segment['text'].strip()
        if model_lang not in LANGUAGES_WITHOUT_SPACES:
            t_words = transcription.split(' ')
        else:
            t_words = [c for c in transcription]

        t_words_clean = [''.join([w for w in word if w.lower() in model_dictionary.keys()]) for word in t_words]
        t_words_nonempty = [x for x in t_words_clean if x != ""]
        t_words_nonempty_idx = [x for x in range(len(t_words_clean)) if t_words_clean[x] != ""]
        segment['word-level'] = []

        fail_fallback = False
        if len(t_words_nonempty) > 0:
            transcription_cleaned = "|".join(t_words_nonempty).lower()
            tokens = [model_dictionary[c] for c in transcription_cleaned]
            trellis = get_trellis(emission, tokens)
            path = backtrack(trellis, emission, tokens)
            if path is None:
                print("Failed to align segment: backtrack failed, resorting to original...")
                fail_fallback = True
            else:
                segments = merge_repeats(path, transcription_cleaned)
                word_segments = merge_words(segments)
                ratio = waveform_segment.size(0) / (trellis.size(0) - 1)

                duration = t2 - t1
                local = []
                t_local = [None] * len(t_words)
                for wdx, word in enumerate(word_segments):
                    t1_ = ratio * word.start
                    t2_ = ratio * word.end
                    local.append((t1_, t2_))
                    t_local[t_words_nonempty_idx[wdx]] = (t1_ * duration + t1, t2_ * duration + t1)
                t1_actual = t1 + local[0][0] * duration
                t2_actual = t1 + local[-1][1] * duration

                segment['start'] = t1_actual
                segment['end'] = t2_actual
                prev_t2 = segment['end']

                # for the .ass output
                for x in range(len(t_local)):
                    curr_word = t_words[x]
                    curr_timestamp = t_local[x]
                    if curr_timestamp is not None:
                        segment['word-level'].append({"text": curr_word, "start": curr_timestamp[0], "end": curr_timestamp[1]})
                    else:
                        segment['word-level'].append({"text": curr_word, "start": None, "end": None})

                # for per-word .srt ouput
                # merge missing words to previous, or merge with next word ahead if idx == 0
                found_first_ts = False
                for x in range(len(t_local)):
                    curr_word = t_words[x]
                    curr_timestamp = t_local[x]
                    if curr_timestamp is not None:
                        word_segments_list.append({"text": curr_word, "start": curr_timestamp[0], "end": curr_timestamp[1]})
                        found_first_ts = True
                    elif not drop_non_aligned_words:
                        # then we merge
                        if not found_first_ts:
                            t_words[x+1] = " ".join([curr_word, t_words[x+1]])
                        else:
                            word_segments_list[-1]['text'] += ' ' + curr_word
        else:
            fail_fallback = True

        if fail_fallback:
            # then we resort back to original whisper timestamps
            # segment['start] and segment['end'] are unchanged
            prev_t2 = 0
            segment['word-level'].append({"text": segment['text'], "start": segment['start'], "end":segment['end']})
            word_segments_list.append({"text": segment['text'], "start": segment['start'], "end":segment['end']})

        print(f"[{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}] {segment['text']}")

    return {"segments": transcript, "word_segments": word_segments_list}

def load_align_model(language_code, device, model_name=None):
    if model_name is None:
        # use default model
        if language_code in DEFAULT_ALIGN_MODELS_TORCH:
            model_name = DEFAULT_ALIGN_MODELS_TORCH[language_code]
        elif language_code in DEFAULT_ALIGN_MODELS_HF:
            model_name = DEFAULT_ALIGN_MODELS_HF[language_code]
        else:
            print(f"There is no default alignment model set for this language ({language_code}).\
                Please find a wav2vec2.0 model finetuned on this language in https://huggingface.co/models, then pass the model name in --align_model [MODEL_NAME]")
            raise ValueError(f"No default align-model for language: {language_code}")

    if model_name in torchaudio.pipelines.__all__:
        pipeline_type = "torchaudio"
        bundle = torchaudio.pipelines.__dict__[model_name]
        align_model = bundle.get_model().to(device)
        labels = bundle.get_labels()
        align_dictionary = {c.lower(): i for i, c in enumerate(labels)}
    else:
        try:
            processor = AutoProcessor.from_pretrained(model_name)
            align_model = Wav2Vec2ForCTC.from_pretrained(model_name)
        except Exception as e:
            print(e)
            print(f"Error loading model from huggingface, check https://huggingface.co/models for finetuned wav2vec2.0 models")
            raise ValueError(f'The chosen align_model "{model_name}" could not be found in huggingface (https://huggingface.co/models) or torchaudio (https://pytorch.org/audio/stable/pipelines.html#id14)')
        pipeline_type = "huggingface"
        align_model = align_model.to(device)
        labels = processor.tokenizer.get_vocab()
        align_dictionary = {char.lower(): code for char,code in processor.tokenizer.get_vocab().items()}

    align_metadata = {"language": language_code, "dictionary": align_dictionary, "type": pipeline_type}

    return align_model, align_metadata

# def write_vtt_words(transcript, file):
#     print("WEBVTT\n", file=file)
#     for segment in transcript:
#         print(
#             f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
#             f"{segment['text'].strip().replace('-->', '->')}\n",
#             file=file,
#             flush=True,
#         )

# def write_srt_words(transcript, file):
#     i = 1
#     for segment in transcript:
#         #for word in segment["words"]:
#         print(
#             f"{i}\n"
#             f"{format_timestamp(segment['start'], always_include_hours=True, decimal_marker=',')} --> "
#             f"{format_timestamp(segment['end'], always_include_hours=True, decimal_marker=',')}\n"
#             f"{segment['text']}\n",
#             file=file,
#             flush=True,
#         )
#         i += 1

def write_csv(transcript, file):
    writer = csv.writer(file)
    #print(transcript)
    for segment in transcript:
        writer.writerow([segment['text'].strip(), segment['start'], segment['end']])

# def write_csv_words(transcript, file):
#     writer = csv.writer(file)
#     for segment in transcript:
#         #for word in segment["words"]:
#         writer.writerow([segment['text'], segment['start'], segment['end']])

def cli():
    import os
    import argparse
    import json

    from whisper.utils import str2bool, optional_float, optional_int, write_txt, write_srt, write_vtt


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio", nargs="+", type=str, help="Audio file(s) to transcribe")
    parser.add_argument("--model", "-m", default="small", choices=whisper.available_models(), help="name of the Whisper model to use")
    parser.add_argument("--model_dir", type=str, default=None, help="The path to save model files; uses ~/.cache/whisper by default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for PyTorch inference")
    parser.add_argument("--output_dir", "-o", type=str, default=".", help="Directory to save the outputs")
    parser.add_argument("--verbose", "-v", type=str2bool, default=True, help="Whether to print out the progress and debug messages of Whisper")

    #output file options
    #parser.add_argument("--punctuations", default=True, help="Whether to include punctuations within the words", type=str2bool)
    parser.add_argument("--segment_out", "-os", default=True, help="Whether to save files with complete segments", type=str2bool)
    parser.add_argument("--word_out", "-ow", default=True, help="Whether to save in simple text format", type=str2bool)
    
    parser.add_argument("--all", default=False, help="Whether to save in all formats", type=str2bool)
    parser.add_argument("--ass", default=False, help="Whether to save in ASS format", type=str2bool)
    parser.add_argument("--csv", default=False, help="Whether to save in CSV format", type=str2bool)
    parser.add_argument("--json", default=False, help="Whether to save in JSON format", type=str2bool)
    parser.add_argument("--srt", default=True, help="Whether to save in SRT format", type=str2bool)
    parser.add_argument("--txt", default=True, help="Whether to save in simple text format", type=str2bool)
    parser.add_argument("--vtt", default=True, help="Whether to save in VTT format", type=str2bool)  

    # alignment params
    parser.add_argument("--align_model", default=None, help="Name of phoneme-level ASR model to do alignment")
    parser.add_argument("--align_extend", default=2, type=float, help="Seconds before and after to extend the whisper segments for alignment")
    parser.add_argument("--align_from_prev", default=True, type=bool, help="Whether to clip the alignment start time of current segment to the end time of the last aligned word of the previous segment")
    parser.add_argument("--drop_non_aligned", action="store_true", help="For word .srt, whether to drop non aliged words, or merge them into neighbouring.")

    #parser.add_argument("--output_type", default="srt", choices=['all', 'srt', 'vtt', 'txt', 'ass', 'csv'], help="File type for desired output save")

    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", "-l", type=str, default=None, choices=sorted(whisper.tokenizer.LANGUAGES.keys()) + sorted([k.title() for k in whisper.tokenizer.TO_LANGUAGE_CODE.keys()]), help=f"Language to use. Among : {', '.join(sorted(k+'('+v+')' for k,v in whisper.tokenizer.LANGUAGES.items()))}.")

    parser.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
    parser.add_argument("--best_of", type=optional_int, default=5, help="number of candidates when sampling with non-zero temperature")
    parser.add_argument("--beam_size", type=optional_int, default=5, help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=None, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--length_penalty", type=float, default=None, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")

    parser.add_argument("--suppress_tokens", type=str, default="-1", help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    parser.add_argument("--initial_prompt", type=str, default=None, help="optional text to provide as a prompt for the first window.")
    parser.add_argument("--condition_on_previous_text", type=str2bool, default=False, help="If True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")
    parser.add_argument("--fp16", type=str2bool, default=True, help="whether to perform inference in fp16; True by default")

    parser.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
    parser.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4, help="If the gzip compression ratio is higher than this value, treat the decoding as failed")
    parser.add_argument("--logprob_threshold", type=optional_float, default=-1.0, help="If the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6, help="If the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")
    parser.add_argument("--threads", type=optional_int, default=0, help="Number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")

    parser.add_argument('--debug', "-d", help="Print some debug information for word alignement", default=False, action="store_true")


    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    model_dir: str = args.pop("model_dir")
    output_dir: str = args.pop("output_dir")
    #output_type: str = args.pop("output_type")
    device: str = args.pop("device")

    align_model: str = args.pop("align_model")
    align_extend: float = args.pop("align_extend")
    align_from_prev: bool = args.pop("align_from_prev")
    drop_non_aligned: bool = args.pop("drop_non_aligned")

    os.makedirs(output_dir, exist_ok=True)

    if model_name.endswith(".en") and args["language"] not in {"en", "English"}:
        if args["language"] is not None:
            warnings.warn(f"{model_name} is an English-only model but receipted '{args['language']}'; using English instead.")
        args["language"] = "en"

    temperature = args.pop("temperature")
    temperature_increment_on_fallback = args.pop("temperature_increment_on_fallback")
    if temperature_increment_on_fallback is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback))
    else:
        temperature = [temperature]

    threads = args.pop("threads")
    if threads > 0:
        torch.set_num_threads(threads)

    seg_out = args.pop("segment_out")
    word_out = args.pop("word_out")
    all_out = args.pop("all")
    ass_out = args.pop("ass")
    csv_out = args.pop("csv")
    json_out = args.pop("json")
    srt_out = args.pop("srt")
    txt_out = args.pop("txt")
    vtt_out = args.pop("vtt")

    model = whisper.load_model(model_name, device=device, download_root=model_dir)

    debug = args.pop("debug")
    if debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("Whisper").setLevel(logging.DEBUG)

    align_language = args["language"] if args["language"] is not None else "en" # default to loading english if not specified
    align_model, align_metadata = load_align_model(align_language, device, model_name=align_model)

    for audio_path in args.pop("audio"):
        print("Performing transcription...")
        result = whisper.transcribe(model, 
            audio_path, 
            temperature=temperature, 
            **args)

        if result["language"] != align_metadata["language"]:
            # load new language
            print(f"New language found ({result['language']})! Previous was ({align_metadata['language']}), loading new alignment model for new language...")
            align_model, align_metadata = load_align_model(result["language"], device)

        print("Performing alignment...")
        result_aligned = align(result["segments"], align_model, align_metadata, audio_path, device,
                                extend_duration=align_extend, start_from_previous=align_from_prev, drop_non_aligned_words=drop_non_aligned)
        audio_basename = os.path.basename(audio_path)

        if output_dir:

            outname = os.path.join(output_dir, os.path.basename(audio_path))
            # save JSON
            if all_out or json_out:
                print("   Saving JSON...")
                if seg_out or word_out:  
                    with open(outname + ".json", "w", encoding="utf-8") as js:
                        json.dump(result_aligned, js, indent=2, ensure_ascii=False)

            # save TXT
            if all_out or txt_out:
                print("   Saving TXT...")
                if seg_out:
                    print("     Segments...")
                    with open(outname + ".txt", "w", encoding="utf-8") as txt:
                        write_txt(result_aligned["segments"], file=txt)
                if word_out:
                    print("     There is no ability to output an aligned word list as TXT yet")


            # save VTT
            if all_out or vtt_out:
                print("   Saving VTT...")
                if seg_out:
                    print("     Segments...")
                    with open(outname + ".vtt", "w", encoding="utf-8") as vtt:
                        write_vtt(result_aligned["segments"], file=vtt)
                if word_out:
                    print("     Words...")
                    with open(outname + ".words.vtt", "w", encoding="utf-8") as vtt:
                        #write_vtt_words(result_aligned["word_segments"], file=vtt)
                        write_vtt(result_aligned["word_segments"], file=vtt)

            # save SRT
            if all_out or srt_out:
                print("   Saving SRT...")
                if seg_out:
                    print("     Segments...")
                    with open(outname + ".srt", "w", encoding="utf-8") as srt:
                        write_srt(result_aligned["segments"], file=srt)
                if word_out:
                    print("     Words...")
                    with open(outname + ".words.srt", "w", encoding="utf-8") as srt:
                        #write_srt_words(result_aligned["word_segments"], file=srt)
                        write_srt(result_aligned["word_segments"], file=srt)

            # save CSV
            if all_out or csv_out:
                print("   Saving CSV...")
                if seg_out:
                    print("     Segments...")
                    with open(outname + ".csv", "w", encoding="utf-8") as csv:
                        write_csv(result_aligned["segments"], file=csv)
                if word_out:
                    print("     Words...")
                    with open(outname + ".word.csv", "w", encoding="utf-8") as csv:
                        write_csv(result_aligned["word_segments"], file=csv)

            # save ASS
            # if all_out or write_ass:
            #     if seg_out:
            #         with open(outname + ".ass", "w", encoding="utf-8") as ass:
            #             write_ass(result_aligned["segments"], file=ass)
            #     if word_out:
            #         print("There is no ability to output an aligned word list in ASS yet")




if __name__ == '__main__':
    cli()
