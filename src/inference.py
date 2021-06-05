import argparse
from pathlib import Path

import torch

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import pipeline

from src import voice_recognition


def format_results(sentences, results):
    classes = {'LABEL_0': 'Non-suicide', 'LABEL_1': 'Suicide'}

    formatted_results = []
    for sent, res in zip(sentences, results):
        formatted_results.append((sent, classes[res['label']]))

    return formatted_results


def predict(sentences, data_path='data/', tokenizer=None, model=None):
    data_path = Path(data_path)

    if tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(data_path / tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    if model:
        model = AutoModelForSequenceClassification.from_pretrained(data_path / model)
    else:
        model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased')

    custom_text_pipeline = pipeline('sentiment-analysis',
                                    model=model,
                                    tokenizer=tokenizer,
                                    device=0 if torch.cuda.is_available() else -1)

    results = custom_text_pipeline(sentences,
                                   truncation=True,
                                   padding="max_length",
                                   max_length=90)

    results = format_results(sentences, results)

    print("The prediction is: ", results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true',
                        help="Predict the sentiment of a fixed, predefined set of sentences.")

    parser.add_argument('--input', default="",
                        help="Predict over a user-given sentences. "
                        "The sentences should be given as a string argument, separated by "
                        "the set of characters '&&' e.g. 'this is sentence one&&Sentence two'.")

    parser.add_argument('--voice', default='',
                        help="Use a voice recognition model to perform inference over an audio transcription. "
                        "If the 'record' string is given as an argument, it will prompt for a recording, "
                        "otherwise provide the path to the audio file as an argument.")

    parser.add_argument('--model', default='bert_labels_corrected',
                        help="")

    parser.add_argument('--tokenizer', default=None,
                        help="")

    parser.add_argument('--data_path', default="data/",
                        help="Base path to the directory where all the pretrained models are stored, default=data/")

    args = parser.parse_args()
    sentences = ""

    if args.demo:
        sentences = [
            "I don't like the way this day is going.",
            "The movie was quite bad to be honest with you.",
            "I feel like I'm killing it",
            "No more content like this please.",
            "This piece of music is really enjoyable."
        ]
    elif args.input:
        sentences = args.input.split("&&")
    elif args.voice:
        sentences = voice_recognition.audio_to_string(args.voice)

    print('The input sentences are:', sentences)

    predict(sentences, args.data_path, tokenizer=args.tokenizer, model=args.model)

