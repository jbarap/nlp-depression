import argparse
from pathlib import Path

import torch

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import pipeline

import tensorflow as tf
import tensorflow_text as text
from keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow.keras import layers

from src import voice_recognition
from src.glove_fastext_bert.prediction_suicidal import predictions


def format_results(sentences, results):
    classes = {'LABEL_0': 'Non-suicidal', 'LABEL_1': 'Suicidal'}

    formatted_results = []
    for sent, res in zip(sentences, results):
        formatted_results.append((sent, classes[res['label']]))

    return formatted_results


def predict(sentences, data_path='data/', tokenizer=None, model=None):
    data_path = Path(data_path)

    if model == 'hf_bert_corrected':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        model = AutoModelForSequenceClassification.from_pretrained(data_path / 'bert_labels_corrected')

        custom_text_pipeline = pipeline('sentiment-analysis',
                                        model=model,
                                        tokenizer=tokenizer,
                                        device=0 if torch.cuda.is_available() else -1)

        results = custom_text_pipeline(sentences,
                                       truncation=True,
                                       padding="max_length",
                                       max_length=90)

        results = format_results(sentences, results)

    elif model == 'tf_ensemble_corrected':
        glove_dataset_dep = 'data/content/gdrive/MyDrive/Clasificador/nlp/glove_suicidal.h5'
        fasttext_dataset_dep = 'data/content/gdrive/MyDrive/Clasificador/nlp/fasttext_dataset_hugging.h5'

        bert_sucidal = tf.saved_model.load('data/content/gdrive/MyDrive/Clasificador/nlp/bert_suicidal')
        bert_sadness = tf.saved_model.load('data/content/gdrive/MyDrive/Clasificador/nlp/bert_sadness')
        glove = keras.models.load_model(glove_dataset_dep)
        fasttext = keras.models.load_model(fasttext_dataset_dep)

        results = []
        for s in sentences:
            r = predictions(s, glove, fasttext, bert_sucidal, bert_sadness)
            decision = 'Non-suicidal' if r[0] > r[1] else 'suicidal'
            results.append((s, decision))

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

    parser.add_argument('--model', default='hf_bert_corrected',
                        choices=['hf_bert_corrected', 'tf_ensemble_corrected'],
                        help="")

    parser.add_argument('--data_path', default="data/",
                        help="Base path to the directory where all the pretrained models are stored, default=data/")

    args = parser.parse_args()
    sentences = ""

    if args.demo:
        sentences = [
            "I don't know what to do anymore. I'm starting to feel alone, and my family isn't great either",
            "The world is a cold place, often you are hated by strangers and there's nothing to do about it.",
            "Not a good start. I didn't get my favorite breakfast this time, oh well.",
            "Today was as good as it gets, minus the extra homework.",
            "I quite enjoyed the meal, it had a unique taste."
        ]
    elif args.input:
        sentences = args.input.split("&&")
    elif args.voice:
        sentences = voice_recognition.audio_to_string(args.voice)

    print('The input sentences are:', sentences)

    predict(sentences, args.data_path, model=args.model)

