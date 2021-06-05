import os
from pathlib import Path

import sounddevice as sd
import soundfile as sf

from speechbrain.pretrained import TransformerASR


def audio_to_string(audio_path='record', duration=5, sample_rate=48000, voice_model_path='data/voice_model/'):
    voice_model_path = Path(voice_model_path)
    print('Loading voice recognition model...')

    asr_model = TransformerASR.from_hparams(
        source="speechbrain/asr-transformer-transformerlm-librispeech",
        savedir=str(voice_model_path),
    )

    transcription = ''
    cleanup = False

    if audio_path == 'record':
        print("No audio files were provided, entering recording mode...")
        input(f'Ready to record {duration} seconds of audio. Press [ENTER] ')
        selection = 'n'

        audio_path = str(voice_model_path / 'temp_recording.wav')
        recording = None

        while 'n' in selection:
            print('Recording...')
            recording = sd.rec(int(duration * sample_rate),
                               samplerate=sample_rate,
                               channels=2)
            sd.wait()

            print('Playing your recording...')
            sd.play(recording, sample_rate)
            selection = input('Keep this result? [y/n]: ').lower()

        sf.write(audio_path, recording, sample_rate, format='WAV')
        cleanup = True

    print('Performing transcription...')
    transcription = asr_model.transcribe_file(audio_path)
    print('The transcription: ', transcription)

    if cleanup:
        os.remove(audio_path)

    return [transcription]


if __name__ == '__main__':
    print(audio_to_string())

