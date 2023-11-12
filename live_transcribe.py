

import argparse
import io
import os
import speech_recognition as sr
import whisper
import torch
import time 

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform
from faster_whisper import WhisperModel

model_size = "small.en"
audio_model = WhisperModel(model_size, device="cpu", compute_type="int8")

'''

from transformers import pipeline

def alternate_word(text):
    unmasker = pipeline('fill-mask', model='roberta-base')
    for i in unmasker(text):
        print(i['token_str'])
'''
def trans():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="large-v2", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large-v2"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)  
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()
    
    # The last time a recording was retreived from the queue.
    phrase_time = None
    # Current raw audio bytes.
    last_sample = bytes()
    data_queue = Queue()
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False
    
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")   
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)
    '''
    # Load / Download model
    model = args.model
    if args.model != "large-v2" and not args.non_english:
        model = model + ".en"
    
    audio_model = whisper.load_model(model)
    '''
  
    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout
    temp_file = NamedTemporaryFile().name
    transcription = ['']
    
    with source:
        recorder.adjust_for_ambient_noise(source)
    
    def record_callback(_, audio:sr.AudioData) -> None:
        data = audio.get_raw_data()
        data_queue.put(data)
        
    

    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)
    print("Model loaded.\n")

    while True:
        tokens_count=0
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data
                
                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())
                # Write wav data to the temporary file as bytes.
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())
                # Read the transcription.
                start_time=time.time()
                
                result,_ = audio_model.transcribe(temp_file,vad_filter=True,vad_parameters=dict(min_silence_duration_ms=500),beam_size=5)
                text =list(result)
                segment=text[0]
                txt=segment.text
                '''
                result=audio_model.transcribe(temp_file,fp16=torch.cuda.is_available())
                txt=result["text"]
                '''
                
                if phrase_complete:
                    transcription.append(txt)
                else:
                    transcription[-1] = txt
        

                # Clear the console to reprint the updated transcription.
                os.system('cls' if os.name=='nt' else 'clear')
                for line in transcription:
                    elapsed_time=time.time()-start_time
                    #tokens_count+=tokens_count/elapsed_time
                    print(elapsed_time)
                    print(line)
                print('', end='', flush=True)
                sleep(0.25)
        except KeyboardInterrupt:
            break
    

if __name__=="__main__":
    trans()
