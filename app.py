import streamlit as st
from faster_whisper import WhisperModel

def transcribe_audio(audio_file):
    model = WhisperModel("large-v2", device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_file, beam_size=5)
    segment_text = ""
    for segment in segments:
        segment_text += segment.text
    return segment_text

def main():
    st.title("Audio Transcription App")
    uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])

    if uploaded_file is not None:
        text = transcribe_audio(uploaded_file)
        st.text_area("Transcribed Text", value=text, height=200)

if __name__ == "__main__":
    main()
