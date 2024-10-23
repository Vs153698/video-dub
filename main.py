import streamlit as st
import whisper
import moviepy.editor as mp
from gtts import gTTS
import pyttsx3
from pydub import AudioSegment
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect

# Language models for translation
marian_models = {
    'french': 'Helsinki-NLP/opus-mt-en-fr',
    'spanish': 'Helsinki-NLP/opus-mt-en-es',
    'german': 'Helsinki-NLP/opus-mt-en-de',
    'italian': 'Helsinki-NLP/opus-mt-en-it',
    'russian': 'Helsinki-NLP/opus-mt-en-ru',
    'chinese': 'Helsinki-NLP/opus-mt-en-zh',
    'japanese': 'Helsinki-NLP/opus-mt-en-ja',
    'portuguese': 'Helsinki-NLP/opus-mt-en-pt',
    'arabic': 'Helsinki-NLP/opus-mt-en-ar',
    'korean': 'Helsinki-NLP/opus-mt-tc-big-en-ko',
    'hindi': 'Helsinki-NLP/opus-mt-en-hi',
}

# Text-to-speech language codes
gtts_supported_languages = {
    'french': 'fr',
    'spanish': 'es',
    'italian': 'it',
    'portuguese': 'pt',
    'russian': 'ru',
    'chinese': 'zh',
    'japanese': 'ja',
    'arabic': 'ar',
    'korean': 'ko',
    'hindi': 'hi',
}

# Load Whisper model
model = whisper.load_model("small")

# Function to extract audio from video
def extract_audio(video_file):
    try:
        with open("temp_video.mp4", "wb") as f:
            f.write(video_file.getbuffer())
        video = mp.VideoFileClip("temp_video.mp4")
        audio_file = "audio.wav"
        video.audio.write_audiofile(audio_file)
        return audio_file
    except Exception as e:
        st.error(f"Error extracting audio: {e}")
        return None

# Transcribe audio using Whisper
def transcribe_audio(audio_file):
    try:
        result = model.transcribe(audio_file, task="transcribe")
        return result['text']
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None

# Translate text using MarianMT
def translate_text(text, target_language):
    try:
        model_name = marian_models.get(target_language)
        if model_name:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
            translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
            return translated_text
        else:
            st.error(f"Model not available for language: {target_language}")
            return None
    except Exception as e:
        st.error(f"Error during translation: {e}")
        return None

# gTTS: Convert text to speech using Google Text-to-Speech
def text_to_speech_gtts(text, lang_code):
    try:
        tts = gTTS(text, lang=lang_code)
        tts_audio = "translated_audio.mp3"
        tts.save(tts_audio)
        sound = AudioSegment.from_mp3(tts_audio)
        sound.export("translated_audio.wav", format="wav")
        return "translated_audio.wav"
    except Exception as e:
        st.error(f"Error in gTTS text-to-speech conversion: {e}")
        return None

# pyttsx3: Offline text-to-speech
def text_to_speech_pyttsx3(text):
    try:
        engine = pyttsx3.init()
        engine.save_to_file(text, "translated_audio.wav")
        engine.runAndWait()
        return "translated_audio.wav"
    except Exception as e:
        st.error(f"Error in pyttsx3 text-to-speech conversion: {e}")
        return None

# Merge new audio with the video
def merge_audio_with_video(video_file, new_audio_file):
    try:
        with open("temp_video_for_merge.mp4", "wb") as f:
            f.write(video_file.getbuffer())
        video = mp.VideoFileClip("temp_video_for_merge.mp4")
        new_audio = mp.AudioFileClip(new_audio_file)
        final_video = video.set_audio(new_audio)
        output_file = "translated_video.mp4"
        final_video.write_videofile(output_file, codec='libx264', audio_codec='aac')
        return output_file
    except Exception as e:
        st.error(f"Error merging audio and video: {e}")
        return None

# Generate subtitle file (.srt)
def generate_subtitle(transcribed_text):
    try:
        subtitle_file = "subtitles.srt"
        with open(subtitle_file, "w") as f:
            lines = transcribed_text.split(".")
            for idx, line in enumerate(lines, start=1):
                f.write(f"{idx}\n00:00:0{idx},000 --> 00:00:0{idx + 1},000\n{line.strip()}\n\n")
        return subtitle_file
    except Exception as e:
        st.error(f"Error generating subtitles: {e}")
        return None

# Automatically detect language of transcribed text
def detect_language(text):
    try:
        return detect(text)
    except Exception as e:
        st.error(f"Error detecting language: {e}")
        return None

# Streamlit UI
def main():
    st.set_page_config(page_title="Video Language Translation App", layout="wide")
    st.sidebar.title("Options")
    st.sidebar.header("Select Preferences")

    video_files = st.sidebar.file_uploader("Upload video(s)", type=["mp4", "mov", "avi"], accept_multiple_files=True)
    target_language = st.sidebar.selectbox("Choose the target language", list(gtts_supported_languages.keys()))
    voice_method = st.sidebar.selectbox("Choose voice synthesis method", ["gTTS", "pyttsx3"])

    st.title("🎥 Video Language Translation App")
    st.write("### Upload videos, select a target language, and choose a voice synthesis method.")

    if st.sidebar.button("Process Videos"):
        if video_files:
            for video_file in video_files:
                st.write(f"Processing: {video_file.name}")
                audio_file = extract_audio(video_file)
                if audio_file:
                    transcription = transcribe_audio(audio_file)
                    if transcription:
                        st.write(f"### Transcribed Text for {video_file.name}:")
                        st.text(transcription)

                        # Detect input language
                        detected_lang = detect_language(transcription)
                        st.write(f"Detected Language: {detected_lang}")

                        translated_text = translate_text(transcription, target_language)
                        if translated_text:
                            st.write("### Translated Text:")
                            st.text(translated_text)

                            # Choose voice synthesis method
                            if voice_method == "gTTS":
                                new_audio_file = text_to_speech_gtts(translated_text, gtts_supported_languages[target_language])
                            else:
                                new_audio_file = text_to_speech_pyttsx3(translated_text)

                            if new_audio_file:
                                st.write("Merging new audio with video...")
                                output_video = merge_audio_with_video(video_file, new_audio_file)
                                if output_video:
                                    st.success(f"Processing complete for {video_file.name}!")
                                    st.video(output_video)

                                    st.write("Generating subtitles...")
                                    subtitle_file = generate_subtitle(transcription)
                                    st.write(f"Subtitles saved as {subtitle_file}")
                                else:
                                    st.error("Error merging audio with video.")
                        else:
                            st.error("Translation failed.")
                    else:
                        st.error("Transcription failed.")
        else:
            st.error("Please upload at least one video file.")

if __name__ == "__main__":
    main()
