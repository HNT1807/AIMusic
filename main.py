import streamlit as st
import torch
import numpy as np
from pathlib import Path
import tempfile
import time
import os
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from audiocraft.data.audio_utils import convert_audio

# Initialize global variables
MODEL = None
MAX_DURATION = 30
SAMPLE_RATE = 32000


@st.cache_resource
def load_model(model_name):
    global MODEL
    if MODEL is None or MODEL.name != model_name:
        MODEL = MusicGen.get_pretrained(model_name)
    return MODEL


def generate_music(model, text, melody, duration):
    model.set_generation_params(duration=duration)

    if melody is not None:
        sr, melody = melody
        melody = torch.from_numpy(melody).float().mean(dim=0, keepdim=True)
        melody = melody[None]
        melody = convert_audio(melody, sr, SAMPLE_RATE, MODEL.audio_channels)
        output = model.generate_with_chroma(
            descriptions=[text],
            melody_wavs=melody,
            melody_sample_rate=SAMPLE_RATE,
            progress=True
        )
    else:
        output = model.generate(descriptions=[text], progress=True)

    output = output.detach().cpu().float()[0]
    return output


def save_audio(samples):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        audio_write(
            tmp_file.name, samples, MODEL.sample_rate, strategy="loudness",
            loudness_headroom_db=16, loudness_compressor=True, add_suffix=False
        )
    return tmp_file.name


def main():
    st.markdown("<h1 style='text-align: center;'>AI MUSIC GENERATION</h1>", unsafe_allow_html=True)


    model_options = [


        "facebook/musicgen-stereo-small",
        "facebook/musicgen-stereo-medium",
        "facebook/musicgen-stereo-melody",
        "facebook/musicgen-stereo-large",
        "facebook/musicgen-stereo-melody-large"
    ]
    model_name = st.selectbox("Select Model", model_options)

    text_input = st.text_area("Describe your music",
                              "An 80s driving pop song with heavy drums and synth pads in the background")

    duration = st.slider("Duration (seconds)", min_value=1, max_value=MAX_DURATION, value=10)

    melody_file = st.file_uploader("Upload a melody file (optional)", type=["mp3", "wav", "ogg"])

    if st.button("Generate Music"):
        with st.spinner("Loading model..."):
            model = load_model(model_name)

        melody = None
        if melody_file is not None:
            with st.spinner("Processing melody..."):
                with tempfile.NamedTemporaryFile(delete=False,
                                                 suffix="." + melody_file.name.split(".")[-1]) as tmp_file:
                    tmp_file.write(melody_file.getvalue())
                    tmp_file.flush()
                    melody = MusicGen.load_melody(tmp_file.name, SAMPLE_RATE)
                os.unlink(tmp_file.name)

        with st.spinner("Generating music..."):
            start_time = time.time()
            output = generate_music(model, text_input, melody, duration)
            generation_time = time.time() - start_time
            st.write(f"Generation took {generation_time:.2f} seconds.")

        with st.spinner("Saving audio..."):
            output_file = save_audio(output)

        st.audio(output_file)
        st.download_button("Download Generated Music", output_file, file_name="generated_music.wav")


if __name__ == "__main__":
    main()