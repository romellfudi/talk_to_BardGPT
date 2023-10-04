"""
Author: Romell Freddy Dominguez
Role: Data Scientist and Software Engineer
GitHub: https://github.com/romellfudi
Email: romllz489@gmail.com
"""
import os
import re
import whisper
import time
import random
from dotenv import load_dotenv
import gradio as gr
import requests
from bardapi import Bard
# from speechbrain.pretrained import EncoderDecoderASR

# Function to establish a connection to Bard using a token
def make_connection_to_bard(token):
    session = requests.Session()
    session.headers = {
        "Host": "bard.google.com",
        "X-Same-Domain": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
        "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
        "Origin": "https://bard.google.com",
        "Referer": "https://bard.google.com/",
    }
    session.cookies.set("__Secure-1PSID", token)
    return Bard(token=token, session=session)

# Function to remove code blocks from a chat
def clean_code_blocks(full_chat):
    return re.sub(r"```[\s\S]*?```", "", full_chat)

# Function to transcribe audio using different models
def transcribe(file_wav, transcriber="speechbrain"):
    global asr_model
    global whisper_base_model
    global whisper_tiny_model
    global whisper_large_model
    transcription = "."
    if transcriber == "speechbrain":
        transcription = asr_model.transcribe_file(file_wav)
    elif transcriber == "base_whisper":
        transcription = whisper_base_model.transcribe(file_wav)["text"]
    elif transcriber == "tiny.en_whisper":
        transcription = whisper_tiny_model.transcribe(file_wav)["text"]
    elif transcriber == "large_whisper":
        transcription = whisper_large_model.transcribe(file_wav)["text"]
    return transcription

# Load environment variables from .env file
load_dotenv()
chat_history = []

# Define global variables for models
global asr_model
global whisper_base_model
global whisper_tiny_model
global whisper_large_model

# Load ASR model (speechbrain) - commented out for now
# asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="pretrained_models/asr-crdnn-rnnlm-librispeech")

# Load whisper models
whisper_base_model = whisper.load_model("base")
# whisper_tiny_model = whisper.load_model("tiny.en")
# whisper_large_model = whisper.load_model("large")

# Establish a connection to the Bard API using the token from environment variables
bard = make_connection_to_bard(os.getenv("VAR_1PSID"))

# Get initial instructions from Bard
bard.get_answer(os.getenv("INSTRUCTION"))

# Check if the "bard.ogg" file exists, if not, create it with Bard's initial message
if not os.path.exists("init.ogg"):
    with open("init.ogg", "wb") as f:
        f.write(bytes(bard.speech("Hello, I am Google Bard AI. Please, let's chat.")['audio']))

# Check if the "empty.ogg" file exists, if not, create it with an empty message
if not os.path.exists("empty.ogg"):
    with open("empty.ogg", "wb") as f:
        f.write(bytes(bard.speech(" ")['audio']))

# Initialize the Gradio interface with a custom layout
with gr.Blocks(title="Bard Chatbot") as app:
    gr.Markdown("<div align='center'><h1>Welcome to the Bard Chatbot!</h1></div>")
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Record your message and chat with Bard.")
            audio_input = gr.Audio(label="Input Audio Channel", source="microphone", type="filepath")
            transcriber_options = [
                # "tiny.en_whisper",
                "base_whisper",
                # "large_whisper",
                # "speechbrain",
            ]
            transcriber = gr.Radio(transcriber_options, value="base_whisper", label="Transcriber")

        with gr.Column():
            gr.Markdown("## Reproduce the Bot Answer")
            audio_output = gr.Audio("init.ogg", label="Audio Output", autoplay=True)
            elapsed_time = gr.Text(label="Elapsed Time")

    with gr.Column():
        chatbot = gr.Chatbot() 

    @gr.on(inputs=[audio_input, transcriber], outputs=[audio_output, elapsed_time, chatbot])
    def chat_with_bard(file_wav, transcriber):
        if file_wav is None:
            # Skip if there's no audio input
            return "empty.ogg", f"{0} sec.", chat_history
        start_time = time.time()
        user_input = transcribe(file_wav, transcriber)
        raw = bard.get_answer(user_input)
        bot_message = random.choice(raw['choices'])['content'][0]
        if raw['images']:
            image = random.choice(raw['images'])
            image_markdown = f"![]({image})\n\n"
        else:
            image_markdown = ""
        full_response = image_markdown + bot_message

        chat_history.append((user_input, full_response))

        with open("bard.ogg", "wb") as f:
            f.write(bytes(bard.speech(clean_code_blocks(bot_message))['audio']))
        return "bard.ogg", f"{round(time.time() - start_time)} sec.", chat_history

# Launch the Gradio interface
app.launch(
    server_name="0.0.0.0",  # Be visible on the local network
    server_port=7860,
)
