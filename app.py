import streamlit as st
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import glob
import os

def vid_to_audio(url=None):
    # importing packages 
    from pytube import YouTube 
    import os 

    # url input from user 
    yt = YouTube(url) 

    # extract only audio 
    video = yt.streams.filter(only_audio=True).first() 

    # check for destination to save file 
    destination = '.'

    # download the file 
    out_file = video.download(output_path=destination) 

    # save the file 
    base, ext = os.path.splitext(out_file) 
    new_file = base + '.mp3'
    os.rename(out_file, new_file) 

    # result of success 
    print(yt.title + " has been successfully downloaded.")

    return "OK"

#vid_to_text(url='https://youtu.be/FE5tva_o7ew?si=ztkKeO7qwcpC36AS')

def audio_to_text():
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    


    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-tiny"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        
        torch_dtype=torch_dtype,
        device=device,
    )

    files = glob.glob('*.mp3')[0]
    current_path = os.getcwd()
    file_path = os.path.join(current_path,files)
    result = pipe(file_path)
    print(result["text"])
    return result["text"]
audio_to_text()

def summarize():
    transcript = audio_to_text()
    from transformers import pipeline

    summarizer = pipeline("summarization", model="philschmid/flan-t5-base-samsum")

    
    #print(summarizer(transcript, do_sample=False))

    return summarizer(transcript, do_sample=False)

yt_link = st.text_input("Enter the YouTube URL: ")

with st.button("Start Summarization"):
    
    with st.status("Downloading the video..."):
        vid_to_audio()
    with st.status("Summarizing..."):
        s = summarize()
    st.write(s)
    