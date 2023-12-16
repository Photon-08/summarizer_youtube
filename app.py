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
    
    #files = glob.glob('*.mp3')[0]
    files = os.listdir()
    # Get a list of all files in the current directory
    files = os.listdir()
    #st.write(files)

    # Create an empty list to collect results
    results = []
    
    # Iterate through the files
    for i in files:
        if ".mp3" in i:
            # Build the full path to the MP3 file
            file_path = os.path.join(os.getcwd(), i)
    
            # Display information (optional)
            st.write("Current Directory:", os.getcwd())
            st.write("File Path:", file_path)
            
            
            result = pipe(file_path)
            print(result)
            return result['text']
    


def summarize():
    transcript = audio_to_text()
    len_trans = len(transcript)

    chunks = int(len_trans/512)
    from transformers import pipeline

    summarizer = pipeline("summarization", model="philschmid/flan-t5-base-samsum")

    
    #print(summarizer(transcript, do_sample=False))
    cutoff = 512
    final_output = ''
    for i in range(chunks):
        if i == 0:
            tran_text = transcript[:512]
            inter_output = summarizer(tran_text, do_sample=False)[0]['summary_text']
            final_output += inter_output
            final_output += ' '

        else:
            tran_text = transcript[cutoff:cutoff*2]
            inter_output = summarizer(tran_text, do_sample=False)[0]['summary_text']
            final_output += inter_output
            final_output += ' '
            cutoff = cutoff*2

    return final_output

yt_link = st.text_input("Enter the YouTube URL: ")

if st.button("Start Summarization"):
    
    with st.status("Downloading the video..."):
        vid_to_audio(url=yt_link)
    with st.status("Summarizing..."):
        s = summarize()
        st.write(s)
    
