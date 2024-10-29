import streamlit as st
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
import torch._inductor
# torch._dynamo.config.fallback = True
# torch._dynamo.disable()

import os
os.environ["TORCH_HOME"] = r"C:/Users/srima/ML_assignment/ML_assignment3/models"

if (torch.cuda.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class NextWord(nn.Module):
    
    def __init__(self, block_size, vocab_size, emb_dim, hidden_size, activation_fn='relu'):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, vocab_size)
        if activation_fn == 'relu':
            self.activation = nn.ReLU()
        elif activation_fn == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError("Unsupported activation function. Use 'relu' or 'tanh'.")

    def forward(self, x):
        x = self.emb(x)
        x = x.view(x.shape[0], -1)
        x = self.activation(self.lin1(x))  # Apply activation function
        x = self.lin2(x)
        return x

st.write("""
         # Next $k$ word predictor
         """)
st.sidebar.title("Next $k$ word predictor")
st.sidebar.caption("App created by team Backpropogators- ES335")


# seed_number = st.slider("Choose the Seed Number", 0, 10000)
k = st.slider("Number of words to be generated $k$", 50, 10000)

# option = st.radio("Generate the Seed Text?", ("Yes", "No"))

g = torch.Generator()
# g.manual_seed(seed_number)

def stream_data(str):
        for word in str.split(" "):
            yield word + " "
            time.sleep(0.03)
import urllib.request
import random
import re


url = "https://www.gutenberg.org/files/1661/1661-0.txt"
response = urllib.request.urlopen(url)
sherlock_text = response.read().decode("utf-8")
sherlock_text = sherlock_text[1504:]

def generate_word_prediction_dataset(text, block_size=5, print_limit=20):
    sentences = re.split(r'\.\s+|\r\n\r\n', text)
    cleaned_sentences = [
        re.sub(r'[^a-zA-Z0-9 ]', ' ', sentence).strip()
        for sentence in sentences
    ]

    cleaned_sentences = [s for s in cleaned_sentences if len(s.split()) >= 2]

    words = [word for sentence in cleaned_sentences for word in sentence.split()]

    vocabulary = set(words)
    
    stoi = {word: i + 1 for i, word in enumerate(vocabulary)}
    stoi["."] = 0 
    # stoi[" "] = len(stoi)+1
    itos = {i: word for word, i in stoi.items()}
    # itos[0] = "." 

    X, Y = [], [] 
    count = 0 

    for sentence in cleaned_sentences:
        sentence_words = sentence.split() 
        context = [0] * block_size  
        for word in sentence_words + ['.']: 
            ix = stoi[word]  
            X.append(context) 
            Y.append(ix) 

            if count < print_limit:
                print(' '.join(itos[i] for i in context), '--->', itos[ix])
                count += 1
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    print(f"Dataset generated with {len(X)} samples")
    return X, Y, stoi, itos

X, Y, stoi, itos = generate_word_prediction_dataset(sherlock_text, block_size=5, print_limit=20)
# print(list(vocabulary)[:10])
# print(len(vocabulary))

def generate_word(model, itos, stoi, block_size,seed_text=None, max_len=10):
    input_indices = [stoi.get(word, 0) for word in seed_text.split()]
    context = [0] * max(0, block_size - len(input_indices)) + input_indices[-block_size:]
    generated_text = seed_text.strip() + ' '
    for i in range(max_len):
        x = torch.tensor(context).view(1, -1).to(device)
        y_pred = model(x)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        word = itos[ix]
        generated_text += word + ' '
        context = context[1:] + [ix]
    return generated_text.strip()

emb_dim = st.selectbox("Select the Embedding Dimension", [64,128])
activation_funct = st.selectbox(" Select the activation function", ["tanh","relu"])
context_size=st.selectbox("Select the context size",[5,10,15])
seed_text = st.text_input("Enter the seed text (no digits)")

btn = st.button("Generate Text")
if btn: 
    st.subheader("Seed Text")
    st.write_stream(stream_data(seed_text))
    # k=len(stoi)
    model=NextWord(context_size,len(stoi), emb_dim, 1024,activation_funct)
    # model=torch.compile(model)
    if emb_dim==128:
        if context_size==5:
            if activation_funct=="tanh":
                model.load_state_dict(torch.load("models/model_emb128_ctx5_actTanh.pth", map_location=device))
            else :
                model.load_state_dict(torch.load("models/model_emb128_ctx5_actReLU.pth", map_location=device))
        if context_size==10:
            if activation_funct=="tanh":
                model.load_state_dict(torch.load("models/model_emb128_ctx10_actTanh.pth", map_location=device))
            else :
                model.load_state_dict(torch.load("models/model_emb128_ctx10_actReLU.pth", map_location=device))
        if context_size==15:
            if activation_funct=="tanh":
                model.load_state_dict(torch.load("models/model_emb128_ctx15_actTanh.pth", map_location=device))
            else :
                model.load_state_dict(torch.load("models/model_emb128_ctx15_actReLU.pth", map_location=device))
    elif emb_dim==64:
        if context_size==5:
            if activation_funct=="tanh":
                model.load_state_dict(torch.load("models/model_emb64_ctx5_actTanh.pth", map_location=device))
            else :
                model.load_state_dict(torch.load("models/model_emb64_ctx5_actReLU.pth"), map_location=device)
        if context_size==10:
            if activation_funct=="tanh":
                model.load_state_dict(torch.load("models/model_emb64_ctx10_actTanh.pth", map_location=device))
            else :
                model.load_state_dict(torch.load("models/model_emb64_ctx10_actReLU.pth"), map_location=device)
        if context_size==15:
            if activation_funct=="tanh":
                model.load_state_dict(torch.load("models/model_emb64_ctx15_actTanh.pth", map_location=device))
            else :
                model.load_state_dict(torch.load("models/model_emb64_ctx15_actReLU.pth"), map_location=device)
    my_str = generate_word(model, itos, stoi, context_size,seed_text,k)
    decoded_string = bytes(my_str, "utf-8").decode("unicode_escape")
    st.header("Generated Text")
    st.write_stream(stream_data(decoded_string))
    st.sidebar.subheader("Seed Text")
    st.sidebar.write_stream(stream_data(seed_text))
    st.sidebar.header("Generated Text")
    st.sidebar.write_stream(stream_data(decoded_string))


