import streamlit as st
import torch
import pickle
import re
import math
from torch import nn

# Assuming your model and tokenizers are in the same directory as app.py
model_path = "model.pth"
source_tokenizer_path = "source_tokenizer.pkl"
target_tokenizer_path = "target_tokenizer.pkl"

# Load tokenizers
with open(source_tokenizer_path, "rb") as f:
    source_tokenizer = pickle.load(f)
with open(target_tokenizer_path, "rb") as f:
    target_tokenizer = pickle.load(f)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define your model architecture (copy-paste from your notebook)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, num_heads=8,
                 num_layers=4, ff_dim=1024, max_len=100):
        super().__init__()
        self.d_model = d_model
        self.encoder_embed = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, ff_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, ff_dim, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        src_emb = self.pos_encoding(self.encoder_embed(src))
        tgt_emb = self.pos_encoding(self.decoder_embed(tgt))

        src_mask = self._generate_square_subsequent_mask(src_emb.size(1)).to(device)
        tgt_mask = self._generate_square_subsequent_mask(tgt_emb.size(1)).to(device)

        encoder_out = self.encoder(src_emb, mask=src_mask)
        decoder_out = self.decoder(tgt_emb, encoder_out, tgt_mask=tgt_mask)
        return self.fc(decoder_out)

    def _generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

# Load the model
model = Transformer(len(source_tokenizer.vocab), len(target_tokenizer.vocab)).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Define the code generation function
def generate_code(model, source_tokenizer, target_tokenizer, text, device):
    tokens = ['<sos>'] + source_tokenizer.tokenize(text) + ['<eos>']
    input_ids = source_tokenizer.convert_tokens_to_ids(tokens)
    input_tensor = torch.tensor([input_ids], device=device)

    with torch.no_grad():
        output = model(input_tensor, input_tensor[:, :-1])

    output_ids = output.argmax(-1).squeeze().cpu().numpy()
    output_tokens = target_tokenizer.convert_ids_to_tokens(output_ids)

    filtered = [t for t in output_tokens if t not in ['<sos>', '<eos>', '<pad>']]
    return ' '.join(filtered).replace(' <eol> ', '\n')

# Define the Streamlit app
def streamlit_app():
    st.title("Pseudocode to C++ Converter")

    pseudocode = st.text_area("Input Pseudocode:", height=200)
    if st.button("Convert"):
        cpp_code = generate_code(model, source_tokenizer, target_tokenizer, pseudocode, device)
        st.code(cpp_code, language="cpp")  # Specify language for syntax highlighting

# Run the app
if __name__ == "__main__":
    streamlit_app()