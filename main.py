import streamlit as st
import pandas as pd
import numpy as np
import deep_learning_modules
import torch
from torch.utils.data import DataLoader
import tokenizers
import warnings

warnings.filterwarnings("ignore")


@st.cache(
    hash_funcs={
        tokenizers.Tokenizer: lambda _: 1,
        tokenizers.AddedToken: lambda _: 1,
    },
    suppress_st_warning=True,
)
def load_model(model_params):
    # 1- Select pretrained model parameters: bertweet or x_distil_bert_l6h256
    deep_learning_modules.bertweet_model_params
    # 2- BiLSTM on top of bert or just mean (BiLSTMTransferLearningClassifier or TransferLearningClassifier)
    dl_model = deep_learning_modules.TransferLearningClassifier
    # 3- freeze bert model or not (freeze_pretrained true or false)
    freeze_pretrained = False
    # 4- trained model path
    model_path = "models/model_bertweet_large-epoch_1"
    model = dl_model(model_params, freeze_pretrained)
    device = torch.device("cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


device = torch.device("cpu")
model_params = deep_learning_modules.bertweet_model_params
model = load_model(model_params)

title = st.text_input("Tweet", "Hey! :)")
test_df = pd.DataFrame([title], columns=["text"])
test_set = deep_learning_modules.TokenizedDataset(test_df, model_params, False)
test_loader = DataLoader(test_set, batch_size=1)
res = model.get_predictions(test_loader, device)
st.markdown(
    "<h1 style='text-align: center; color: black;'>Sentiment:{}</h1>".format(
        "🙂" if res[0][1] == 1 else "🙁"
    ),
    unsafe_allow_html=True,
)
