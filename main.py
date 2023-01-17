import streamlit as st
import pandas as pd
import numpy as np
import deep_learning_modules
import torch
from torch.utils.data import DataLoader
import tokenizers
import warnings
import os
import time
import gdown
from PIL import Image

warnings.filterwarnings("ignore")
im = Image.open("res/favicon.ico")
st.set_page_config(
    page_title="Tweet Sentiment Classifier", page_icon=im
)  # , layout="wide")


@st.experimental_singleton(show_spinner=False)
def load_model(_model_params):
    model_path = "models/model_bertweet_large-epoch_1"
    checkFiles = [model_path]
    for path in checkFiles:
        if os.path.exists(path) == False:
            print("I miss :", path)
            msg = st.warning("🚩 Models need to be downloaded... ")
            try:
                with st.spinner("Initiating..."):
                    time.sleep(3)
                    url_pth = "https://drive.google.com/uc?id=1-9Q9HPeTBNBi8c3DWDAQKWTEbh22R0vQ"
                    output = model_path
                    gdown.download(url_pth, output, quiet=False)

                    msg.success("Download was successful ✅")
            except Exception as e:
                msg.error(e)

    dl_model = deep_learning_modules.TransferLearningClassifier
    freeze_pretrained = False

    model = dl_model(_model_params, freeze_pretrained)
    device = torch.device("cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


@st.experimental_singleton(show_spinner=False)
def load_test_data():
    return pd.read_csv("data/test_cleaned.txt")


def get_random_tweet():
    test_data = load_test_data().sample()["text"].values[0]
    st.session_state["current_text"] = test_data


slate = st.empty()
body = slate.container()
with body:
    # title_image = Image.open("viz/wikispeedia.png")

    st.markdown(
        "<h1 style='text-align: center; '>Tweets Sentiment Classifier 🐦</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="big-font">The task of this model is to predict if a tweet message used to contain a positive 🙂 or negative 🙁 smiley, by considering only the remaining text.<br>Click on the button below to see the output of a random tweet (that the model has never seen during the training phase)</p>',
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([1.3, 1, 1])
    with col2:
        start_button = st.button(
            "Give me a random tweet!",
            on_click=get_random_tweet,
        )


def run():
    device = torch.device("cpu")
    model_params = deep_learning_modules.bertweet_model_params
    model = load_model(model_params)
    if "current_text" not in st.session_state:
        get_random_tweet()
    with body:
        title = st.text_input("Tweet", st.session_state["current_text"])
        test_df = pd.DataFrame([title], columns=["text"])
        test_set = deep_learning_modules.TokenizedDataset(test_df, model_params, False)
        test_loader = DataLoader(test_set, batch_size=1)
        res = model.get_predictions(test_loader, device)
        st.markdown(
            "<h1 style='text-align: center; color: grey;'>Sentiment:{}</h1>".format(
                "🙂" if res[0][1] == 1 else "🙁"
            ),
            unsafe_allow_html=True,
        )


run()
