import torch
import torch.nn as nn

from models import Backbone, Classifier
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp

def load_image_classifier(config, n_classes):
    backbone = Backbone(config["backbone"], config["pretrained"])
    classifier = Classifier(n_classes)
    device = torch.device("cpu")
    model = nn.Sequential(backbone, classifier).to(device)
    model.load_state_dict(torch.load(config["image_classifier_path"], map_location=torch.device('cpu')))
    model.eval()
    return model

def load_llm_model(config):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm_model = LlamaCpp(
        model_path=config["llm_model_path"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        top_p=config["top_p"],
        callback_manager=callback_manager,
        verbose=True,
    )
    return llm_model
