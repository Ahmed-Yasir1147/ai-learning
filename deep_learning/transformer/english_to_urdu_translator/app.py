from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

import torch
import regex
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI(
    title="English to Urdu Translation",
    description="Web based application to translate English text to Urdu using mT5-small",
    version=1.0
)



# schema for input text in English
class EnglishInput(BaseModel):
    english_text: str