from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import regex
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# initialize
app = FastAPI(
    title="Dialogue Summarizer",
    description="Dialogue summarization using T5-Small",
    version=1.0
)

# load the model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("model/saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("model/saved_summary_model")

# selecting the right device
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
device = torch.device(device)
model.to(device)

# templating => for rendering frontend code
templates = Jinja2Templates(
    directory="templates"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# input schema for dialogue
class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text: str) -> str:
    text = regex.sub(r"\r\n", " ", text) # escape sequences
    text = regex.sub(r"\s+", " ", text) # reduce more than one spaces to one 
    text = regex.sub(r"<.*?>", " ", text) # remove html tags
    text.strip() # remove starting and ending spaces
    return text

def summarize_dialogue(text : str) -> str:
    text = clean_data(text)
    input_tokens = tokenizer(
        text, 
        max_length=512, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    )
    model.to(device)
    output_tokens = model.generate(
        input_ids=input_tokens["input_ids"],
        attention_mask=input_tokens["attention_mask"],
        max_new_tokens=128,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(
        output_tokens[0],
        skip_special_tokens=True,
    )
    return summary


# sending frontend code to client
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html", 
        context={"request": request}
    )

@app.post("/summarize/")     
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

