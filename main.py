from fastapi import FastAPI
from pydantic import BaseModel
from translation import Translator


class TranslationRequest(BaseModel):
    text: str
    src_lang: str = 'rus_Cyrl'
    tgt_lang: str = 'tyv_Cyrl'
    by_sentence: bool = True


app = FastAPI()
translator = Translator()


@app.post("/translate")
def translate(request: TranslationRequest):
    """
    Perform translation with a fine-tuned NLLB model.
    The language codes are supposed to be in 8-letter format, like "eng_Latn".
    Their list can be returned by /list-languages.
    """
    output = translator.translate(
        request.text,
        src_lang=request.src_lang,
        tgt_lang=request.tgt_lang,
        by_sentence=request.by_sentence,
    )
    return {"translation": output}


@app.get("/list-languages")
def list_languages():
    """Show the mapping of supported languages: from their English names to their 8-letter codes."""
    return translator.languages
