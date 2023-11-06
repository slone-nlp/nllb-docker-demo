import re
import sys
import typing as tp
import unicodedata

import torch
from sacremoses import MosesPunctNormalizer
from sentence_splitter import SentenceSplitter
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

MODEL_URL = "slone/nllb-rus-tyv-v2-extvoc"
L1 = "rus_Cyrl"
L2 = "myv_Cyrl"
LANGUAGES = {
    "Орус | Русский | Russian": L1,
    "Тыва | Тувинский | Tyvan": L2,
}


def get_non_printing_char_replacer(replace_by: str = " ") -> tp.Callable[[str], str]:
    non_printable_map = {
        ord(c): replace_by
        for c in (chr(i) for i in range(sys.maxunicode + 1))
        # same as \p{C} in perl
        # see https://www.unicode.org/reports/tr44/#General_Category_Values
        if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
    }

    def replace_non_printing_char(line) -> str:
        return line.translate(non_printable_map)

    return replace_non_printing_char


class TextPreprocessor:
    """
    Mimic the text preprocessing made for the NLLB model.
    This code is adapted from the Stopes repo of the NLLB team:
    https://github.com/facebookresearch/stopes/blob/main/stopes/pipelines/monolingual/monolingual_line_processor.py#L214
    """

    def __init__(self, lang="en"):
        self.mpn = MosesPunctNormalizer(lang=lang)
        self.mpn.substitutions = [
            (re.compile(r), sub) for r, sub in self.mpn.substitutions
        ]
        self.replace_nonprint = get_non_printing_char_replacer(" ")

    def __call__(self, text: str) -> str:
        clean = self.mpn.normalize(text)
        clean = self.replace_nonprint(clean)
        # replace 𝓕𝔯𝔞𝔫𝔠𝔢𝔰𝔠𝔞 by Francesca
        clean = unicodedata.normalize("NFKC", clean)
        return clean


def fix_tokenizer(tokenizer, new_lang=L2):
    """Add a new language token to the tokenizer vocabulary
    (this should be done each time after its initialization)
    """
    old_len = len(tokenizer) - int(new_lang in tokenizer.added_tokens_encoder)
    tokenizer.lang_code_to_id[new_lang] = old_len - 1
    tokenizer.id_to_lang_code[old_len - 1] = new_lang
    # always move "mask" to the last position
    tokenizer.fairseq_tokens_to_ids["<mask>"] = (
        len(tokenizer.sp_model)
        + len(tokenizer.lang_code_to_id)
        + tokenizer.fairseq_offset
    )

    tokenizer.fairseq_tokens_to_ids.update(tokenizer.lang_code_to_id)
    tokenizer.fairseq_ids_to_tokens = {
        v: k for k, v in tokenizer.fairseq_tokens_to_ids.items()
    }
    if new_lang not in tokenizer._additional_special_tokens:
        tokenizer._additional_special_tokens.append(new_lang)
    # clear the added token encoder; otherwise a new token may end up there by mistake
    tokenizer.added_tokens_encoder = {}
    tokenizer.added_tokens_decoder = {}


def sentenize_with_fillers(text, splitter, fix_double_space=True, ignore_errors=False):
    """Apply a sentence splitter and return the sentences and all separators before and after them"""
    if fix_double_space:
        text = re.sub(" +", " ", text)
    sentences = splitter.split(text)
    fillers = []
    i = 0
    for sentence in sentences:
        start_idx = text.find(sentence, i)
        if ignore_errors and start_idx == -1:
            # print(f"sent not found after {i}: `{sentence}`")
            start_idx = i + 1
        assert start_idx != -1, f"sent not found after {i}: `{sentence}`"
        fillers.append(text[i:start_idx])
        i = start_idx + len(sentence)
    fillers.append(text[i:])
    return sentences, fillers


class Translator:
    def __init__(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_URL, low_cpu_mem_usage=True)
        if torch.cuda.is_available():
            self.model.cuda()
        self.tokenizer = NllbTokenizer.from_pretrained(MODEL_URL)
        fix_tokenizer(self.tokenizer)

        self.splitter = SentenceSplitter("ru")
        self.preprocessor = TextPreprocessor()

        self.languages = LANGUAGES

    def translate(
        self,
        text,
        src_lang=L1,
        tgt_lang=L2,
        max_length="auto",
        num_beams=4,
        by_sentence=True,
        preprocess=True,
        **kwargs,
    ):
        """Translate a text sentence by sentence, preserving the fillers around the sentences."""
        if by_sentence:
            sents, fillers = sentenize_with_fillers(
                text, splitter=self.splitter, ignore_errors=True
            )
        else:
            sents = [text]
            fillers = ["", ""]
        if preprocess:
            sents = [self.preprocessor(sent) for sent in sents]
        results = []
        for sent, sep in zip(sents, fillers):
            results.append(sep)
            results.append(
                self.translate_single(
                    sent,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    max_length=max_length,
                    num_beams=num_beams,
                    **kwargs,
                )
            )
        results.append(fillers[-1])
        return "".join(results)

    def translate_single(
        self,
        text,
        src_lang=L1,
        tgt_lang=L2,
        max_length="auto",
        num_beams=4,
        n_out=None,
        **kwargs,
    ):
        self.tokenizer.src_lang = src_lang
        encoded = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        if max_length == "auto":
            max_length = int(32 + 2.0 * encoded.input_ids.shape[1])
        generated_tokens = self.model.generate(
            **encoded.to(self.model.device),
            forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang],
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=n_out or 1,
            **kwargs,
        )
        out = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        if isinstance(text, str) and n_out is None:
            return out[0]
        return out


if __name__ == "__main__":
    print("Initializing a translator to pre-download models...")
    translator = Translator()
    print("Initialization successful!")
