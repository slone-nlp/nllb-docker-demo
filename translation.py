import torch
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
from sentence_splitter import SentenceSplitter
MODEL_URL = 'slone/nllb-rus-tyv-v2-extvoc'
LANGUAGES = {
    "Russian": "rus_Cyrl",
    "Tyvan": "tyv_Cyrl",
}


def fix_tokenizer(tokenizer, new_lang='tyv_Cyrl'):
    """ Add a new language token to the tokenizer vocabulary
    (this should be done each time after its initialization)
    """
    old_len = len(tokenizer) - int(new_lang in tokenizer.added_tokens_encoder)
    tokenizer.lang_code_to_id[new_lang] = old_len-1
    tokenizer.id_to_lang_code[old_len-1] = new_lang
    # always move "mask" to the last position
    tokenizer.fairseq_tokens_to_ids["<mask>"] = len(tokenizer.sp_model) + len(tokenizer.lang_code_to_id) + tokenizer.fairseq_offset

    tokenizer.fairseq_tokens_to_ids.update(tokenizer.lang_code_to_id)
    tokenizer.fairseq_ids_to_tokens = {v: k for k, v in tokenizer.fairseq_tokens_to_ids.items()}
    if new_lang not in tokenizer._additional_special_tokens:
        tokenizer._additional_special_tokens.append(new_lang)
    # clear the added token encoder; otherwise a new token may end up there by mistake
    tokenizer.added_tokens_encoder = {}
    tokenizer.added_tokens_decoder = {}


class Translator:
    def __init__(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_URL)
        if torch.cuda.is_available():
            self.model.cuda()
        self.tokenizer = NllbTokenizer.from_pretrained(MODEL_URL)
        fix_tokenizer(self.tokenizer)

        self.splitter = SentenceSplitter("ru")

        self.languages = LANGUAGES

    def translate(
        self,
        text,
        src_lang='rus_Cyrl',
        tgt_lang='tyv_Cyrl',
        max_length='auto',
        num_beams=4,
        no_repeat_ngram_size=4,
        by_sentence=True,
        separator="\n",
        **kwargs
    ):
        """ Translate a text sentence by sentence """
        results = []
        if by_sentence:
            sents = self.splitter.split(text)
        else:
            sents = [text]
        for sent in sents:
            results.append(self.translate_single(
                sent,
                src_lang=src_lang, tgt_lang=tgt_lang,
                max_length=max_length,
                num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size,
                **kwargs
            ))
        return separator.join(results)

    def translate_single(
        self,
        text,
        src_lang='rus_Cyrl',
        tgt_lang='tyv_Cyrl',
        max_length='auto',
        num_beams=4,
        no_repeat_ngram_size=4,
        n_out=None,
        **kwargs
    ):
        self.tokenizer.src_lang = src_lang
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        if max_length == 'auto':
            max_length = int(32 + 2.0 * encoded.input_ids.shape[1])
        generated_tokens = self.model.generate(
            **encoded.to(self.model.device),
            forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang],
            max_length=max_length,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_return_sequences=n_out or 1,
            **kwargs
        )
        out = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        if isinstance(text, str) and n_out is None:
            return out[0]
        return out


if __name__ == '__main__':
    print("Initializing a translator to pre-download models...")
    translator = Translator()
    print("Initialization successful!")
