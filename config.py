import transformers

BERT_PATH = ""
MODEL_PATH = ".../model.bin"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)

#tokens added to vocab for this particular case
TOKENIZER.add_tokens(["😃","😁","😆","😡","🤬", "🖕","💯","💩","🤢","🤮"])
