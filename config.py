import transformers

BERT_PATH = "C:/Users/sgich/Desktop/EaglesFinal/input/bert_base_uncased/"
MODEL_PATH = "C:/Users/sgich/Desktop/EaglesFinal/src/model.bin"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
TOKENIZER.add_tokens(["😃","😁","😆","😡","🤬", "🖕","💯","💩","🤢","🤮"])
