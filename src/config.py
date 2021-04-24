import transformers

MAX_LEN = 140
BATCH_SIZE = 8
EPOCHS = 5
BERT_PATH = ""
MODEL_PATH = "model.bin"
TRAINING_FILE = "data/train_final_data.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
TOKENIZER.add_tokens(["😃","😁","😆","😡","🤬", "🖕","💯","💩","🤢","🤮"])
