import config
import transformers
import torch.nn as nn
import streamlit as st


class HateSpeechClassifier(nn.Module):
    @st.cache(allow_output_mutation=True)
    def __init__(self):
        super(HateSpeechClassifier, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH, return_dict=False)
        self.bert.resize_token_embeddings(len(config.TOKENIZER))
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(768, 1)

    @st.cache(allow_output_mutation=True)
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        output = self.drop(pooled_output)
        output = self.out(output)
        return output



