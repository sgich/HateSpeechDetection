import config
import torch

# create dataset as expected by Pytorch
# create a class object called KenyaHateSpeechDataset

class KenyaHateSpeechDataset:

  def __init__(self, reviews, targets):
    self.reviews = reviews
    self.targets = targets
    self.tokenizer = config.TOKENIZER
    self.max_len = config.MAX_LEN

  def __len__(self):
    return len(self.reviews)

  def __getitem__(self, item):
    review = str(self.reviews[item])
    target = self.targets[item]
    inputs = self.tokenizer.encode_plus(
      review,
      None,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      truncation=True


    )

    input_ids = inputs['input_ids']
    mask = inputs['attention_mask']

    padding_length = self.max_len - len(input_ids)
    input_ids = input_ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)


    return {
      'review_text': review,
      'input_ids': torch.tensor(input_ids, dtype=torch.long),
      'attention_mask': torch.tensor(mask, dtype=torch.long),
      'targets': torch.tensor(self.targets[item], dtype=torch.float)
    }

