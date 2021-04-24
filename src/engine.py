# defining the helper func to training during each epoch which tracks loss and performs back propagation
# to change weights/biases in order to reduce loss

import torch
import torch.nn as nn


from tqdm import tqdm

# defining type of loss function to track
def loss_fn(outputs, targets):
    # return nn.CrossEntropyLoss()(outputs, targets)
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1,1))

def train_epoch(model, data_loader,optimizer,scheduler,device):

    model.train()

    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        input_ids = d['input_ids'].to(device, dtype=torch.long)
        attention_mask = d['attention_mask'].to(device, dtype=torch.long)
        targets = d['targets'].to(device, dtype=torch.float)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()


def eval_model(model, data_loader, device, n_examples):
    model.eval()

    fin_targets = []
    fin_outputs = []

    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            input_ids = d['input_ids'].to(device, dtype=torch.long)
            attention_mask = d['attention_mask'].to(device, dtype=torch.long)
            targets = d['targets'].to(device, dtype=torch.float)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask

            )

            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    return fin_outputs, fin_targets
