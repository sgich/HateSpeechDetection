import config
import engine
import dataset
import resample
import torch
import pandas as pd
import numpy as np

from model import HateSpeechClassifier
from collections import defaultdict
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW, get_linear_schedule_with_warmup


# define a func for hist dictionary to store loss and accuracy

# CUDA_LAUNCH_BLOCKING=1
def run():
    # split data to train, test and validation
    RANDOM_SEED = 101
    df = pd.read_csv(config.TRAINING_FILE).dropna()

    # split dataset for training and testing(validation)
    df_t, df_valid = model_selection.train_test_split(df,
                                                          test_size=0.2,
                                                          random_state=RANDOM_SEED)

    # use text augmentation to balance the dataset for training

    df_train = resample.augment_text(df_t, resample.aug10p, num_threads=8, num_times=3)


    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = dataset.KenyaHateSpeechDataset(
        reviews=df_train.tweet.values,
        targets= df_train.label.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=2
    )

    valid_dataset = dataset.KenyaHateSpeechDataset(
        reviews=df_valid.tweet.values,
        targets= df_valid.label.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=4
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = HateSpeechClassifier()
    model = model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias","LayerNorm.weight"]
    optimizer_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.001},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},

    ]
    optimizer = AdamW(optimizer_parameters, lr=2e-5)

    total_steps = int(len(df_train)/config.BATCH_SIZE * config.EPOCHS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps)

    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(config.EPOCHS):

        print(f'Epoch {epoch + 1}/{config.EPOCHS}')
        print('-' * 10)

        engine.train_epoch(model,train_data_loader,optimizer,
                           scheduler, device)

        outputs, targets = engine.eval_model(model, valid_data_loader,
                                             device,len(df_valid))

        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)


        print(f'Validation Accuracy Score = {accuracy}')

        print()

        if accuracy > best_accuracy:
             torch.save(model.state_dict(), config.MODEL_PATH)
             best_accuracy = accuracy

if __name__ == "__main__":
    run()


