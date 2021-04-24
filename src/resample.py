import pandas as pd
import numpy as np
import nlpaug.augmenter.word as nlpaw
from tqdm import tqdm


def augment_sentence(sentence, aug, num_threads):
    return aug.augment(sentence, num_thread=num_threads)


def augment_text(df, aug, num_threads, num_times):

    # Get rows of data to augment
    to_augment = df[df['label'] == 1]
    to_augmentX = to_augment['tweet']
    to_augmentY = np.ones(len(to_augmentX.index) * num_times, dtype=np.int8)

    # Build up dictionary containing augmented data
    aug_dict = {'tweet': [], 'label': to_augmentY}
    for i in tqdm(range(num_times)):
        augX = [augment_sentence(x, aug, num_threads) for x in to_augmentX]
        aug_dict['tweet'].extend(augX)

    # Build DataFrame containing augmented data
    aug_df = pd.DataFrame.from_dict(aug_dict)

    return df.append(aug_df, ignore_index=True).sample(frac=1, random_state=42)


# Define nlpaug augmentation object
aug10p = nlpaw.ContextualWordEmbsAug(model_path='bert-base-uncased', aug_min=1, aug_p=0.1, action="substitute")

#code adapted from: https://github.com/RayWilliam46
