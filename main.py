import sys
import logging
import torch
import pandas as pd
from scripts.dataloader import Dataloader, DataFrameLoader
from scripts.augmentation import augment_df, replace_tokens
from scripts.config import TRAIN_DATA, GOOGLE_NEWS_MODEL, PROCESSED_DATA
from scripts.config import LOGS_DIR, LOG_FILE, MODELS_DIR
from scripts.config import Config
from scripts.preprocess import TextPreprocessor
from scripts.vocab import Vocabulary
from scripts.train import train_model
from scripts.dataset import MovieDataset
from scripts.utils import save_model_weights



def main():

    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(sys.stdout),
    ]
    )

    df = DataFrameLoader(TRAIN_DATA).df

    pre = TextPreprocessor(
        lowercase=False,
        remove_punctuation=True,
        replace_numbers=False,
        remove_stopwords=False,
        custom_stopwords=False
    )

    df["tokens"] = df.description.astype(str).apply(pre.preprocess)
    loader = Dataloader(GOOGLE_NEWS_MODEL)

    counts = df.title.value_counts()

    # add new data
    aug_threshold = Config.AUGMENT_THRESHOLD
    swaps = Config.SWAPS_PER_SAMPLE
    copies = Config.COPIES_PER_SAMPLE
    AUG_PICKLE = PROCESSED_DATA / 'augmentated_movies.pkl'


    if AUG_PICKLE.exists():
        df = pd.read_pickle(AUG_PICKLE)
        logging.info(f"Loaded augmented DF from {AUG_PICKLE}")
    else:
        df = augment_df(
            df=df,
            augment_threshold=aug_threshold,
            copies_per_sample=copies,
            w2v_model=loader.model,
        )
        df.to_pickle(AUG_PICKLE)

    genre_id, uniques = pd.factorize(df.title)
    df["genre_id"] = genre_id


    vocab = Vocabulary.build_from_dataframe(
        df_tokens=df.tokens,
        kv=loader.model,
        min_freq=2,
        max_vocab=100_000,
        save_dir=PROCESSED_DATA
    )

    counts = df.title.value_counts()

    PAD_IDX, = vocab.encode(["<pad>"])
    TRAIN_ACTIVATE = True
    SAVE_MODEL = True

    if TRAIN_ACTIVATE:
        model = train_model(
            full_ds=df,
            vocab=vocab,
            batch_size=Config.BATCH_SIZE,
            num_epochs=Config.NUM_EPOCHS,
            test_num=Config.TEST_NUM,
            learning_rate=Config.LEARNING_RATE,
            emb_dim=Config.EMBEDDING_DIM,
            hidden_size=Config.HIDDEN_SIZE,
            num_classes=Config.NUM_CLASSES,
            pad_idx=PAD_IDX,
            max_len=Config.MAX_SEQ_LENGTH,
            use_class_weights=Config.USE_CLASS_WEIGHTS,
            max_class_weight=Config.MAX_CLASS_WEIGHT,
            min_class_weight=Config.MIN_CLASS_WEIGHT,
            use_clip_grad=Config.USE_CLIP_GRAD,
            grad_clip_norm=Config.GRAD_CLIP_NORM,
            use_sampler=Config.USE_SAMPLER,
            log_interval=Config.LOG_INTERVAL,
        )


    if SAVE_MODEL:
        save_model_weights(
            model=model,
            path=MODELS_DIR
        )

    MODEL_FILE = MODELS_DIR / "model_weights.pt"


if __name__ == "__main__":
    main()