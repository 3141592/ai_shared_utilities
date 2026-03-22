from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable
from ai_shared_utilities.builders import build_acl_imdb, build_asv_raw
from ai_shared_utilities.builders import build_jena_climate, build_asv_clean_nt
from ai_shared_utilities.builders import build_dogs_vs_cats, build_oxford_pets
from ai_shared_utilities.builders import build_glove_6B,  build_spa_eng
from ai_shared_utilities.builders import build_fasttext_wiki_news, build_celeba_gan
from ai_shared_utilities.builders import build_the_verdict, build_john

@dataclass
class Asset:
    name: str
    kind: str
    relative_path: str
    description: Optional[str] = None
    builder: Optional[Callable[[], None]] = None
    depends_on: Optional[list[str]] = None    
    framework: Optional[str] = None
    path: Optional[Path] = None  # to be filled in after building/downloading


DATASETS = {
    "dogs_vs_cats": Asset(
        name="dogs_vs_cats",
        kind="datasets",
        relative_path="dogs-vs-cats",
        description="Kaggle Dogs vs Cats image classification dataset",
        builder=build_dogs_vs_cats
    ),

    "oxford_pets": Asset(
        name="oxford_pets",
        kind="datasets",
        relative_path="images",
        description="Oxford-IIIT Pet dataset (images and segmentation annotations)",
        builder=build_oxford_pets,
    ),

    "aclImdb": Asset(
        name="aclImdb",
        kind="datasets",
        relative_path="aclImdb",
        description="Large Movie Review Dataset for binary sentiment classification",
        builder=build_acl_imdb
    ),

    "spa-eng": Asset(
        name="spa-eng",
        kind="datasets",
        relative_path="spa-eng/spa.txt",
        description="Spanish-English translation dataset",
        builder=build_spa_eng
    ),

    "celeba_gan": Asset(
        name="celeba_gan",
        kind="datasets",
        relative_path="celeba_gan",
        description="CelebA dataset subset used for GAN examples",
        builder=build_celeba_gan
    ),

    "asv_raw": Asset(
        name="asv_raw",
        kind="datasets",
        relative_path="interpretability/asv.txt",
        description="American Standard Version Bible",
        builder=build_asv_raw
    ),

    "asv_clean_nt": Asset(
        name="asv_clean_nt",
        kind="datasets",
        relative_path="interpretability/asv_clean_nt.txt",
        description="Cleaned New Testament used for language modeling",
        builder=build_asv_clean_nt,
        depends_on=["asv_raw"]
    ),

    "john": Asset(
        name="john",
        kind="datasets",
        relative_path="interpretability/john.txt",
        description="John's Gospel used for language modeling",
        builder=build_john,
        depends_on=["asv_raw"]
    ),

    "the_verdict": Asset(
        name="the_verdict",
        kind="datasets",
        relative_path="interpretability/the_verdict.txt",
        description="Short text used in LLM-from-scratch examples",
        builder=build_the_verdict
    ),

    "jena_climate": Asset(
        name="jena_climate",
        kind="datasets",
        relative_path="jena_climate_2009_2016.csv",
        description="Temperature forecasting dataset",
        builder=build_jena_climate
    ),
}

EMBEDDINGS = {
    # "glove_6B": Asset(...),
    # "fasttext_wiki_news": Asset(...),
    "glove.6B.50d": Asset(
        name="glove.6B.50d",
        kind="embeddings",
        relative_path="glove.6B/glove.6B.50d.txt",
        description="GloVe 6B 50-dimensional word embeddings",
        builder=build_glove_6B
    ),
    "glove.6B.100d": Asset(
        name="glove.6B.100d",
        kind="embeddings",
        relative_path="glove.6B/glove.6B.100d.txt",
        description="GloVe 6B 100-dimensional word embeddings",
        builder=build_glove_6B
    ),  
    "glove.6B.200d": Asset(
        name="glove.6B.200d",
        kind="embeddings",
        relative_path="glove.6B/glove.6B.200d.txt",
        description="GloVe 6B 200-dimensional word embeddings",
        builder=build_glove_6B  
    ),
    "glove.6B.300d": Asset(
        name="glove.6B.300d",
        kind="embeddings",
        relative_path="glove.6B/glove.6B.300d.txt",
        description="GloVe 6B 300-dimensional word embeddings",
        builder=build_glove_6B
    ),

    "wiki-news-300d-1M": Asset(
        name="wiki-news-300d-1M",
        kind="embeddings",
        relative_path="fasttext/wiki-news-300d-1M.vec",
        description="FastText English word embeddings trained on Wikipedia and news",
        builder=build_fasttext_wiki_news
    ),
}

MODELS = {
    # future model checkpoints
    "binary_1gram": Asset(
        name="binary_1gram",
        kind="models",
        framework="keras",
        relative_path="binary_1gram.keras",
        description="Binary sentiment classifier trained on IMDB 1-gram features",
    ),
    "binary_2gram": Asset(
        name="binary_2gram",
        kind="models",
        framework="keras",
        relative_path="binary_2gram.keras",
        description="Binary sentiment classifier trained on IMDB 2-gram features",
    ),

    "binary_2gram": Asset(
        name="binary_2gram",
        kind="models",
        framework="keras",
        relative_path="binary_2gram.keras",
        description="Binary sentiment classifier trained on IMDB 2-gram features",
    ),

    "tfidf_2gram": Asset(
        name="tfidf_2gram",
        kind="models",
        framework="keras",
        relative_path="tfidf_2gram.keras",
        description="TF-IDF 2-gram sentiment classifier",
    ),

    "glove_embeddings_sequence_model": Asset(
        name="glove_embeddings_sequence_model",
        kind="models",
        framework="keras",
        relative_path="glove_embeddings_sequence_model.keras",
        description="GloVe embeddings sequence model for sentiment classification   ",
    ),

    "one_hot_bidir_gru_with_masking": Asset(
        name="one_hot_bidir_gru_with_masking",
        kind="models",
        framework="keras",
        relative_path="one_hot_bidir_gru_with_masking.keras",
        description="One-hot encoded bidirectional GRU model with masking for sentiment classification ",
    ),

    "one_hot_bidir_gru": Asset(
        name="one_hot_bidir_gru",
        kind="models",
        framework="keras",
        relative_path="one_hot_bidir_gru.keras",
        description="One-hot encoded bidirectional GRU model for sentiment classification ",
    ),

    "one_hot_bidir_lstm": Asset(
        name="one_hot_bidir_lstm",
        kind="models",
        framework="keras",
        relative_path="one_hot_bidir_lstm.keras",
        description="One-hot encoded bidirectional LSTM model for sentiment classification ",
    ),

    "fasttext_embeddings_sequence_model": Asset(
        name="fasttext_embeddings_sequence_model",
        kind="models",
        framework="keras",
        relative_path="fasttext_embeddings_sequence_model.keras",
        description="FastText embeddings sequence model for sentiment classification ",
    ),

    "full_transformer_encoder": Asset(
        name="full_transformer_encoder",
        kind="models",
        framework="keras",
        relative_path="full_transformer_encoder.keras",
        description="Full Transformer encoder for sentiment classification ",
    ),

    "full_transformer_encoder_exploration": Asset(
        name="full_transformer_encoder_exploration",
        kind="models",
        framework="keras",
        relative_path="full_transformer_encoder_exploration.keras",
        description="Full Transformer encoder for sentiment classification ",
    ),

    "transformer_encoder_exploration": Asset(
        name="transformer_encoder_exploration",
        kind="models",
        framework="keras",
        relative_path="transformer_encoder_exploration.keras",
        description="Full Transformer encoder for sentiment classification ",
    ),

    "transformer_encoder": Asset(
        name="transformer_encoder",
        kind="models",
        framework="keras",
        relative_path="transformer_encoder.keras",
        description="Full Transformer encoder for sentiment classification ",
    ),

}

ARCHIVES = {
    # optional raw archives if you want to track them explicitly
}

ASSETS = {}
ASSETS.update(DATASETS)
ASSETS.update(EMBEDDINGS)
ASSETS.update(MODELS)
ASSETS.update(ARCHIVES)
REGISTRY = ASSETS

# ---- registry validation ----

for key, asset in ASSETS.items():
    if key != asset.name:
        raise ValueError(
            f"Registry mismatch: key '{key}' does not match asset.name '{asset.name}'"
        )
