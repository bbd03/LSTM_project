from gensim.models import KeyedVectors
from pathlib import Path
import pandas as pd

class Dataloader:
    def __init__(self, model_path):
        self.model_path = self._to_path(model_path)
        self._validate_model_path(self.model_path)

        self._model = None
        self.kv_path = None

    def _to_path(self, model_path) -> Path:
        if isinstance(model_path, Path):
            return model_path
        if isinstance(model_path, str):
            return Path(model_path)
        raise TypeError(f"expected str or Path, got {type(model_path)}")
    
    def _validate_model_path(self, model_path):
        """
        ensures the path exists
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"model file not found at: {self.model_path}")
        if self.model_path.suffix != '.bin':
            raise ValueError(f"expected a .bin binary file, got: {self.model_path.suffix}")
    
    @property
    def model(self) -> KeyedVectors:
        """
        Loads the embeddings model only once.
        Tries .kv first; falls back to .bin and creates .kv.
        """
        if self._model is None:
            kv_path = self.model_path.with_suffix(".kv")

            if kv_path.exists():
                print(f"loading from cached .kv: {kv_path.name}")
                self._model = KeyedVectors.load(str(kv_path))
            else:
                print(f"loading from .bin: {self.model_path.name}")
                self._model = KeyedVectors.load_word2vec_format(str(self.model_path), binary=True)

                print(f"saving cached .kv to: {kv_path.name}")
                self._model.save(str(kv_path))

        return self._model


class DataFrameLoader:
    def __init__(self, df_path):
        self.df_path = self._to_path(df_path)
        self._validate_path(self.df_path)
        self.df = self._load(self.df_path)


    def _to_path(self, df_path : Path) -> Path:
        if isinstance(df_path, Path):
            return df_path
        if isinstance(df_path, str):
            return Path(df_path)
        raise TypeError(f"expected str or Path, got {type(df_path)}")    


    def _validate_path(self, df_path : Path) -> Path:
        if not self.df_path.exists():
            raise FileNotFoundError(f"text file not found at: {self.df_path}")
        if self.df_path.suffix != '.txt':
            raise ValueError(f"expected a .txt file, got: {self.df_path.suffix}")
        return df_path

    def _load(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, delimiter=':::', names=['name', 'title', 'description'], engine='python')
        df.title = df.title.apply(lambda x: x.strip())
        return df
    