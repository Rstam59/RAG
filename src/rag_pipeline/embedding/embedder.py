from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(
            texts,
            batch_size = self.batch_size,
            show_progress_bar = True,
            normalize_embeddings = True
        ) 

        return [v.tolist for v in vecs]