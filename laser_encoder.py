__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Optional, List, Any, Iterable
import os

import torch
from jina import Executor, DocumentArray, requests
from laserembeddings import Laser


def _batch_generator(data: List[Any], batch_size: int):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


class LaserEncoder(Executor):
    """
    Encode an array of string in size `B` into an ndarray in size `B x D`

    The ndarray potentially is BatchSize x (Channel x Height x Width)

    :class:`LaserEncoder` is a encoder based on Facebook Research's LASER
    (Language-Agnostic SEntence Representations) to compute multilingual
    sentence embeddings: https://github.com/facebookresearch/LASER
    :param path_to_bpe_codes: path to bpe codes from Laser.
        Defaults to Laser.DEFAULT_BPE_CODES_FILE.
    :param path_to_bpe_vocab: path to bpe vocabs from Laser.
        Defaults to Laser.DEFAULT_BPE_VOCAB_FILE.
    :param path_to_encoder: path to the encoder from Laser.
        Defaults to Laser.DEFAULT_ENCODER_FILE.
    :param default_batch_size: size of each batch
    :param default_traversal_paths: traversal path of the Documents, (e.g. 'r', 'c')
    :param on_gpu: set to True if using GPU
    :param language: language of the text. Defaults to english(en).
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    def __init__(
            self,
            path_to_bpe_codes: Optional[str] = None,
            path_to_bpe_vocab: Optional[str] = None,
            path_to_encoder: Optional[str] = None,
            on_gpu: bool = False,
            default_batch_size: int = 32,
            default_traversal_paths: List[str] = ['r'],
            language: str = 'en',
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._path_to_bpe_codes = path_to_bpe_codes
        self._path_to_bpe_vocab = path_to_bpe_vocab
        self._path_to_encoder = path_to_encoder
        self.on_gpu = on_gpu
        self.default_batch_size = default_batch_size
        self.default_traversal_paths = default_traversal_paths
        self.language = language.lower()

        self.model = Laser(
            bpe_codes=self._path_to_bpe_codes,
            bpe_vocab=self._path_to_bpe_vocab,
            encoder=self._path_to_encoder,
        )
        self.device = torch.device('cuda:0') if self.on_gpu else torch.device('cpu')
        self.model.bpeSentenceEmbedding.encoder.encoder.to(self.device)

    @requests
    def encode(self, docs: Optional[DocumentArray], parameters: dict, **kwargs):
        if docs:
            document_batches_generator = self._get_input_data(docs, parameters)
            self._create_embeddings(document_batches_generator)

    def _create_embeddings(self, document_batches_generator: Iterable):
        for document_batch in document_batches_generator:
            text_batch = [d.text for d in document_batch]

            embeddings = self.model.embed_sentences(text_batch, lang=self.language)
            for document, embedding in zip(document_batch, embeddings):
                document.embedding = embedding

    def _get_input_data(self, docs: DocumentArray, parameters: dict):
        traversal_paths = parameters.get('traversal_paths', self.default_traversal_paths)
        batch_size = parameters.get('batch_size', self.default_batch_size)

        # traverse thought all documents which have to be processed
        flat_docs = docs.traverse_flat(traversal_paths)

        # filter out documents without images
        filtered_docs = [doc for doc in flat_docs if doc.text is not None]

        return _batch_generator(filtered_docs, batch_size)