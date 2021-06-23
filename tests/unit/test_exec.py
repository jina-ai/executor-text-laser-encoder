__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from jina.executors import BaseExecutor
import pytest
from jinahub.encoder.laser_encoder import LaserEncoder
from jina import Document, DocumentArray


@pytest.fixture()
def docs_generator():
    return DocumentArray((Document(text='random text') for _ in range(30)))


def test_flair_batch(docs_generator):
    encoder = LaserEncoder()
    docs = docs_generator
    encoder.encode(docs, parameters={'batch_size': 10, 'traversal_paths': ['r']})

    assert len(docs.get_attributes('embedding')) == 30
    assert docs[0].embedding.shape == (1024,)
