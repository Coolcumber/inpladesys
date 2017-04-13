import json

from datatypes import Document, Segment, Segmentation
from typing import List
from util import directory, file

from .abstract_dataset_loader import AbstractDatasetLoader


class Pan16DatasetLoader(AbstractDatasetLoader):
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir

    def load_documents(self) -> List[Document]:
        files = directory.get_files(
            self.dataset_dir, filter=lambda x: x.endswith(".txt"), sort=True)
        return [file.read_all_text(f) for f in files]

    def load_segmentations(self) -> List[Segmentation]:
        files = directory.get_files(
            self.dataset_dir, lambda x: x.endswith(".truth"), sort=True)
        texts = [file.read_all_text(f) for f in files]

        def json_to_segmentation(json_text):
            authseg = json.loads(json_text)["authors"]
            segments = []
            for i, a in enumerate(authseg):
                for s in a:
                    length = s["to"] - s["from"] + 1
                    segments.append(Segment(s["from"], length, author=i))
            return Segmentation(len(authseg), segments)
        return [json_to_segmentation(text) for text in texts]
