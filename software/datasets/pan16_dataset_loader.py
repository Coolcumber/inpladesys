import json

from datatypes import Document, Segment, Segmentation
from typing import List
from util import directory, file

from .abstract_dataset_loader import AbstractDatasetLoader


class Pan16DatasetLoader(AbstractDatasetLoader):
    def __init__(self, dataset_dir: str):
        super().__init__(dataset_dir)

    def load_documents(self) -> List[Document]:
        files = directory.get_files(
            self.dataset_dir, lambda x: x.endswith(".txt"))
        return [file.read_all_text(f) for f in files]

    def load_segmentations(self) -> List[Segmentation]:
        files = directory.get_files(
            self.dataset_dir, lambda x: x.endswith(".truth"))
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
