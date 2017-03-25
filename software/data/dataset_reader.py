from .dataset import Dataset


class MiniBatchReader:
    def __init__(self, dataset: Dataset, batch_size: int):
        self.current_batch_number = 0
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_batches = dataset.size // batch_size

    def reset(self, shuffle: bool = False):
        if shuffle:
            self.dataset.shuffle()
        self.current_batch_number = 0

    def get_next_batch(self):
        """ Return the next `batch_size` image-label pairs. """
        end = self.current_batch_number + self.batch_size
        if end > self.dataset.size:  # Finished epoch
            return None
        else:
            start = self.current_batch_number
        self.current_batch_number = end
        return self.dataset[start:end]

    def __iter__(self):
        b = self.get_next_batch()
        while b is not None:
            b = self.get_next_batch()
            yield b
