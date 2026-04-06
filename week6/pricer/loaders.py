from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
from pricer.parser import parse
import os

CHUNK_SIZE = 1000

cpu_count = os.cpu_count()
WORKERS = max(cpu_count - 1, 1)


class ItemLoader:
    def __init__(self, category):
        self.category = category
        self.dataset = None

    def from_datapoint(self, datapoint):
        """
        Try to create an Item from this datapoint
        Return the Item if successful, or None if it shouldn't be included
        """
        return parse(datapoint, self.category)

    def from_chunk(self, chunk):
        """
        Create a list of Items from this chunk of elements from the Dataset
        """
        batch = [self.from_datapoint(datapoint) for datapoint in chunk]
        return [item for item in batch if item is not None]

    def chunk_generator(self):
        """
        Iterate over the Dataset, yielding chunks of datapoints at a time
        """
        size = len(self.dataset)
        for i in range(0, size, CHUNK_SIZE):
            yield self.dataset.select(range(i, min(i + CHUNK_SIZE, size)))

    def load_in_parallel(self, workers):
        """
        ThreadPoolExecutor 사용 - Windows Jupyter 환경에서도 안정적으로 동작
        """
        results = []
        chunks = list(self.chunk_generator())
        chunk_count = len(chunks)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, batch in enumerate(pool.map(self.from_chunk, chunks)):
                results.extend(batch)
                if (i + 1) % 10 == 0 or (i + 1) == chunk_count:
                    print(f"  진행 중: {i+1}/{chunk_count} 청크 완료 ({len(results):,}개 항목)", flush=True)
        return results

    def load(self, workers=WORKERS):
        """
        Load in this dataset; the workers parameter specifies how many processes
        should work on loading and scrubbing the data
        """
        start = datetime.now()
        print(f"Loading dataset {self.category}", flush=True)
        self.dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{self.category}",
            split="full",
            trust_remote_code=True,
        )
        results = self.load_in_parallel(workers)
        finish = datetime.now()
        print(
            f"Completed {self.category} with {len(results):,} datapoints in {(finish - start).total_seconds() / 60:.1f} mins",
            flush=True,
        )
        return results
