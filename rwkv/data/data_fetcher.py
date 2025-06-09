"""
Provides a DataFetcher class for asynchronously fetching data.
"""

class DataFetcher:
    """
    A class to fetch data items based on a key, utilizing a task queue
    for requests and an output queue for receiving data.

    This is typically used in a producer-consumer pattern where data is prepared
    or fetched asynchronously and consumed when ready.
    """
    def __init__(self, task_queue, out_queue):
        """
        Initializes the DataFetcher.

        Args:
            task_queue: A queue (e.g., multiprocessing.Queue) to send task requests.
            out_queue: A queue (e.g., multiprocessing.Queue) to receive results.
        """
        self.task_queue = task_queue
        self.out_queue = out_queue
        self.storage = {}  # Internal cache for fetched data

    def get(self, key):
        """
        Retrieves a data item associated with the given key.

        If the item is not in the internal storage, it waits for data
        to arrive from the out_queue, stores it, and then returns the
        requested item. Items are stored with a 'group_i' identifier
        that matches the key.

        Args:
            key: The key (typically 'group_i') of the data item to retrieve.

        Returns:
            The data batch associated with the key.
        """
        while key not in self.storage:
            group_i, batch = self.out_queue.get()
            self.storage[group_i] = batch

        out = self.storage[key]
        del self.storage[key]
        return out

    def enqueue(self, task):
        """
        Enqueues a new task to be processed for data fetching.

        Args:
            task: The task item to be put on the task_queue.
        """
        self.task_queue.put(task)
