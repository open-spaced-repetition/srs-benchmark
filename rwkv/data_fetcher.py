class DataFetcher:
    def __init__(self, task_queue, out_queue):
        self.task_queue = task_queue
        self.out_queue = out_queue
        self.storage = {}

    def get(self, key):
        while key not in self.storage:
            group_i, batch = self.out_queue.get()
            self.storage[group_i] = batch

        out = self.storage[key]
        del self.storage[key]
        return out

    def enqueue(self, task):
        self.task_queue.put(task)
