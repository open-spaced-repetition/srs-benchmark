from io import BytesIO
import torch
from collections import deque


def copy_downcast(master_model, model, dtype):
    master_params = dict(master_model.named_parameters())
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(master_params[name].to(dtype))


def get_number_of_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_tensor(txn, key, device):
    tensor_bytes = txn.get(key.encode())
    buffer = BytesIO(tensor_bytes)
    return torch.load(buffer, weights_only=True, map_location=device)


def save_tensor(txn, key, tensor):
    tensor = tensor.clone().contiguous()
    buffer = BytesIO()
    torch.save(tensor, buffer)
    txn.put(key.encode(), buffer.getvalue())


class SlidingWindowAverage:
    def __init__(self, len: int):
        self.len = len
        self.queue = deque()
        self.tot = 0
        self.n = 0

    def at_capacity(self):
        return self.len == len(self.queue)

    def add_value(self, avg, weight=1):
        self.tot += avg * weight
        self.n += weight

        self.queue.append((avg, weight))
        if len(self.queue) > self.len:
            prev_avg, prev_n = self.queue.popleft()
            self.tot -= prev_avg * prev_n
            self.n -= prev_n

    def get_value(self):
        assert self.n > 0
        return self.tot / self.n


class KeyValueAverage:
    def __init__(self):
        self.values = {}
        self.weights = {}
        self.tot = 0
        self.n = 0

    def add_value(self, key, avg, weight=1):
        if key not in self.values:
            self.values[key] = 0
            self.weights[key] = 0

        self.tot -= self.values[key] * self.weights[key]
        self.n -= self.weights[key]
        self.values[key] = avg
        self.weights[key] = weight
        self.tot += self.values[key] * self.weights[key]
        self.n += self.weights[key]

    def get_value(self):
        assert self.n > 0
        return self.tot / self.n
