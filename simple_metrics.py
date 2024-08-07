import time
import json
from collections import defaultdict

class SimpleMetrics:
    def __init__(self):
        self.metrics = defaultdict(float)
        self.start_time = time.time()

    def set(self, name, value):
        self.metrics[name] = value

    def inc(self, name, value=1):
        self.metrics[name] += value

    def dec(self, name, value=1):
        self.metrics[name] -= value

    def get_all(self):
        return dict(self.metrics)

    def save_to_file(self, filename='metrics.json'):
        with open(filename, 'w') as f:
            json.dump(self.get_all(), f)

class Gauge:
    def __init__(self, name, description):
        self.name = name
        self.description = description

    def set(self, value):
        metrics.set(self.name, value)

metrics = SimpleMetrics()

def start_http_server(port):
    print(f"Metrics would be served on port {port} if we were using Prometheus.")