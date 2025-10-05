from prometheus_client import start_http_server, Summary, Counter, Histogram
import time
import random

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total number of requests')
REQUEST_LATENCY = Histogram('http_request_latency_seconds', 'Request latency')
PROCESSING_TIME = Summary('request_processing_seconds', 'Time spent processing requests')

@PROCESSING_TIME.time()
def process_request():
    REQUEST_COUNT.inc()
    with REQUEST_LATENCY.time():
        time.sleep(random.uniform(0.1, 0.5))  # simulasi proses request

if __name__ == "__main__":
    # Jalankan Prometheus exporter di port 8001
    start_http_server(8001)
    print("Exporter running on http://localhost:8001/metrics")
    while True:
        process_request()
