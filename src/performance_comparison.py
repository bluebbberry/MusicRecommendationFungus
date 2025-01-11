# performance_comparison.py
import numpy as np
import requests
from mastodon_client import MastodonClient
import logging

logging.basicConfig(level=logging.INFO)

class PerformanceComparison:
    def __init__(self, model, api_token, api_url, group_switch_threshold=1.5):
        self.model = model
        self.api_token = api_token
        self.api_url = api_url
        self.group_switch_threshold = group_switch_threshold

    def compare_performance(self):
        other_group_results = self.fetch_other_group_results()
        current_performance = np.linalg.norm(self.model)
        for other_group_result in other_group_results:
            other_performance = np.linalg.norm(other_group_result)
            if other_performance > current_performance:
                logging.info("Switching to a new model. Current model underperforming.")
                self.switch_group()
                break
            else:
                logging.info("Current model is performing well.")

    def fetch_other_group_results(self):
        query = """
        PREFIX data: <http://example.org/data/>
        SELECT ?weights WHERE { ?model data:weights ?weights }
        LIMIT 5
        """
        response = requests.post("http://localhost:3030/ds/query", data={'query': query}, headers={'Accept': 'application/sparql-results+json'})
        results = response.json().get("results", {}).get("bindings", [])
        return [np.array(eval(result['weights']['value'])) for result in results]

    def switch_group(self):
        self.model = np.random.rand(3)
        mastodon_api = MastodonClient()
        mastodon_api.post_status("Switched to a new learning group based on performance comparison.")
