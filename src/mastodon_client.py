# mastodon_api.py
import requests
import numpy as np
import os
import logging
from federated_learning import FederatedLearning

logging.basicConfig(level=logging.INFO)

class MastodonClient:
    def __init__(self):
        self.api_token = os.getenv("MASTODON_API_KEY")
        self.instance_url = os.getenv("MASTODON_INSTANCE_URL")
        self.hashtag = os.getenv("NUTRIAL_TAG")
        self.federated_learning = FederatedLearning(self)
        self.ids_of_replied_statuses = []

    def post_status(self, status_text):
        url = f"{self.instance_url}/api/v1/statuses"
        payload = {'status': status_text}
        headers = {
            'Authorization': f'Bearer {self.api_token}',
            'Content-Type': 'application/json'
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            logging.info(f"Posted to Mastodon: {status_text}")
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error posting status: {e}")
            return None

    def fetch_latest_statuses(self, model):
        base_url = f"{self.instance_url}/api/v1"

        headers = {
            'Authorization': f'Bearer {self.api_token}',
            'Accept': 'application/json'
        }

        params = {
            'type': 'statuses',
            'tag': self.hashtag,
            'limit': 30
        }

        response = requests.get(f"{base_url}/timelines/tag/{self.hashtag}",
                                headers=headers,
                                params=params)

        if response.status_code == 200:
            data = response.json()
            logging.info(f"Found {len(data)} latest statuses")
            statuses = data
            return statuses
        else:
            logging.error(f"Error: {response.status_code}")
            return None

    def reply_to_status(self, status_id, username, message):
        # Construct the reply message mentioning the user
        reply_message = f"@{username} {message}"

        # Prepare the request headers
        headers = {
            'Authorization': f'Bearer {self.api_token}',
            'Content-Type': 'application/json'
        }

        # Prepare the request payload
        payload = {
            'status': reply_message,
            'in_reply_to_id': status_id
        }
        logging.info("Reply to status with id " + str(status_id) + ": " + reply_message)

        # Send the POST request
        response = requests.post(f'{self.instance_url}/api/v1/statuses', json=payload, headers=headers)

        if response.status_code == 200:
            self.ids_of_replied_statuses.append(status_id)
            print("Reply sent successfully!")
        else:
            print(f"Failed to send reply: {response.status_code}")

    def answer_user_feedback(self):
        statuses = self.fetch_latest_statuses(None)
        feedback = 1
        fresh_statuses = filter(lambda s: s["id"] not in self.ids_of_replied_statuses, statuses)
        for status in fresh_statuses:
            if "babyfungus" in status['content']:
                reply = self.federated_learning.generate_reply(status['content'])
                self.reply_to_status(status['id'], status['account']['username'], reply)
                feedback /= 2
        return feedback
