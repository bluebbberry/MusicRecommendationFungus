import time
import logging
import os
from dotenv import load_dotenv
from rdf_knowledge_graph import RDFKnowledgeGraph
from mastodon_client import MastodonClient
import datetime
import random

from song_recommend_service import SongRecommendService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class MusicRecommendationFungus:
    def __init__(self):
        logging.info("[INIT] Initializing Music Recommendation instance")
        self.mastodon = MastodonClient()
        self.rdf_kg = RDFKnowledgeGraph(mastodon_client=self.mastodon,
                                        fuseki_server=os.getenv("FUSEKI_SERVER_UPDATE_URL"),
                                        fuseki_query=os.getenv("FUSEKI_SERVER_QUERY_URL"))
        self.rdf_kg.insert_gradient(2)
        self.rdf_kg.retrieve_all_gradients(None)
        self.song_recommendation_service = SongRecommendService(songs_csv='songs.csv', user_ratings_csv='user_ratings.csv')
        self.feedback_threshold = float(os.getenv("FEEDBACK_THRESHOLD", 0.5))
        logging.info(f"[CONFIG] Feedback threshold set to {self.feedback_threshold}")

    def start(self):
        switch_team = True
        found_initial_team = False
        i = 0
        while True:
            logging.info(f"[START] Starting epoche {i} (at {datetime.datetime.now()})")
            try:
                if switch_team or not found_initial_team:
                    logging.info("[CHECK] Searching for a new fungus group")
                    link_to_model = self.rdf_kg.look_for_new_fungus_group()
                    if link_to_model is not None:
                        found_initial_team = True
                else:
                    logging.info("[WAIT] No new groups found.")
                    link_to_model = None

                if link_to_model is not None:
                    logging.info("[TRAINING] New fungus group detected, initiating training")
                    model = self.rdf_kg.fetch_model_from_knowledge_base(link_to_model)
                    updates = self.rdf_kg.fetch_updates_from_knowledge_base(link_to_model)
                    model = self.train_and_deploy_model(model, updates)
                    # aggregate knowledge from other nodes
                    # TODO: Include again
                    # self.rdf_kg.aggregate_updates_from_other_nodes(link_to_model, model)

                feedback = self.answer_user_feedback()
                logging.info(f"[FEEDBACK] Received feedback: {feedback}")

                switch_team = self.decide_whether_to_switch_team(feedback)

                self.evolve_behavior(feedback)

                logging.info("[SLEEP] Sleeping for 5 seconds")
                time.sleep(5)
                i = i + 1
            except Exception as e:
                logging.error(f"[ERROR] An error occurred: {e}", exc_info=True)
                time.sleep(60)

    def train_and_deploy_model(self, model, updates):
        try:
            logging.info("[TRAINING] Starting model training")
            model = self.song_recommendation_service.train(model)
            logging.info("Posting model update to Mastodon.")
            self.mastodon.post_status(f"Model updated: {self.song_recommendation_service.model}")
            logging.info(f"[RESULT] Model trained successfully. Model: {model.tolist()}")

            self.rdf_kg.save_model(model)
            logging.info("[STORE] Model saved to RDF Knowledge Graph")

            self.mastodon.post_status(f"Training complete. Updated model: {model.tolist()}")
            logging.info("[NOTIFY] Status posted to Mastodon")
            return model
        except Exception as e:
            logging.error(f"[ERROR] Failed during training and deployment: {e}", exc_info=True)

    def decide_whether_to_switch_team(self, feedback):
        switch_decision = feedback < self.feedback_threshold
        logging.info(f"[DECISION] Switch team: {switch_decision}")
        return switch_decision

    def answer_user_feedback(self):
        statuses = self.mastodon.fetch_latest_statuses(None)
        feedback = 1
        fresh_statuses = filter(lambda s: s["id"] not in self.mastodon.ids_of_replied_statuses, statuses)
        for status in fresh_statuses:
            if "babyfungus" in status['content']:
                reply = self.song_recommendation_service.get_song_recommendations(status['content'])
                self.mastodon.reply_to_status(status['id'], status['account']['username'], reply)
                feedback /= 2
        return feedback

    def evolve_behavior(self, feedback):
        mutation_chance = 0.1
        if random.random() < mutation_chance:
            logging.info("Randomly mutated")
            old_threshold = self.feedback_threshold
            self.feedback_threshold *= random.uniform(0.9, 1.1)  # Randomly adjust threshold
            logging.info(f"[EVOLVE] Feedback threshold mutated from {old_threshold} to {self.feedback_threshold}")

if __name__ == "__main__":
    logging.info("[STARTUP] Launching MusicRecommendationFungus instance")
    baby_fungus = MusicRecommendationFungus()
    baby_fungus.start()
