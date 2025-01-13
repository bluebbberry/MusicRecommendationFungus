# rdf_knowledge_graph.py
import requests
from rdflib import Graph, Namespace, Literal
import logging
from SPARQLWrapper import SPARQLWrapper, JSON
import json
import base64
import torch
import os
# Load environment variables
from dotenv import load_dotenv
load_dotenv()
import csv
import pandas as pd
import random

logging.basicConfig(level=logging.INFO)

class RDFKnowledgeGraph:
    def __init__(self, mastodon_client, fuseki_url=os.getenv("FUSEKI_SERVER_URL"), dataset="my-knowledge-base"):
        self.update_url = f"{fuseki_url}/{dataset}/update"
        self.query_url = f"{fuseki_url}/{dataset}/query"
        self.fuseki_url = fuseki_url + "/" + dataset
        self.DATA_NS = Namespace("http://example.org/data/")
        self.graph = Graph()
        self.graph.bind("data", self.DATA_NS)
        self.mastodon_client = mastodon_client
        self.sparql = SPARQLWrapper(self.fuseki_url)

    def save_to_knowledge_graph(self, model):
        self.graph.set((self.DATA_NS["model"], self.DATA_NS["weights"], Literal(str(model.tolist()))))
        response = requests.post(self.update_url, data=self.graph.serialize(format='nt'))
        if response.ok:
            logging.info("Model successfully saved to knowledge graph.")
        else:
            logging.error(f"Error saving model: {response.status_code}")

    def share_gradients(self, gradients):
        self.graph.set((self.DATA_NS["model"], self.DATA_NS["gradients"], Literal(str(gradients.tolist()))))
        response = requests.post(self.update_url, data=self.graph.serialize(format='nt'))
        if response.ok:
            logging.info("Gradients successfully shared.")
        else:
            logging.error("Failed to share gradients.")

    def aggregate_gradients(self):
        query = """
        PREFIX data: <http://example.org/data/>
        SELECT ?gradients WHERE { ?model data:gradients ?gradients }
        LIMIT 5
        """
        response = requests.post(self.query_url, data={'query': query}, headers={'Accept': 'application/sparql-results+json'})
        results = response.json().get("results", {}).get("bindings", [])
        aggregated_gradients = []
        for result in results:
            gradients = eval(result['gradients']['value'])
            aggregated_gradients.append(gradients)
        return aggregated_gradients

    def look_for_new_fungus_group(self):
        logging.info("Stage 1: Looking for a new fungus group to join...")
        messages = []
        random_mycelial_tag = random.choice(os.getenv("MYCELIAL_TAG").split(";"))
        statuses = self.mastodon_client.fetch_latest_statuses(None, random_mycelial_tag)
        for status in statuses:
            messages.append(status["content"])
        if not messages:
            logging.warning("No messages found under the nutrial hashtag. Trying again later...")
            return None

        for message in messages:
            if "model-link" in message:
                logging.info("Found request with join link. Preparing to join calculation ...")
                link_to_knowledge_base = self.extract_after_model_link(message)
                # set new hashtag
                self.mastodon_client.nutrial_tag = random_mycelial_tag
                return link_to_knowledge_base
        logging.info("Announcing request to join the next epoch.")
        self.mastodon_client.post_status(f"Request-to-join: Looking for a training group. {self.mastodon_client.nutrial_tag}")
        return None

    def save_model(self, model_name, model):
        self.insert_model_state(model_name, model.get_state())

    def fetch_all_model_from_knowledge_base(self, link_to_model):
        return self.retrieve_all_model_states(link_to_model)

    def fetch_updates_from_knowledge_base(self, link_to_model):
        return self.retrieve_all_gradients(link_to_model)

    def insert_data(self, sparql_insert_query):
        headers = {"Content-Type": "application/sparql-update"}
        response = requests.post(self.update_url, data=sparql_insert_query, headers=headers)
        return response.status_code, response.text

    def insert_gradient(self, gradient):
        logging.info("Insert gradient: {gradient}".format(gradient=gradient))
        sparql_insert_query = f'''
        PREFIX ex: <http://example.org/>
        INSERT DATA {{
            ex:{0} ex:gradients "{gradient}" ;
                        ex:hasAccuracy "{10}" .
        }}
        '''
        return self.insert_data(sparql_insert_query)

    def retrieve_all_gradients(self, link_to_model):
        """
        Retrieves all gradient values from the Fuseki server.

        Returns:
            list: A list of gradient values, or an empty list if no gradients are found.
        """
        sparql_select_query = '''
        PREFIX ex: <http://example.org/>
        SELECT ?gradient
        WHERE {
            ?agent ex:gradients ?gradient .
        }
        '''

        # Send the SPARQL SELECT query to retrieve all gradient values
        params = {'query': sparql_select_query, 'format': 'application/json'}
        response = requests.get(self.query_url, params=params)

        if response.status_code == 200:
            result = response.json()
            gradients = [binding['gradient']['value'] for binding in result['results']['bindings']]
            for gradient in gradients:
                logging.info("found gradients: " + gradient)
            return gradients
        else:
            print(f"Error retrieving data: {response.status_code} - {response.text}")
            return []

    def insert_model_state(self, model_name, model_state):
        """
        Inserts the model parameters into the Fuseki knowledge base using base64 encoding.
        """
        # Convert tensors to lists for serialization
        state_dict = {k: v.tolist() for k, v in model_state.items()}
        state_json = json.dumps(state_dict)
        state_encoded = base64.b64encode(state_json.encode('utf-8')).decode('utf-8')
        sparql = SPARQLWrapper(self.fuseki_url)
        sparql_insert_query = f'''
        PREFIX ex: <http://example.org/>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        INSERT DATA {{
            ex:{model_name} a ex:ContentBasedModel ;
                            ex:modelState "{state_encoded}" .
        }}
        '''
        sparql.setQuery(sparql_insert_query)
        sparql.setMethod('POST')
        sparql.setReturnFormat(JSON)
        try:
            sparql.query()
            print(f"Model '{model_name}' inserted successfully.")
        except Exception as e:
            print(f"Error inserting model: {e}")

    def insert_song_data(self, song_id, title, genre, artist, tempo, duration):
        """
        Inserts the individual song data into the Fuseki knowledge base.
        """
        # Prepare the SPARQL query to insert the song data
        sparql = SPARQLWrapper(self.fuseki_url)
        sparql_insert_query = f'''
        PREFIX ex: <http://example.org/>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        INSERT DATA {{
            ex:song_{song_id} a ex:Song ;
                               ex:songId {song_id} ;
                               ex:title "{title}" ;
                               ex:genre "{genre}" ;
                               ex:artist "{artist}" ;
                               ex:tempo {tempo} ;
                               ex:duration {duration} .
        }}
        '''

        sparql.setQuery(sparql_insert_query)
        sparql.setMethod('POST')
        sparql.setReturnFormat(JSON)

        try:
            sparql.query()
            print(f"Song '{title}' inserted successfully.")
        except Exception as e:
            print(f"Error inserting song: {e}")

    def get_all_songs(self):
        """
        Retrieves all songs and their data from the Fuseki knowledge base.
        """
        # Prepare the SPARQL query to retrieve all song data
        sparql = SPARQLWrapper(self.fuseki_url)
        sparql_select_query = '''
        PREFIX ex: <http://example.org/>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        SELECT ?song_id ?title ?genre ?artist ?tempo ?duration WHERE {
            ?song a ex:Song ;
                  ex:songId ?song_id ;
                  ex:title ?title ;
                  ex:genre ?genre ;
                  ex:artist ?artist ;
                  ex:tempo ?tempo ;
                  ex:duration ?duration .
        }
        '''

        sparql.setQuery(sparql_select_query)
        sparql.setReturnFormat(JSON)

        try:
            results = sparql.query().convert()

            # Process the result
            songs = []
            for song_data in results["results"]["bindings"]:
                song_info = {
                    "song_id": song_data["song_id"]["value"],
                    "title": song_data["title"]["value"],
                    "genre": song_data["genre"]["value"],
                    "artist": song_data["artist"]["value"],
                    "tempo": int(song_data["tempo"]["value"]),
                    "duration": int(song_data["duration"]["value"])
                }
                songs.append(song_info)

            # Convert the list of dictionaries into a pandas DataFrame
            if songs:
                songs_df = pd.DataFrame(songs)
                return songs_df
            else:
                print("No songs found in the database.")
                return pd.DataFrame()  # Return an empty DataFrame if no data is found
        except Exception as e:
            print(f"Error retrieving song data: {e}")
            return []

    def retrieve_all_model_states(self, link_to_model):
        """
        Retrieves all model parameters stored in the Fuseki server and decodes them.
        """
        sparql = SPARQLWrapper(self.fuseki_url)
        sparql_select_query = '''
        PREFIX ex: <http://example.org/>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        SELECT ?model ?modelState
        WHERE {
            ?model a ex:ContentBasedModel ;
                   ex:modelState ?modelState .
        }
        '''
        sparql.setQuery(sparql_select_query)
        sparql.setReturnFormat(JSON)
        try:
            results = sparql.query().convert()
            models = []
            for result in results["results"]["bindings"]:
                model = result["model"]["value"]
                state_encoded = result["modelState"]["value"]
                state_json = base64.b64decode(state_encoded).decode('utf-8')
                state_dict = json.loads(state_json)
                # Convert lists back to tensors
                model_state = {k: torch.tensor(v) for k, v in state_dict.items()}
                models.append({"model": model, "modelState": model_state})
            return models
        except Exception as e:
            print(f"Error retrieving models: {e}")
            return []

    def aggregate_model_states(self, current_model_state, all_model_states, current_model_weight=0.5):
        """
        Aggregates model states from multiple nodes using a weighted averaging strategy.
        The current model has a higher weight in the averaging process.
        """
        if not all_model_states:
            print("No models available for aggregation.")
            return current_model_state

        # Extract states and convert tensors to numpy arrays for averaging
        state_keys = current_model_state.keys()
        aggregated_state = {k: current_model_weight * current_model_state[k].numpy() for k in state_keys}

        # Add the other models with a lower weight
        for model in all_model_states:
            for k in state_keys:
                aggregated_state[k] += (1 - current_model_weight) * model["modelState"][k].numpy() / len(all_model_states)

        # Convert back to tensors
        aggregated_state = {k: torch.tensor(v) for k, v in aggregated_state.items()}

        print("Model states aggregated successfully with weighted averaging.")
        return aggregated_state

    def insert_songs_from_csv(self, csv_file):
        """
        Inserts song data from a CSV file into the knowledge base.
        """
        with open(csv_file, mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                song_id = row['song_id']
                title = row['title']
                genre = row['genre']
                artist = row['artist']
                tempo = int(row['tempo'])
                duration = int(row['duration'])

                # Insert each song into the knowledge base
                self.insert_song_data(song_id, title, genre, artist, tempo, duration)

    def extract_after_model_link(self, text):
        # Find the index of "model-link:"
        model_link_index = text.find("model-link:")

        if model_link_index == -1:
            return ""

        # Extract the substring after "model-link:"
        result = text[model_link_index + len("model-link:") + 1:]

        # Find the index of the first whitespace character
        whitespace_index = result.find(" ")

        if whitespace_index != -1:
            return result[:whitespace_index]
        else:
            return result.strip()

    def on_found_group_to_join(self, link_to_model):
        self.mastodon_client.post_status("[FUNGUS] model-link: " + str(link_to_model) + " #" + self.mastodon_client.nutrial_tag)
        if link_to_model is not None:
            found_initial_team = True
            self.fuseki_url = link_to_model
        else:
            # default to fuseki server
            self.fuseki_url = os.getenv("FUSEKI_SERVER_URL")
