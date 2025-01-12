# rdf_knowledge_graph.py
import requests
from rdflib import Graph, Namespace, Literal
import logging
from SPARQLWrapper import SPARQLWrapper, JSON

logging.basicConfig(level=logging.INFO)

class RDFKnowledgeGraph:
    def __init__(self, fuseki_server, fuseki_query, mastodon_client, base_url="http://localhost:3030", dataset="my-knowledge-base"):
        self.FUSEKI_SERVER = fuseki_server
        self.FUSEKI_QUERY = fuseki_query
        self.DATA_NS = Namespace("http://example.org/data/")
        self.graph = Graph()
        self.graph.bind("data", self.DATA_NS)
        self.mastodon_client = mastodon_client
        self.update_url = f"{base_url}/{dataset}/update"
        self.sparql_url = f"{base_url}/{dataset}/query"
        self.fuseki_url = base_url + "/" + dataset
        self.sparql = SPARQLWrapper(self.fuseki_url)

    def save_to_knowledge_graph(self, model):
        self.graph.set((self.DATA_NS["model"], self.DATA_NS["weights"], Literal(str(model.tolist()))))
        response = requests.post(self.FUSEKI_SERVER, data=self.graph.serialize(format='nt'))
        if response.ok:
            logging.info("Model successfully saved to knowledge graph.")
        else:
            logging.error(f"Error saving model: {response.status_code}")

    def share_gradients(self, gradients):
        self.graph.set((self.DATA_NS["model"], self.DATA_NS["gradients"], Literal(str(gradients.tolist()))))
        response = requests.post(self.FUSEKI_SERVER, data=self.graph.serialize(format='nt'))
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
        response = requests.post(self.FUSEKI_QUERY, data={'query': query}, headers={'Accept': 'application/sparql-results+json'})
        results = response.json().get("results", {}).get("bindings", [])
        aggregated_gradients = []
        for result in results:
            gradients = eval(result['gradients']['value'])
            aggregated_gradients.append(gradients)
        return aggregated_gradients

    def look_for_new_fungus_group(self):
        logging.info("Stage 1: Looking for a new fungus group to join...")
        messages = []
        statuses = self.mastodon_client.fetch_latest_statuses(None)
        for status in statuses:
            messages.append(status["content"])
        if not messages:
            logging.warning("No messages found under the nutrial hashtag. Trying again later...")
            return None

        for message in messages:
            if "kb-link" in message:
                logging.info("Found request with join link. Preparing to join calculation ...")
                link_to_knowledge_base = "http://example.org/data/"
                return link_to_knowledge_base
        logging.info("Announcing request to join the next epoch.")
        self.mastodon_client.post_status(f"Request-to-join: Looking for a training group. {self.mastodon_client.hashtag}")
        return None

    def save_model(self, gradient):
        self.insert_gradient(gradient)

    def fetch_model_from_knowledge_base(self, link_to_model):
        return self.retrieve_all_models(link_to_model)

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
        response = requests.get(self.sparql_url, params=params)

        if response.status_code == 200:
            result = response.json()
            gradients = [binding['gradient']['value'] for binding in result['results']['bindings']]
            for gradient in gradients:
                logging.info("found gradients: " + gradient)
            return gradients
        else:
            print(f"Error retrieving data: {response.status_code} - {response.text}")
            return []

    def insert_model(self, model_name, songs_csv, user_ratings_csv, num_epochs, hidden_dim, lr):
        """
        Inserts the model parameters into the Fuseki knowledge base.
        """
        sparql = SPARQLWrapper(self.fuseki_url)
        sparql_insert_query = f'''
        PREFIX ex: <http://example.org/>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        INSERT DATA {{
            ex:{model_name} a ex:ContentBasedModel ;
                            ex:songsData "{songs_csv}" ;
                            ex:userRatingsData "{user_ratings_csv}" ;
                            ex:numEpochs "{num_epochs}"^^xsd:integer ;
                            ex:hiddenDim "{hidden_dim}"^^xsd:integer ;
                            ex:learningRate "{lr}"^^xsd:float .
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

    def retrieve_all_models(self, link_to_model):
        """
        Retrieves all model parameters stored in the Fuseki server.
        """
        sparql = SPARQLWrapper(self.fuseki_url)
        sparql_select_query = '''
        PREFIX ex: <http://example.org/>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        SELECT ?model ?songsData ?userRatingsData ?numEpochs ?hiddenDim ?learningRate
        WHERE {
            ?model a ex:ContentBasedModel ;
                   ex:songsData ?songsData ;
                   ex:userRatingsData ?userRatingsData ;
                   ex:numEpochs ?numEpochs ;
                   ex:hiddenDim ?hiddenDim ;
                   ex:learningRate ?learningRate .
        }
        '''
        sparql.setQuery(sparql_select_query)
        sparql.setReturnFormat(JSON)
        try:
            results = sparql.query().convert()
            models = [
                {
                    "model": result["model"]["value"],
                    "songsData": result["songsData"]["value"],
                    "userRatingsData": result.get("userRatingsData", {}).get("value", "N/A"),
                    "numEpochs": int(result["numEpochs"]["value"]),
                    "hiddenDim": int(result["hiddenDim"]["value"]),
                    "learningRate": float(result["learningRate"]["value"]),
                }
                for result in results["results"]["bindings"]
            ]
            return models
        except Exception as e:
            print(f"Error retrieving models: {e}")
            return []

    def aggregate_updates_from_other_nodes(self, link_to_model, model):
        updates = self.fetch_updates_from_knowledge_base(link_to_model)
        aggregated_updates = [model]  # Include self gradients with higher weight

        # Process and aggregate updates
        for update in updates:
            try:
                # Convert string representation of list back to list
                gradients = eval(update)
                aggregated_updates.append(gradients)
            except Exception as e:
                logging.error(f"Error parsing gradient update: {e}")

        # Calculate the weighted average giving more weight to self gradients
        if aggregated_updates:
            import numpy as np
            weights = [0.5] + [0.5 / len(updates)] * len(updates)
            averaged_gradients = np.average(aggregated_updates, axis=0, weights=weights).tolist()
            logging.info(f"Weighted averaged gradients computed: {averaged_gradients}")
            return averaged_gradients
        else:
            logging.warning("No updates available for aggregation.")
            return []
