# MusicRecommendationFungus

This project simulates federated learning with interactions across an RDF Knowledge Graph and Mastodon. The idea: train a model while collaborating with other agents and posting updates to Mastodon.

## Requirements

- Python 3.8+
- Pip (Python package manager)
- Running RDF Knowledge Graph server (e.g., Fuseki)
- Mastodon account + API token

## Setup

### 1. Clone the Repo

```bash
git clone https://github.com/bluebbberry/BabyFungus.git
cd babyfungus
```

### 2. Install Dependencies

Set up a virtual environment (optional but recommended) and install the dependencies:

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/MacOS
# or
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### 3. Configure

- **RDF Knowledge Graph**: Make sure your Fuseki server is running and update the URLs in the code (e.g., `FUSEKI_SERVER`, `FUSEKI_QUERY`).
- **Mastodon API**: Create a Mastodon API token and set the Mastodon URL in the code (`your_mastodon_api_token`, `https://mastodon.social`).

### 4. Run

To start everything:

```bash
python main.py
```

The system will:
1. Train the model every minute.
2. Post updates to Mastodon.
3. Respond to Mastodon requests (e.g., for predictions).
4. Share gradients and aggregate other groups' models using the RDF graph.

### 5. Play

Now your system is running, and you can interact with it on Mastodon using `#babyfungus`. You can post prediction requests, and the system will respond.

## Code Overview

### 1. **RDF Knowledge Graph (`rdf_knowledge_graph.py`)**
   - Handles communication with Fuseki (saves weights, shares gradients, aggregates data).

### 2. **Federated Learning (`federated_learning.py`)**
   - Trains a simple model (gradient descent), shares gradients, aggregates, posts to Mastodon.

### 3. **Mastodon API (`mastodon_api.py`)**
   - Posts updates, fetches and processes requests with `#babyfungus` hashtag.

### 4. **Performance Comparison (`performance_comparison.py`)**
   - Compares your model's performance with others by fetching weights from the RDF graph. If another model performs better, it "switches groups".

### 5. **Main Program (`main.py`)**
   - Starts threads for training and Mastodon interaction.

## Troubleshooting

1. **RDF Issues**: Ensure Fuseki server is up and endpoints are correct.
2. **Mastodon Issues**: Check the API token and Mastodon URL.
3. **Missing Packages**: If something's missing, run `pip install -r requirements.txt`.

## Improvements

- **Complex Models**: Swap the simple model with something more interesting (e.g., neural networks).
- **Error Handling**: More error catching for API failures, network issues.
- **Scale Up**: Add more agents for federated learning.

## License

MIT License. See [LICENSE](LICENSE) file for details.
