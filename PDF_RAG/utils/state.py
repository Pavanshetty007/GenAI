from config import MAX_EXCHANGES

class State:
    def __init__(self):
        self.processed_documents = {}
        self.all_chunks         = []
        self.chunk_to_doc_map   = {}
        self.vectorizer         = None
        self.chunk_embeddings   = None
        self.bm25_index         = None
        self.kg                 = None
        self.max_chat_history   = MAX_EXCHANGES

state = State()
