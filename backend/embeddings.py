import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# nemotron-3-embed-1b reached end of life; 3.5 shares its text embedding space and 2048 dimensions,
# so the vector(2048) column and already-indexed chunks stay compatible.
EMBED_MODEL = "nvidia/nemotron-3.5-embed-1b"

_client = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ["NVIDIA_API_KEY"],
        )
    return _client

def get_embedding(text: str) -> list[float]:
    response = _get_client().embeddings.create(
        model=EMBED_MODEL,
        input=[text],
        encoding_format="float",
        extra_body={"input_type": "query", "modality": "text", "truncate": "END"},
    )
    return response.data[0].embedding

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Handles batching because NVIDIA has a max batch size of 256."""
    all_embeddings = []
    batch_size = 256
    
    # Process in chunks of 256
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        response = _get_client().embeddings.create(
            model=EMBED_MODEL,
            input=batch,
            encoding_format="float",
            extra_body={"input_type": "passage", "modality": "text", "truncate": "END"},
        )
        
        # Ensure correct order within the batch
        sorted_data = sorted(response.data, key=lambda x: x.index)
        all_embeddings.extend([d.embedding for d in sorted_data])
        
    return all_embeddings