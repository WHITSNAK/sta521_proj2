def iter_by_chunk(iterable, chunksize=10):
    """Helper to get a pre-determined chunk of outputs all together from a iterable"""
    chunks = []
    for item in iterable:
        chunks.append(item)
        if len(chunks) == chunksize:
            yield chunks
            chunks = []
    
    yield chunks
