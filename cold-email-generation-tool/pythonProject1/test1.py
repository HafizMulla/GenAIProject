# Chromadb tutorial
import chromadb

client = chromadb.Client()
collection = client.create_collection(name="my_collection")

collection.add(
    documents = [
        "This document is about New York",
        "This document is about Mumbai",
    ],
    ids = ["id1","id2"]
)

all_doc = collection.get()
print(all_doc)