from pymongo import MongoClient

def get_collection(collection_name: str):
    client = MongoClient("mongodb://localhost:27017")
    db = client["chatbot_db"]
    collection = db[collection_name]
    collection.create_index("timestamp")
    return collection
