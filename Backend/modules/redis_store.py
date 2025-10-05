import redis
import numpy as np
import pickle

class RedisEmbeddings:
    def __init__(self, host='localhost', port=6379, db=0):
        self.r = redis.Redis(host=host, port=port, db=db)

    def set_embedding(self, student_id: str, embedding: np.ndarray):
        """
        Store embedding in Redis (serialized with pickle)
        """
        self.r.set(student_id, pickle.dumps(embedding))
        print(f"Saved embedding for student_id={student_id}")

    def get_embedding(self, student_id: str) -> np.ndarray:
        """
        Retrieve embedding from Redis
        """
        data = self.r.get(student_id)
        if data:
            return pickle.loads(data)
        return None

    def get_all_embeddings(self):
        """
        Return all embeddings as dict {student_id: embedding}
        """
        all_keys = self.r.keys()
        return {k.decode(): pickle.loads(self.r.get(k)) for k in all_keys}

    def delete_embedding(self, student_id: str):
        self.r.delete(student_id)
        print(f"Deleted embedding for student_id={student_id}")
