import insightface
import cv2
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name='buffalo_l'):
        # Load ArcFace model directly
        self.model = insightface.model_zoo.get_model(model_name)
        self.model.prepare(ctx_id=0)

    def get_embedding_from_array(self, img_array):
        """
        Compute embedding from a cropped/aligned face (112x112 BGR)
        """
        try:
            # Convert to RGB
            if img_array.shape[2] == 3:
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img_array

            # Convert to float32 and normalize
            img_rgb = img_rgb.astype(np.float32)
            img_rgb = (img_rgb - 127.5) / 128.0

            # Convert to CHW and add batch dimension
            img_input = np.transpose(img_rgb, (2, 0, 1))  # HWC -> CHW
            img_input = np.expand_dims(img_input, axis=0)  # 1 x 3 x 112 x 112

            emb = self.model.forward(img_input)  # returns 512-d vector
            emb = emb / np.linalg.norm(emb)      # L2-normalize
            return emb
        except Exception as e:
            print("Embedding error:", e)
            import traceback
            traceback.print_exc()
            return None



    def get_average_embedding_from_arrays(self, img_arrays):
        embeddings = []
        for img in img_arrays:
            emb = self.get_embedding_from_array(img)
            if emb is not None:
                embeddings.append(emb)
        if not embeddings:
            return None
        return np.mean(embeddings, axis=0)
