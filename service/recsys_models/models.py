import os
import pickle
from typing import List, Optional

from rectools.dataset.identifiers import IdMap
from rectools.models import ImplicitALSWrapperModel
from rectools.tools.ann import UserToItemAnnRecommender

from service.recsys_models.knn_model import UserKnn, load_knn

# TODO: написать аналогичный класс для модели популярного и убрать хардкод
popular_ans = [10440, 15297, 9728, 13865, 4151, 3734, 2657, 4880, 142, 6809]


class UserKNN:
    def __init__(self, N_recs: int = 10):
        self.user_knn: Optional[UserKnn] = None
        self.N_recs: int = N_recs
        self.load_flag: bool = False

    def load(self) -> None:
        if not self.load_flag:
            KNN_MODEL_PATH = "service/recsys_models/tfidf_model_with_popular.pkl"
            if os.path.exists(KNN_MODEL_PATH):
                self.user_knn = load_knn(KNN_MODEL_PATH)
            self.load_flag = True

    def predict(self, user_id: int) -> List[int]:
        if user_id in self.user_knn.users_mapping:
            # self.user_knn is not None and hasattr(self.user_knn, "users_mapping")
            reco = self.user_knn.recommend([user_id], self.N_recs).item_id.to_list()
            if len(reco) < self.N_recs:
                reco += [item for item in popular_ans if item not in reco][: self.N_recs - len(reco)]
        else:
            reco = popular_ans
        return reco


class ANN_ALS:
    def __init__(self, N_recs: int = 10):
        self.N_recs: int = N_recs
        self.als_wrapper: Optional[ImplicitALSWrapperModel] = None
        self.ann: Optional[UserToItemAnnRecommender] = None
        self.item_id_map: Optional[IdMap] = None
        self.user_id_map: Optional[IdMap] = None
        self.item_vectors: Optional[object] = None
        self.user_vectors: Optional[object] = None
        self.load_flag: bool = False
        self.popular_model_answer: List[int] = popular_ans

    def load(self) -> None:
        if not self.load_flag:
            with open("service/recsys_models/als/als.pkl", "rb") as file:
                self.als_wrapper = pickle.load(file)
            if self.als_wrapper is not None:
                self.user_vectors, self.item_vectors = self.als_wrapper.get_vectors()

                with open("service/recsys_models/als/user_id_map.pkl", "rb") as file:
                    self.user_id_map = pickle.load(file)
                with open("service/recsys_models/als/item_id_map.pkl", "rb") as file:
                    self.item_id_map = pickle.load(file)

                index_init_params = {"method": "hnsw", "space": "negdotprod"}
                self.ann = UserToItemAnnRecommender(
                    user_vectors=self.user_vectors,
                    item_vectors=self.item_vectors,
                    user_id_map=self.user_id_map,
                    item_id_map=self.item_id_map,
                    index_init_params=index_init_params,
                )
                self.ann.index.loadIndex("service/recsys_models/als/als_ann_index.pkl")
                self.load_flag = True

    def predict(self, user_id: int) -> list:
        if user_id in self.user_id_map.external_ids:
            # self.user_id_map is not None and hasattr(self.user_id_map, "external_ids")
            if self.ann is not None:
                return self.ann.get_item_list_for_user(user_id, top_n=self.N_recs).tolist()
        return self.popular_model_answer
