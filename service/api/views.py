import os
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from starlette import status

from service.api.exceptions import UserNotFoundError
from service.log import app_logger

from ..recsys_models.knn_model import get_offline_recomendations, load_knn
from ..recsys_models.models import ANN_ALS, ANN_DSSM

load_dotenv()


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


class UnauthorizedMessage(BaseModel):
    """
    Answer for 401 error about missing or unknown Bearer token
    """

    detail: str = "Bearer token missing or unknown"
    description: str = "You don't use Bearer token or choose incorrect one"


class SuccessMessage(BaseModel):
    detail: str = "You've succesfully rich /health"


class ModelNotFoundMessage(BaseModel):
    """
    Answer for 404 error about incorrect model name
    """

    detail: str = "Model is not found"
    description: str = "You choose incorrect model name"


router = APIRouter()

get_bearer_token = HTTPBearer(auto_error=False)

KNN_MODEL_PATH = "service/recsys_models/tfidf_model_with_popular.pkl"
tfidf_model_with_popular = None
if os.path.exists(KNN_MODEL_PATH):
    tfidf_model_with_popular = load_knn(KNN_MODEL_PATH)

KNN_MODEL_PREDICTIONS_PATH = "service/recsys_models/knn_tfidf_model_predictions.parquet"
offline_tfidf_model_with_popular_df = None
if os.path.exists(KNN_MODEL_PREDICTIONS_PATH):
    offline_tfidf_model_with_popular_df = pd.read_parquet(KNN_MODEL_PREDICTIONS_PATH)
    offline_tfidf_model_with_popular_df.set_index("user_id", inplace=True)

ann_als_model = ANN_ALS(10)
ann_als_model.load()

ann_dssm_model = ANN_DSSM(10)
ann_dssm_model.load()


async def get_current_user(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
) -> str:
    if not auth or not auth.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token is missing",
        )

    token = auth.credentials
    if token != os.getenv("API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )

    return token


@router.get(path="/health", tags=["Health"], responses={status.HTTP_200_OK: {"model": SuccessMessage}})
async def health(token: str = Depends(get_current_user)) -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    responses={
        status.HTTP_200_OK: {"model": RecoResponse},
        status.HTTP_401_UNAUTHORIZED: {"model": UnauthorizedMessage},
        status.HTTP_404_NOT_FOUND: {"model": ModelNotFoundMessage},
    },
)
async def get_reco(
    request: Request, model_name: str, user_id: int, token: str = Depends(get_current_user)
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    k_recs = request.app.state.k_recs

    if model_name == "range_model":
        reco = list(range(k_recs))
    elif model_name == "tfidf_model_with_popular":
        if tfidf_model_with_popular is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="tfidf_model_with_popular is not loaded",
            )
        reco = tfidf_model_with_popular.recommend([user_id], k_recs).item_id.tolist()
    elif model_name == "offline_tfidf_model_with_popular":
        if offline_tfidf_model_with_popular_df is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="offline_tfidf_model_with_popular is not loaded",
            )
        reco = get_offline_recomendations(user_id, offline_tfidf_model_with_popular_df)
    elif model_name == "ann_als_f128":
        reco = ann_als_model.predict(user_id)
    elif model_name == "dssm":
        reco = ann_dssm_model.predict(user_id)
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ModelNotFoundMessage().detail,
        )
    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
