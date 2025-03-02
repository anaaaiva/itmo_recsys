import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from starlette import status

from service.api.exceptions import UserNotFoundError
from service.log import app_logger

load_dotenv()


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


class UnauthorizedMessage(BaseModel):
    """
    Ответ при ошибке 401 при отсутствующем или неверном Bearer токене
    """

    detail: str = 'Bearer token missing or unknown'
    description: str = 'Вы не указали Bearer token или указали неверный'


class SuccessMessage(BaseModel):
    detail: str = 'Вы успешно достучались до /health'


class ModelNotFoundMessage(BaseModel):
    """
    Ответ при ошибке 404 при неправильном имени модели
    """

    detail: str = 'Model is not found'
    description: str = 'Вы ввели неправильное имя модели'


router = APIRouter()

get_bearer_token = HTTPBearer(auto_error=False)


async def get_current_user(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
) -> str:
    if not auth or not auth.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Bearer token is missing',
        )

    token = auth.credentials
    if token not in [os.getenv('PERSONAL_TOKEN')]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid token',
        )

    return token


@router.get(path='/health', tags=['Health'], responses={status.HTTP_200_OK: {'model': SuccessMessage}})
async def health(token: str = Depends(get_current_user)) -> str:
    return 'I am alive'


@router.get(
    path='/reco/{model_name}/{user_id}',
    tags=['Recommendations'],
    response_model=RecoResponse,
    responses={
        status.HTTP_200_OK: {'model': RecoResponse},
        status.HTTP_401_UNAUTHORIZED: {'model': UnauthorizedMessage},
        status.HTTP_404_NOT_FOUND: {'model': ModelNotFoundMessage},
    },
)
async def get_reco(
    request: Request, model_name: str, user_id: int, token: str = Depends(get_current_user)
) -> RecoResponse:
    app_logger.info(f'Request for model: {model_name}, user_id: {user_id}')

    if user_id > 10**9:
        raise UserNotFoundError(error_message=f'User {user_id} not found')

    k_recs = request.app.state.k_recs

    if model_name == 'range_model':
        reco = list(range(k_recs))
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ModelNotFoundMessage().detail,
        )
    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
