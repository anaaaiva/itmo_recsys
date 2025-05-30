import os
from http import HTTPStatus

from dotenv import load_dotenv
from starlette.testclient import TestClient

from service.settings import ServiceConfig

GET_RECO_PATH = "/reco/{model_name}/{user_id}"

if not os.getenv("API_KEY"):
    load_dotenv()

API_KEY = os.getenv("API_KEY")


def test_auth():
    assert API_KEY is not None, "API_KEY не задан"


def test_health(client: TestClient) -> None:
    with client:
        response = client.get("/health", headers={"Authorization": f"Bearer {API_KEY}"})
    assert response.status_code == HTTPStatus.OK


def test_get_reco_success(
    client: TestClient,
    service_config: ServiceConfig,
) -> None:
    user_id = 123
    path = GET_RECO_PATH.format(model_name="range_model", user_id=user_id)
    with client:
        response = client.get(path, headers={"Authorization": f"Bearer {API_KEY}"})
    assert response.status_code == HTTPStatus.OK
    response_json = response.json()
    assert response_json["user_id"] == user_id
    assert len(response_json["items"]) == service_config.k_recs
    assert all(isinstance(item_id, int) for item_id in response_json["items"])


def test_get_reco_for_unknown_user(
    client: TestClient,
) -> None:
    user_id = 10**10
    path = GET_RECO_PATH.format(model_name="range_model", user_id=user_id)
    with client:
        response = client.get(path, headers={"Authorization": f"Bearer {API_KEY}"})
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["errors"][0]["error_key"] == "user_not_found"


def test_model_not_found_error(client: TestClient) -> None:
    incorrect_path = GET_RECO_PATH.format(model_name="BeSt_MoDeL_iN_tHe_WoRlD", user_id=123)

    correct_path = GET_RECO_PATH.format(model_name="range_model", user_id=123)

    with client:
        bad_response = client.get(incorrect_path, headers={"Authorization": f"Bearer {API_KEY}"})
        good_response = client.get(correct_path, headers={"Authorization": f"Bearer {API_KEY}"})

    assert good_response.status_code == HTTPStatus.OK
    assert bad_response.status_code == HTTPStatus.NOT_FOUND


def test_authorization_with_token(client: TestClient) -> None:
    invalid_token = "INVALIDTOKEN"

    with client:
        good_response = client.get("/reco/range_model/123", headers={"Authorization": f"Bearer {API_KEY}"})
        bad_response = client.get("/reco/range_model/123", headers={"Authorization": f"Bearer {invalid_token}"})

    assert good_response.status_code == HTTPStatus.OK
    assert bad_response.status_code == HTTPStatus.UNAUTHORIZED
