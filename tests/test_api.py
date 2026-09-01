import pytest
from fastapi.testclient import TestClient

from app.agents.information_extractor import InformationExtractor
from app.dependencies import get_ticket_analysis_service
from app.main import app
from app.providers.base import LLMProvider
from app.providers.exceptions import LLMProviderError
from app.schemas.extraction import CategoryPrediction, ExtractedField, ExtractionResult
from app.services.ticket_analysis_service import TicketAnalysisService

client = TestClient(app)

_DEFAULT_CANNED = ExtractionResult(
    category=CategoryPrediction(value=None, confidence=0.0),
    extracted_fields={},
)


class StubProvider(LLMProvider):
    def __init__(self, canned_result: ExtractionResult | None = None, raise_error: bool = False):
        self._canned_result = canned_result
        self._raise_error = raise_error

    async def extract_information(self, issue_text: str) -> ExtractionResult:
        if self._raise_error:
            raise LLMProviderError("mock provider failure")
        return self._canned_result


def override_with(canned: ExtractionResult | None = None, raise_error: bool = False):
    def _override():
        provider = StubProvider(canned_result=canned or _DEFAULT_CANNED, raise_error=raise_error)
        extractor = InformationExtractor(provider)
        return TicketAnalysisService(extractor)

    return _override


@pytest.fixture(autouse=True)
def stub_service_by_default():
    """
    Ensures every test in this file — even ones that never explicitly
    override the dependency — hits a stub, not the real GeminiProvider.
    Without this, requests that should fail at body-validation (422)
    instead crash earlier during dependency construction, since FastAPI
    resolves Depends() and the request body together, not body-first.
    """
    app.dependency_overrides[get_ticket_analysis_service] = override_with()
    yield
    app.dependency_overrides.clear()


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


def test_analyze_returns_valid_result():
    canned = ExtractionResult(
        category=CategoryPrediction(value="wifi_internet", confidence=0.9),
        extracted_fields={"symptom_type": ExtractedField(value="drops", confidence=0.9)},
    )
    app.dependency_overrides[get_ticket_analysis_service] = override_with(canned=canned)
    response = client.post("/analyze", json={"message": "My wifi keeps dropping."})
    assert response.status_code == 200
    body = response.json()
    assert body["category"]["value"] == "wifi_internet"
    assert "completeness_score" in body
    assert "missing_or_uncertain_fields" in body


def test_analyze_rejects_empty_message():
    response = client.post("/analyze", json={"message": ""})
    assert response.status_code == 422


def test_analyze_missing_message_field_returns_422():
    response = client.post("/analyze", json={})
    assert response.status_code == 422


def test_analyze_provider_failure_returns_503():
    app.dependency_overrides[get_ticket_analysis_service] = override_with(raise_error=True)
    response = client.post("/analyze", json={"message": "Something broke."})
    assert response.status_code == 503
    assert "mock provider failure" in response.json()["detail"]


def test_analyze_hallucinated_category_sanitized_through_full_stack():
    canned = ExtractionResult(
        category=CategoryPrediction(value="not_a_real_category", confidence=0.7),
        extracted_fields={},
    )
    app.dependency_overrides[get_ticket_analysis_service] = override_with(canned=canned)
    response = client.post("/analyze", json={"message": "Something vague."})
    assert response.status_code == 200
    assert response.json()["category"]["value"] is None


def test_analyze_printer_example_returns_expected_shape():
    canned = ExtractionResult(
        category=CategoryPrediction(value="printer_support", confidence=0.9),
        extracted_fields={
            "symptom": ExtractedField(value="offline", confidence=0.9),
            "connection_type": ExtractedField(value="network", confidence=0.85),
        },
    )
    app.dependency_overrides[get_ticket_analysis_service] = override_with(canned=canned)
    response = client.post(
        "/analyze", json={"message": "Printer showing offline, connected over network."}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["category"]["value"] == "printer_support"
    assert "printer_model" in body["missing_or_uncertain_fields"]