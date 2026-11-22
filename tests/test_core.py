"""
Core tests for the invoice extractor.

What is tested here
-------------------
1. Loader: corrupt PDF, empty file, missing file, unsupported extension.
2. CLI (subprocess): corrupt and empty files → exit 1, error JSON on stderr,
   nothing on stdout.
3. LLM module (mocked): API timeout, unparseable response, empty response,
   truncated response all raise RuntimeError and are never cached.
4. Validate: schema errors, real calendar dates, totals check, non-invoice,
   date_null_on_invoice review flag.
5. Correction retry logic: validate → trigger → outcome classification.

These tests must NOT make real API calls and must NOT open labelled documents.
"""

import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# conftest.py has already set env vars and added src to sys.path
from loader import LoadError, load_document
from validate import TOLERANCE, ValidationResult, validate_extraction

PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_extract(*args: str) -> subprocess.CompletedProcess:
    """Run extract.py as a subprocess from the project root."""
    return subprocess.run(
        [sys.executable, "extract.py", *args],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )


def _make_corrupt_pdf(tmp_path: Path) -> Path:
    p = tmp_path / "corrupt.pdf"
    p.write_bytes(b"%PDF-1.4 not really\xDE\xAD\xBE\xEF")
    return p


def _make_empty_file(tmp_path: Path) -> Path:
    p = tmp_path / "empty.pdf"
    p.write_bytes(b"")
    return p


def _fresh_bytes(tag: str) -> bytes:
    """Unique bytes per test so cache keys never collide."""
    return f"pytest-unique-{tag}".encode()


# ---------------------------------------------------------------------------
# 1. Loader: bad input files
# ---------------------------------------------------------------------------

class TestLoaderBadInputs:
    def test_corrupt_pdf_raises_load_error(self, tmp_path):
        with pytest.raises(LoadError, match="Cannot open PDF"):
            load_document(str(_make_corrupt_pdf(tmp_path)))

    def test_empty_file_raises_load_error(self, tmp_path):
        with pytest.raises(LoadError, match="File is empty"):
            load_document(str(_make_empty_file(tmp_path)))

    def test_missing_file_raises_load_error(self):
        with pytest.raises(LoadError, match="File not found"):
            load_document("/no/such/path/file.pdf")

    def test_unsupported_extension_raises_load_error(self, tmp_path):
        p = tmp_path / "file.docx"
        p.write_bytes(b"fake content")
        with pytest.raises(LoadError, match="Unsupported file type"):
            load_document(str(p))

    def test_force_text_on_image_raises_load_error(self, tmp_path):
        p = tmp_path / "image.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe0fake jpeg")
        with pytest.raises(LoadError, match="text mode"):
            load_document(str(p), force_mode="text")


# ---------------------------------------------------------------------------
# 2. CLI subprocess: hard failure behavior
# ---------------------------------------------------------------------------

class TestCLIHardFailures:
    def test_corrupt_pdf_exit1_nothing_on_stdout(self, tmp_path):
        result = _run_extract(str(_make_corrupt_pdf(tmp_path)))
        assert result.returncode == 1, "corrupt PDF must exit 1"
        assert result.stdout.strip() == "", "stdout must be empty on hard failure"
        err = json.loads(result.stderr.strip())
        assert err["error"] == "load_error"
        assert "message" in err

    def test_empty_file_exit1_nothing_on_stdout(self, tmp_path):
        result = _run_extract(str(_make_empty_file(tmp_path)))
        assert result.returncode == 1
        assert result.stdout.strip() == ""
        err = json.loads(result.stderr.strip())
        assert err["error"] == "load_error"

    def test_missing_file_exit1_nothing_on_stdout(self):
        result = _run_extract("/no/such/file_does_not_exist.pdf")
        assert result.returncode == 1
        assert result.stdout.strip() == ""
        err = json.loads(result.stderr.strip())
        assert err["error"] == "load_error"

    def test_error_json_has_exactly_two_keys(self, tmp_path):
        """Error JSON must have exactly 'error' and 'message' — no extra fields."""
        result = _run_extract(str(_make_corrupt_pdf(tmp_path)))
        err = json.loads(result.stderr.strip())
        assert set(err.keys()) == {"error", "message"}


# ---------------------------------------------------------------------------
# 3. LLM module: failures are RuntimeError and not cached
# ---------------------------------------------------------------------------

class TestLLMFailures:
    """
    These tests mock the OpenAI client to avoid real API calls.
    They verify that call_llm raises RuntimeError for each failure scenario
    and that nothing is written to the cache.
    """

    def _call(self, tag: str, **override):
        """Helper: import call_llm fresh and call it with a unique cache key."""
        from llm import call_llm

        kwargs = dict(
            messages=[{"role": "user", "content": "test"}],
            file_bytes=_fresh_bytes(tag),
            input_mode="text",
            prompt_version="v1",
            attempt_type=f"test_{tag}",
            correction_msg="",
        )
        kwargs.update(override)
        return call_llm(**kwargs)

    def test_api_timeout_raises_runtime_error_after_retries(self):
        """RateLimitError triggers retry ×3, then RuntimeError is raised."""
        # Use RateLimitError because constructing APITimeoutError needs httpx.Request.
        # Both are retried identically in call_llm.
        import openai

        with patch("llm._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            # Simulate a valid HTTP response object for RateLimitError
            mock_resp = MagicMock()
            mock_resp.status_code = 429
            mock_resp.headers = {}

            mock_client.chat.completions.create.side_effect = openai.RateLimitError(
                message="rate limit hit",
                response=mock_resp,
                body={},
            )

            with patch("llm.time.sleep"):  # skip backoff delays in tests
                with pytest.raises(RuntimeError, match="API call failed after 3 attempts"):
                    self._call("timeout")

    def test_unparseable_response_raises_runtime_error(self):
        """A response that cannot be parsed as JSON raises RuntimeError."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[0].message.content = "DEFINITELY NOT JSON: {{{ broken ]]}"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        with patch("llm._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response

            with pytest.raises(RuntimeError, match="not parseable as JSON"):
                self._call("unparse")

    def test_empty_response_body_raises_runtime_error(self):
        """An empty response body is treated as an API failure."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[0].message.content = ""  # empty

        with patch("llm._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response

            with pytest.raises(RuntimeError, match="empty response"):
                self._call("empty_body")

    def test_truncated_response_raises_runtime_error(self):
        """finish_reason='length' (truncated) is treated as an API failure."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].finish_reason = "length"
        mock_response.choices[0].message.content = '{"is_invoice": true, "vendor":'

        with patch("llm._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response

            with pytest.raises(RuntimeError, match="truncated"):
                self._call("truncated")

    def test_failed_responses_are_not_cached(self, tmp_path):
        """Failed API responses must not be written to the cache."""
        import os
        from llm import CACHE_DIR

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[0].message.content = "not json"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 2

        unique_bytes = _fresh_bytes("no_cache_test")

        with patch("llm._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response

            with pytest.raises(RuntimeError):
                from llm import call_llm
                call_llm(
                    messages=[{"role": "user", "content": "x"}],
                    file_bytes=unique_bytes,
                    input_mode="text",
                    prompt_version="v1",
                    attempt_type="no_cache",
                )

        # Verify no cache file was created for this key
        import hashlib
        h = hashlib.sha256()
        for part in [unique_bytes, b"text", b"v1",
                     os.environ["OPENROUTER_MODEL"].encode(), b"no_cache", b""]:
            h.update(part)
            h.update(b"\x00")
        key = h.hexdigest()
        cache_file = CACHE_DIR / key[:2] / f"{key}.json"
        assert not cache_file.exists(), f"Cache must not be written for failed responses: {cache_file}"


# ---------------------------------------------------------------------------
# 4. Validate: schema, dates, totals
# ---------------------------------------------------------------------------

GOOD_INVOICE: dict = {
    "is_invoice": True,
    "vendor": "Acme Corp",
    "invoice_number": "INV-001",
    "invoice_date": "2026-01-15",
    "currency": "USD",
    "line_items": [
        {"description": "Widget A", "quantity": 2, "amount": 100.0},
        {"description": "Widget B", "quantity": None, "amount": 50.0},
    ],
    "tax_total": 27.0,
    "total": 177.0,
}

GOOD_NON_INVOICE: dict = {
    "is_invoice": False,
    "vendor": None,
    "invoice_number": None,
    "invoice_date": None,
    "currency": None,
    "line_items": None,
    "tax_total": None,
    "total": None,
}


class TestValidateSchema:
    def test_valid_invoice_passes(self):
        r = validate_extraction(GOOD_INVOICE)
        assert r.passed(), f"Expected clean pass, got errors: {r.errors}"
        assert not r.needs_review

    def test_valid_non_invoice_passes(self):
        r = validate_extraction(GOOD_NON_INVOICE)
        assert r.passed()
        assert not r.needs_review

    def test_missing_field_is_schema_error(self):
        bad = dict(GOOD_INVOICE)
        del bad["vendor"]
        r = validate_extraction(bad)
        assert r.schema_errors
        assert r.needs_review

    def test_non_invoice_with_data_is_schema_error(self):
        """is_invoice=false but vendor is not null → schema if/then violation."""
        bad = dict(GOOD_NON_INVOICE)
        bad["vendor"] = "Should Be Null"
        r = validate_extraction(bad)
        assert r.schema_errors
        assert r.needs_review

    def test_bad_currency_pattern_is_schema_error(self):
        bad = dict(GOOD_INVOICE)
        bad["currency"] = "usd"  # must be uppercase
        r = validate_extraction(bad)
        assert r.schema_errors

    def test_bad_date_pattern_is_schema_error(self):
        bad = dict(GOOD_INVOICE)
        bad["invoice_date"] = "15/01/2026"  # wrong format
        r = validate_extraction(bad)
        assert r.schema_errors

    def test_line_item_missing_description_is_schema_error(self):
        bad = dict(GOOD_INVOICE)
        bad["line_items"] = [{"quantity": 1, "amount": 10.0}]  # no description
        r = validate_extraction(bad)
        assert r.schema_errors


class TestValidateDates:
    def test_feb_30_is_invalid_calendar_date(self):
        bad = dict(GOOD_INVOICE)
        bad["invoice_date"] = "2026-02-30"
        r = validate_extraction(bad)
        assert any("date_invalid" in e for e in r.errors)
        assert r.needs_review

    def test_feb_29_non_leap_is_invalid(self):
        bad = dict(GOOD_INVOICE)
        bad["invoice_date"] = "2026-02-29"  # 2026 is not a leap year
        r = validate_extraction(bad)
        assert any("date_invalid" in e for e in r.errors)

    def test_feb_29_on_leap_year_is_valid(self):
        good = dict(GOOD_INVOICE)
        good["invoice_date"] = "2024-02-29"  # 2024 IS a leap year
        good["total"] = 177.0
        r = validate_extraction(good)
        assert not any("date_invalid" in e for e in r.errors)

    def test_null_date_on_invoice_flags_review(self):
        ambiguous = dict(GOOD_INVOICE)
        ambiguous["invoice_date"] = None
        r = validate_extraction(ambiguous)
        assert any("date_null_on_invoice" in e for e in r.errors)
        assert r.needs_review

    def test_null_date_on_non_invoice_no_flag(self):
        """Non-invoices always have null dates — no ambiguous-date flag."""
        r = validate_extraction(GOOD_NON_INVOICE)
        assert not any("date_null_on_invoice" in e for e in r.errors)


class TestValidateTotals:
    def test_balancing_totals_pass(self):
        r = validate_extraction(GOOD_INVOICE)
        assert not r.totals_errors

    def test_mismatch_beyond_tolerance_fails(self):
        bad = dict(GOOD_INVOICE)
        bad["total"] = 200.0  # lines(150) + tax(27) = 177 ≠ 200
        r = validate_extraction(bad)
        assert r.totals_errors
        assert r.needs_review

    def test_mismatch_within_tolerance_passes(self):
        close = dict(GOOD_INVOICE)
        close["total"] = 177.005  # diff < 0.01
        r = validate_extraction(close)
        assert not r.totals_errors

    def test_discount_row_included_in_sum(self):
        """Negative line amounts (discounts) must be included in the sum."""
        inv = {
            "is_invoice": True,
            "vendor": "Studio",
            "invoice_number": "PHS-1042",
            "invoice_date": "2026-03-14",
            "currency": "USD",
            "line_items": [
                {"description": "Service A", "quantity": 1, "amount": 3200.0},
                {"description": "Discount (10%)", "quantity": None, "amount": -450.0},
            ],
            "tax_total": 334.13,
            "total": 3084.13,  # 3200 - 450 + 334.13
        }
        r = validate_extraction(inv)
        assert not r.totals_errors, f"Unexpected totals errors: {r.totals_errors}"

    def test_null_line_items_skips_totals_check(self):
        """When line_items is null, the totals check is skipped."""
        inv = dict(GOOD_NON_INVOICE)
        inv["is_invoice"] = True
        inv["vendor"] = "X"
        inv["invoice_number"] = "1"
        inv["invoice_date"] = "2026-01-01"
        inv["currency"] = "USD"
        inv["tax_total"] = 0
        inv["total"] = 100.0
        # line_items is null → totals check skipped, no totals error
        r = validate_extraction(inv)
        assert not r.totals_errors

    def test_zero_tax_is_valid(self):
        inv = dict(GOOD_INVOICE)
        inv["tax_total"] = 0
        inv["total"] = 150.0  # 100 + 50 + 0
        r = validate_extraction(inv)
        assert not r.totals_errors


class TestValidateRetryFlags:
    def test_schema_error_triggers_retry(self):
        bad = dict(GOOD_INVOICE)
        del bad["vendor"]
        r = validate_extraction(bad)
        assert r.should_retry()

    def test_totals_error_triggers_retry(self):
        bad = dict(GOOD_INVOICE)
        bad["total"] = 999.0
        r = validate_extraction(bad)
        assert r.should_retry()

    def test_date_null_does_not_trigger_retry(self):
        """Ambiguous/null date sets needs_review but must not trigger a retry."""
        ambig = dict(GOOD_INVOICE)
        ambig["invoice_date"] = None
        r = validate_extraction(ambig)
        assert r.needs_review
        assert not r.should_retry()

    def test_clean_result_does_not_trigger_retry(self):
        r = validate_extraction(GOOD_INVOICE)
        assert not r.should_retry()

