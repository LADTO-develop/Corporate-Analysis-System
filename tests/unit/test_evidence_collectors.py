"""Unit tests for optional external evidence collectors."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Mapping
from pathlib import Path

from cas.evidence.collectors import collect_external_evidence, external_evidence_enabled


class _FakeResponse:
    def __init__(self, payload: object, *, content: bytes = b"") -> None:
        self._payload = payload
        self._content = content

    @property
    def content(self) -> bytes:
        return self._content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


class _FakeSession:
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        if "naver.com" in url:
            return _FakeResponse(
                {
                    "items": [
                        {
                            "title": "<b>테스트기업</b> 횡령 의혹 공시",
                            "description": "횡령 관련 위험 신호가 보도되었습니다.",
                            "originallink": "https://example.com/news",
                            "pubDate": "Thu, 14 May 2026 09:00:00 +0900",
                        }
                    ]
                }
            )
        return _FakeResponse({"list": []})

    def post(
        self,
        url: str,
        *,
        json: Mapping[str, object] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        return _FakeResponse({"results": []})


class _WeakKeywordSession:
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        if "naver.com" in url:
            return _FakeResponse(
                {
                    "items": [
                        {
                            "title": "상장폐지 및 횡령 공시 안내",
                            "description": "일반 시장 안내 페이지이며 회사명 직접 언급은 없습니다.",
                            "originallink": "https://example.com/guide",
                            "pubDate": "Thu, 14 May 2026 09:00:00 +0900",
                        }
                    ]
                }
            )
        return _FakeResponse({"list": []})

    def post(
        self,
        url: str,
        *,
        json: Mapping[str, object] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        return _FakeResponse({"results": []})


class _DisclosureSnippetSession:
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        if "naver.com" in url:
            return _FakeResponse(
                {
                    "items": [
                        {
                            "title": "테스트기업 [기재정정]감사보고서제출",
                            "description": "DART 공시 제목을 그대로 보여주는 검색 결과입니다.",
                            "originallink": "https://dart.example.com/report",
                            "pubDate": "Thu, 14 May 2026 09:00:00 +0900",
                        }
                    ]
                }
            )
        return _FakeResponse({"list": []})

    def post(
        self,
        url: str,
        *,
        json: Mapping[str, object] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        return _FakeResponse({"results": []})


class _ScopedKeywordSession:
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        if "naver.com" in url:
            return _FakeResponse(
                {
                    "items": [
                        {
                            "title": "재계 인사 동향",
                            "description": "테스트기업의 대표가 선임되었습니다. 다른 회사 임원이 횡령 혐의로 기소되었습니다.",
                            "originallink": "https://example.com/mixed-snippet",
                            "pubDate": "Thu, 14 May 2026 09:00:00 +0900",
                        }
                    ]
                }
            )
        return _FakeResponse({"list": []})

    def post(
        self,
        url: str,
        *,
        json: Mapping[str, object] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        return _FakeResponse({"results": []})


class _DuplicateSession:
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        if "naver.com" in url:
            return _FakeResponse(
                {
                    "items": [
                        {
                            "title": "테스트기업 횡령 의혹 보도",
                            "description": "테스트기업 횡령 관련 후속 보도입니다.",
                            "originallink": "https://news.example.com/article/1?utm=naver",
                            "pubDate": "Thu, 14 May 2026 09:00:00 +0900",
                        }
                    ]
                }
            )
        return _FakeResponse({"list": []})

    def post(
        self,
        url: str,
        *,
        json: Mapping[str, object] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        return _FakeResponse(
            {
                "results": [
                    {
                        "title": "테스트기업 횡령 의혹 보도",
                        "content": "테스트기업 횡령 관련 후속 보도입니다.",
                        "url": "https://news.example.com/article/1?utm=tavily",
                    }
                ]
            }
        )


class _WideWebSession:
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        if "naver.com" in url:
            assert params is not None
            assert params["display"] == 3
            return _FakeResponse(
                {
                    "items": [
                        {
                            "title": "테스트기업 신용등급 관련 뉴스",
                            "description": "테스트기업의 차입금과 유동성 관련 직접 기사입니다.",
                            "originallink": "https://news.example.com/direct",
                            "pubDate": "Thu, 14 May 2026 09:00:00 +0900",
                        },
                        *[
                            {
                                "title": f"일반 횡령 배임 안내 {idx}",
                                "description": "회사명 직접 언급이 없는 일반 안내입니다.",
                                "originallink": f"https://news.example.com/weak-{idx}",
                                "pubDate": "Thu, 14 May 2026 09:00:00 +0900",
                            }
                            for idx in range(1, 5)
                        ],
                    ]
                }
            )
        return _FakeResponse({"list": []})

    def post(
        self,
        url: str,
        *,
        json: Mapping[str, object] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        assert json is not None
        assert json["max_results"] == 5
        return _FakeResponse(
            {
                "results": [
                    {
                        "title": f"웹 일반 상장폐지 안내 {idx}",
                        "content": "기업명 직접 언급이 없는 일반 검색 결과입니다.",
                        "url": f"https://web.example.com/weak-{idx}",
                    }
                    for idx in range(1, 6)
                ]
            }
        )


class _NaverQueryCaptureSession:
    def __init__(self) -> None:
        self.naver_queries: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        if "naver.com" in url:
            assert params is not None
            self.naver_queries.append(str(params["query"]))
            return _FakeResponse({"items": []})
        return _FakeResponse({"list": []})

    def post(
        self,
        url: str,
        *,
        json: Mapping[str, object] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        return _FakeResponse({"results": []})


class _PreferredShareDelistingSession:
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        if "naver.com" in url:
            return _FakeResponse({"items": []})
        return _FakeResponse({"list": []})

    def post(
        self,
        url: str,
        *,
        json: Mapping[str, object] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        return _FakeResponse(
            {
                "results": [
                    {
                        "title": "상장폐지 - 현대모비스 (012330) - KRX 공시",
                        "content": (
                            "상장폐지일 : '15.01.16 - 본 상장폐지 안내는 "
                            "현대모비스1우선주에만 해당되는 사항이며, "
                            "동 안내에 따른 조치사항은 현대모비스보통주에는 영향을 미치지 않습니다."
                        ),
                        "url": "https://kind.krx.co.kr/common/disclsviewer.do",
                    }
                ]
            }
        )


class _OpenDartSession:
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        if "corpCode.xml" in url:
            return _FakeResponse({}, content=_corp_code_zip())
        if "opendart" in url:
            assert params is not None
            assert params["corp_code"] == "00123456"
            assert params["end_de"] == "20240511"
            disclosure_type = str(params.get("pblntf_ty", ""))
            if disclosure_type == "B":
                return _FakeResponse(
                    {
                        "list": [
                            {
                                "report_nm": "횡령ㆍ배임혐의발생",
                                "rcept_no": "202405110001",
                                "rcept_dt": "20240511",
                            }
                        ]
                    }
                )
            if disclosure_type == "F":
                return _FakeResponse(
                    {
                        "list": [
                            {
                                "report_nm": "감사보고서 제출",
                                "rcept_no": "202403310001",
                                "rcept_dt": "20240331",
                            }
                        ]
                    }
                )
            return _FakeResponse({"list": []})
        if "naver.com" in url:
            return _FakeResponse({"items": []})
        return _FakeResponse({"list": []})

    def post(
        self,
        url: str,
        *,
        json: Mapping[str, object] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        return _FakeResponse({"results": []})


class _RoutineOpenDartSession:
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        if "corpCode.xml" in url:
            return _FakeResponse({}, content=_corp_code_zip())
        if "opendart" in url:
            assert params is not None
            if str(params.get("pblntf_ty", "")) == "A":
                return _FakeResponse(
                    {
                        "list": [
                            {
                                "report_nm": "사업보고서 (2023.12)",
                                "rcept_no": "202403300001",
                                "rcept_dt": "20240330",
                            }
                        ]
                    }
                )
            return _FakeResponse({"list": []})
        if "naver.com" in url:
            return _FakeResponse({"items": []})
        return _FakeResponse({"list": []})

    def post(
        self,
        url: str,
        *,
        json: Mapping[str, object] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        return _FakeResponse({"results": []})


class _HistoricalEvidenceFilterSession:
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        if "naver.com" in url:
            return _FakeResponse(
                {
                    "items": [
                        {
                            "title": "테스트기업 과거 유동성 점검",
                            "description": "테스트기업 유동성 관련 과거 기사입니다.",
                            "originallink": "https://news.example.com/past",
                            "pubDate": "Tue, 31 Dec 2024 09:00:00 +0900",
                        },
                        {
                            "title": "테스트기업 미래 횡령 보도",
                            "description": "기준일 이후 기사라 과거 재현 평가에서는 제외해야 합니다.",
                            "originallink": "https://news.example.com/future",
                            "pubDate": "Thu, 14 May 2026 09:00:00 +0900",
                        },
                    ]
                }
            )
        return _FakeResponse({"list": []})

    def post(
        self,
        url: str,
        *,
        json: Mapping[str, object] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        return _FakeResponse(
            {
                "results": [
                    {
                        "title": "테스트기업 과거 차입금 기사",
                        "content": "테스트기업 차입금 관련 기준일 이전 웹 근거입니다.",
                        "url": "https://web.example.com/past",
                        "published_date": "2024-12-30",
                    },
                    {
                        "title": "테스트기업 미래 상장폐지 기사",
                        "content": "기준일 이후 웹 근거입니다.",
                        "url": "https://web.example.com/future",
                        "published_date": "2026-05-14",
                    },
                    {
                        "title": "테스트기업 날짜 없는 웹 결과",
                        "content": "과거 재현 평가에서는 날짜 없는 웹 결과를 제외합니다.",
                        "url": "https://web.example.com/undated",
                    },
                ]
            }
        )


def test_external_evidence_enabled_flag() -> None:
    assert external_evidence_enabled({"CAS_ENABLE_EXTERNAL_EVIDENCE": "1"})
    assert external_evidence_enabled({"CAS_ENABLE_EXTERNAL_EVIDENCE": "true"})
    assert not external_evidence_enabled({})


def test_collect_external_evidence_stays_disabled_by_default() -> None:
    snapshot = collect_external_evidence(company_name="테스트기업", env={})

    assert snapshot["status"] == "disabled"
    assert snapshot["enabled"] is False
    assert snapshot["items"] == []


def test_collect_external_evidence_merges_provider_items() -> None:
    snapshot = collect_external_evidence(
        company_name="테스트기업",
        stock_code="000001",
        env={
            "CAS_ENABLE_EXTERNAL_EVIDENCE": "1",
            "NAVER_CLIENT_ID": "dummy",
            "NAVER_CLIENT_SECRET": "dummy",
        },
        session=_FakeSession(),
    )

    assert snapshot["status"] == "ready"
    assert snapshot["has_critical_risk"] is True
    assert "횡령" in snapshot["critical_terms"]
    assert snapshot["items"]
    first_item = snapshot["items"][0]
    assert isinstance(first_item, dict)
    assert first_item["company_match"] is True
    assert first_item["critical_context_confirmed"] is True
    assert first_item["evidence_quality"] == "high"
    assert float(first_item["evidence_score"]) >= 0.75
    assert "횡령" in first_item["critical_terms"]
    assert snapshot["verified_item_count"] == 1


def test_collect_external_evidence_marks_keyword_only_results_as_weak() -> None:
    snapshot = collect_external_evidence(
        company_name="테스트기업",
        stock_code="000001",
        env={
            "CAS_ENABLE_EXTERNAL_EVIDENCE": "1",
            "NAVER_CLIENT_ID": "dummy",
            "NAVER_CLIENT_SECRET": "dummy",
        },
        session=_WeakKeywordSession(),
    )

    assert snapshot["status"] == "ready"
    assert snapshot["direct_match_count"] == 0
    assert snapshot["verified_item_count"] == 0
    assert snapshot["high_confidence_critical_count"] == 0
    item = snapshot["items"][0]
    assert isinstance(item, dict)
    assert item["company_match"] is False
    assert item["evidence_relevance"] == "weak"
    assert item["evidence_quality"] == "low"
    assert item["veto_candidate"] is False


def test_collect_external_evidence_downgrades_disclosure_title_snippets() -> None:
    snapshot = collect_external_evidence(
        company_name="테스트기업",
        stock_code="000001",
        env={
            "CAS_ENABLE_EXTERNAL_EVIDENCE": "1",
            "NAVER_CLIENT_ID": "dummy",
            "NAVER_CLIENT_SECRET": "dummy",
        },
        session=_DisclosureSnippetSession(),
    )

    assert snapshot["status"] == "ready"
    assert snapshot["has_critical_risk"] is False
    assert snapshot["veto_candidate_count"] == 0
    item = snapshot["items"][0]
    assert item["source"] == "naver_news"
    assert item["disclosure_severity"] == "caution"
    assert item["critical_terms"] == []
    assert item["veto_candidate"] is False
    assert float(item["evidence_score"]) <= 0.68


def test_collect_external_evidence_requires_company_and_keyword_same_context() -> None:
    snapshot = collect_external_evidence(
        company_name="테스트기업",
        stock_code="000001",
        env={
            "CAS_ENABLE_EXTERNAL_EVIDENCE": "1",
            "NAVER_CLIENT_ID": "dummy",
            "NAVER_CLIENT_SECRET": "dummy",
        },
        session=_ScopedKeywordSession(),
    )

    assert snapshot["status"] == "ready"
    assert snapshot["veto_candidate_count"] == 0
    assert snapshot["high_confidence_critical_count"] == 0
    item = snapshot["items"][0]
    assert item["company_match"] is True
    assert item["critical_terms"] == ["횡령"]
    assert item["critical_context_confirmed"] is False
    assert item["veto_candidate"] is False
    assert item["evidence_quality"] == "low"
    assert float(item["evidence_score"]) < 0.55


def test_collect_external_evidence_deduplicates_same_article_url() -> None:
    snapshot = collect_external_evidence(
        company_name="테스트기업",
        stock_code="000001",
        env={
            "CAS_ENABLE_EXTERNAL_EVIDENCE": "1",
            "NAVER_CLIENT_ID": "dummy",
            "NAVER_CLIENT_SECRET": "dummy",
            "TAVILY_API_KEY": "dummy",
        },
        session=_DuplicateSession(),
    )

    assert snapshot["status"] == "ready"
    assert len(snapshot["items"]) == 1
    item = snapshot["items"][0]
    assert isinstance(item, dict)
    assert item["duplicate_count"] == 2
    assert "duplicate_merged" in item["verification_flags"]
    verification_summary = snapshot["verification_summary"]
    assert isinstance(verification_summary, dict)
    assert verification_summary["duplicate_merged_count"] == 1


def test_collect_external_evidence_prioritizes_direct_news_and_limits_weak_web() -> None:
    snapshot = collect_external_evidence(
        company_name="테스트기업",
        stock_code="000001",
        env={
            "CAS_ENABLE_EXTERNAL_EVIDENCE": "1",
            "NAVER_CLIENT_ID": "dummy",
            "NAVER_CLIENT_SECRET": "dummy",
            "TAVILY_API_KEY": "dummy",
        },
        session=_WideWebSession(),
    )

    assert snapshot["status"] == "ready"
    items = snapshot["items"]
    assert items[0]["company_match"] is True
    assert items[0]["source"] == "naver_news"
    weak_web_items = [
        item
        for item in items
        if item["source"] in {"naver_news", "tavily"}
        and item["company_match"] is False
        and item["evidence_quality"] == "low"
    ]
    assert len(weak_web_items) == 3
    verification_summary = snapshot["verification_summary"]
    assert isinstance(verification_summary, dict)
    assert verification_summary["weak_web_item_count"] == 3


def test_collect_external_evidence_uses_short_naver_keyword_queries() -> None:
    session = _NaverQueryCaptureSession()

    snapshot = collect_external_evidence(
        company_name="삼성전자(주)",
        stock_code="005930",
        env={
            "CAS_ENABLE_EXTERNAL_EVIDENCE": "1",
            "NAVER_CLIENT_ID": "dummy",
            "NAVER_CLIENT_SECRET": "dummy",
        },
        session=session,
    )

    assert snapshot["status"] == "no_results"
    assert session.naver_queries
    assert "삼성전자 소송" in session.naver_queries
    assert "삼성전자 회사채" in session.naver_queries
    assert all("005930" not in query for query in session.naver_queries)
    assert snapshot["naver_queries"] == session.naver_queries
    provider = snapshot["providers"]["naver_news"]
    assert isinstance(provider, dict)
    assert provider["queries"] == session.naver_queries


def test_collect_external_evidence_does_not_veto_preferred_share_delisting() -> None:
    snapshot = collect_external_evidence(
        company_name="현대모비스(주)",
        stock_code="012330",
        env={
            "CAS_ENABLE_EXTERNAL_EVIDENCE": "1",
            "TAVILY_API_KEY": "dummy",
        },
        session=_PreferredShareDelistingSession(),
    )

    assert snapshot["status"] == "ready"
    assert snapshot["veto_candidate_count"] == 0
    assert snapshot["high_confidence_critical_count"] == 0
    item = snapshot["items"][0]
    assert item["company_match"] is True
    assert item["critical_terms"] == ["상장폐지"]
    assert item["critical_context_confirmed"] is False
    assert item["veto_candidate"] is False


def test_collect_external_evidence_resolves_corp_code_and_collects_opendart(
    tmp_path: Path,
) -> None:
    snapshot = collect_external_evidence(
        company_name="테스트기업",
        stock_code="000001",
        as_of_date="2024-05-11",
        env={
            "CAS_ENABLE_EXTERNAL_EVIDENCE": "1",
            "OPENDART_API_KEY": "dummy",
            "CAS_OPENDART_CORP_CODE_CACHE_PATH": str(tmp_path / "corp_codes.csv"),
        },
        session=_OpenDartSession(),
    )

    assert snapshot["status"] == "ready"
    assert snapshot["corp_code"] == "00123456"
    assert snapshot["as_of_date"] == "2024-05-11"
    provider = snapshot["providers"]["opendart"]
    assert isinstance(provider, dict)
    assert provider["status"] == "ready"
    assert provider["query_window"]["end_date"] == "2024-05-11"
    dart_items = [item for item in snapshot["items"] if item["source"] == "opendart"]
    assert len(dart_items) == 2
    assert dart_items[0]["provider_relevance"] == "risk"
    assert dart_items[0]["disclosure_severity"] == "veto"
    assert dart_items[0]["veto_candidate"] is True
    assert dart_items[0]["disclosure_type"] == "B"
    assert dart_items[1]["disclosure_severity"] == "caution"
    assert dart_items[1]["disclosure_type"] == "F"


def test_collect_external_evidence_treats_routine_opendart_as_context(
    tmp_path: Path,
) -> None:
    snapshot = collect_external_evidence(
        company_name="테스트기업",
        stock_code="000001",
        as_of_date="2024-05-11",
        env={
            "CAS_ENABLE_EXTERNAL_EVIDENCE": "1",
            "OPENDART_API_KEY": "dummy",
            "CAS_OPENDART_CORP_CODE_CACHE_PATH": str(tmp_path / "corp_codes.csv"),
        },
        session=_RoutineOpenDartSession(),
    )

    assert snapshot["status"] == "ready"
    assert snapshot["has_critical_risk"] is False
    assert snapshot["verified_item_count"] == 0
    assert snapshot["veto_candidate_count"] == 0
    item = snapshot["items"][0]
    assert item["source"] == "opendart"
    assert item["provider_relevance"] == "routine"
    assert item["disclosure_severity"] == "routine"
    assert item["critical_terms"] == []
    assert item["veto_candidate"] is False
    assert float(item["evidence_score"]) < 0.55


def test_collect_external_evidence_filters_web_items_after_historical_as_of_date() -> None:
    snapshot = collect_external_evidence(
        company_name="테스트기업",
        stock_code="000001",
        as_of_date="2024-12-31",
        env={
            "CAS_ENABLE_EXTERNAL_EVIDENCE": "1",
            "NAVER_CLIENT_ID": "dummy",
            "NAVER_CLIENT_SECRET": "dummy",
            "TAVILY_API_KEY": "dummy",
        },
        session=_HistoricalEvidenceFilterSession(),
    )

    assert snapshot["status"] == "ready"
    urls = {str(item["url"]) for item in snapshot["items"]}
    assert "https://news.example.com/past" in urls
    assert "https://web.example.com/past" in urls
    assert "https://news.example.com/future" not in urls
    assert "https://web.example.com/future" not in urls
    assert "https://web.example.com/undated" not in urls
    providers = snapshot["providers"]
    assert isinstance(providers, dict)
    naver_filter = providers["naver_news"]["as_of_date_filter"]
    tavily_filter = providers["tavily"]["as_of_date_filter"]
    assert naver_filter["historical_mode"] is True
    assert naver_filter["filtered_after_cutoff_count"] >= 1
    assert tavily_filter["filtered_undated_count"] == 1


def _corp_code_zip() -> bytes:
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<result>
  <list>
    <corp_code>00123456</corp_code>
    <corp_name>테스트기업</corp_name>
    <stock_code>000001</stock_code>
    <modify_date>20240501</modify_date>
  </list>
</result>
"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("CORPCODE.xml", xml.encode("utf-8"))
    return buffer.getvalue()
