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


class _CountingFakeSession(_FakeSession):
    def __init__(self) -> None:
        self.get_count = 0
        self.post_count = 0

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        self.get_count += 1
        return super().get(url, params=params, headers=headers, timeout=timeout)

    def post(
        self,
        url: str,
        *,
        json: Mapping[str, object] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        self.post_count += 1
        return super().post(url, json=json, timeout=timeout)


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


class _AggregatedDisclosureNewsSession:
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
                            "title": "[전일 주요 공시] 한미약품ㆍ테스트기업 등",
                            "description": (
                                "임직원 횡령ㆍ배임 유죄 △다른기업, 무상증자 권리락 "
                                "△테스트기업, 현저한 시황변동 관련 조회공시"
                            ),
                            "originallink": "https://example.com/market-wrap",
                            "pubDate": "Fri, 13 Dec 2019 08:06:00 +0900",
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


class _FutureOpenDartLeakSession(_OpenDartSession):
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
            if str(params.get("pblntf_ty", "")) == "B":
                return _FakeResponse(
                    {
                        "list": [
                            {
                                "report_nm": "횡령ㆍ배임혐의발생",
                                "rcept_no": "202601010001",
                                "rcept_dt": "20260101",
                            }
                        ]
                    }
                )
            return _FakeResponse({"list": []})
        if "naver.com" in url:
            return _FakeResponse({"items": []})
        return _FakeResponse({"list": []})


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


class _BenignTradingHaltOpenDartSession:
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
            if str(params.get("pblntf_ty", "")) == "B":
                return _FakeResponse(
                    {
                        "list": [
                            {
                                "report_nm": "주권매매거래정지(무상증자)",
                                "rcept_no": "202012310001",
                                "rcept_dt": "20201231",
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


class _ProceduralDisclosureOpenDartSession:
    def __init__(self, report_name: str, *, disclosure_type: str = "B") -> None:
        self.report_name = report_name
        self.disclosure_type = disclosure_type

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
            if str(params.get("pblntf_ty", "")) == self.disclosure_type:
                return _FakeResponse(
                    {
                        "list": [
                            {
                                "report_nm": self.report_name,
                                "rcept_no": "202012310001",
                                "rcept_dt": "20201231",
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


class _DetailMaterialityOpenDartSession:
    def __init__(
        self,
        report_name: str,
        *,
        document_text: str = "",
        bsn_sp_rows: list[dict[str, object]] | None = None,
        disclosure_type: str = "B",
    ) -> None:
        self.report_name = report_name
        self.document_text = document_text
        self.bsn_sp_rows = bsn_sp_rows or []
        self.disclosure_type = disclosure_type

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
        if "bsnSp.json" in url:
            assert params is not None
            assert params["corp_code"] == "00123456"
            return _FakeResponse({"list": self.bsn_sp_rows})
        if "document.xml" in url:
            assert params is not None
            assert params["rcept_no"] == "202012310001"
            return _FakeResponse({}, content=_document_zip(self.document_text))
        if "opendart" in url:
            assert params is not None
            if str(params.get("pblntf_ty", "")) == self.disclosure_type:
                return _FakeResponse(
                    {
                        "list": [
                            {
                                "report_nm": self.report_name,
                                "rcept_no": "202012310001",
                                "rcept_dt": "20201231",
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
    assert snapshot["veto_candidate_count"] == 0
    assert snapshot["high_confidence_critical_count"] == 0
    assert snapshot["items"]
    first_item = snapshot["items"][0]
    assert isinstance(first_item, dict)
    assert str(first_item["event_id"]).startswith("evt_")
    assert first_item["source_evidence_type"] == "news_search_snippet"
    assert first_item["source_reliability"] == "medium_contextual_snippet"
    assert first_item["company_disambiguation"] == "name_only_search_result"
    assert first_item["temporal_status"] == "on_or_before_as_of_date"
    assert first_item["as_of_date_violation"] is False
    assert first_item["company_match"] is True
    assert first_item["critical_context_confirmed"] is True
    assert first_item["veto_candidate"] is False
    assert first_item["evidence_quality"] == "high"
    assert float(first_item["evidence_score"]) >= 0.75
    assert "횡령" in first_item["critical_terms"]
    assert snapshot["verified_item_count"] == 1
    verification_summary = snapshot["verification_summary"]
    assert isinstance(verification_summary, dict)
    assert verification_summary["event_count"] == 1
    assert verification_summary["company_disambiguation_counts"]["name_only_search_result"] == 1


def test_collect_external_evidence_reuses_cached_snapshot(tmp_path: Path) -> None:
    session = _CountingFakeSession()
    env = {
        "CAS_ENABLE_EXTERNAL_EVIDENCE": "1",
        "CAS_EXTERNAL_EVIDENCE_CACHE_ENABLED": "1",
        "CAS_EXTERNAL_EVIDENCE_CACHE_SESSION": "1",
        "CAS_STAGE2_CACHE_DIR": str(tmp_path),
        "NAVER_CLIENT_ID": "dummy",
        "NAVER_CLIENT_SECRET": "dummy",
    }

    first_snapshot = collect_external_evidence(
        company_name="테스트기업",
        stock_code="000001",
        env=env,
        session=session,
    )
    first_get_count = session.get_count
    second_snapshot = collect_external_evidence(
        company_name="테스트기업",
        stock_code="000001",
        env=env,
        session=session,
    )

    assert first_snapshot["cache_hit"] is False
    assert second_snapshot["cache_hit"] is True
    assert session.get_count == first_get_count
    assert second_snapshot["items"]


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
    assert snapshot["has_critical_risk"] is False
    assert snapshot["critical_terms"] == []
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
    assert snapshot["has_critical_risk"] is False
    assert snapshot["critical_terms"] == []
    assert snapshot["veto_candidate_count"] == 0
    assert snapshot["high_confidence_critical_count"] == 0
    item = snapshot["items"][0]
    assert item["company_match"] is True
    assert item["critical_terms"] == ["횡령"]
    assert item["critical_context_confirmed"] is False
    assert item["veto_candidate"] is False
    assert item["evidence_quality"] == "low"
    assert float(item["evidence_score"]) < 0.55


def test_collect_external_evidence_does_not_confirm_aggregated_news_list_terms() -> None:
    snapshot = collect_external_evidence(
        company_name="테스트기업",
        stock_code="000001",
        as_of_date="2022-12-31",
        env={
            "CAS_ENABLE_EXTERNAL_EVIDENCE": "1",
            "NAVER_CLIENT_ID": "dummy",
            "NAVER_CLIENT_SECRET": "dummy",
        },
        session=_AggregatedDisclosureNewsSession(),
    )

    assert snapshot["status"] == "ready"
    assert snapshot["has_critical_risk"] is False
    assert snapshot["veto_candidate_count"] == 0
    assert snapshot["high_confidence_critical_count"] == 0
    item = snapshot["items"][0]
    assert item["company_match"] is True
    assert item["critical_terms"] == ["배임", "횡령"]
    assert item["critical_context_confirmed"] is False
    assert item["veto_candidate"] is False
    assert item["evidence_quality"] == "low"


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
    assert str(item["event_id"]).startswith("evt_")
    assert item["duplicate_count"] == 2
    assert item["duplicate_sources"] == ["naver_news", "tavily"]
    assert "duplicate_merged" in item["verification_flags"]
    verification_summary = snapshot["verification_summary"]
    assert isinstance(verification_summary, dict)
    assert verification_summary["event_count"] == 1
    assert verification_summary["duplicate_merged_count"] == 1
    assert verification_summary["source_evidence_type_counts"]["news_search_snippet"] == 1


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
    assert str(dart_items[0]["event_id"]).startswith("evt_")
    assert dart_items[0]["source_evidence_type"] == "direct_disclosure"
    assert dart_items[0]["source_reliability"] == "high_direct_disclosure"
    assert dart_items[0]["company_disambiguation"] == "resolved_by_disclosure_corp_code"
    assert dart_items[0]["temporal_status"] == "on_or_before_as_of_date"
    assert dart_items[0]["provider_relevance"] == "risk"
    assert dart_items[0]["disclosure_severity"] == "veto"
    assert dart_items[0]["veto_candidate"] is True
    assert dart_items[0]["disclosure_type"] == "B"
    assert dart_items[1]["disclosure_severity"] == "caution"
    assert dart_items[1]["disclosure_type"] == "F"


def test_collect_external_evidence_marks_defensive_as_of_date_leaks(
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
        session=_FutureOpenDartLeakSession(),
    )

    assert snapshot["status"] == "ready"
    assert snapshot["has_critical_risk"] is False
    assert snapshot["veto_candidate_count"] == 0
    assert snapshot["high_confidence_critical_count"] == 0
    item = snapshot["items"][0]
    assert item["source"] == "opendart"
    assert item["temporal_status"] == "after_as_of_date"
    assert item["as_of_date_violation"] is True
    assert item["critical_context_confirmed"] is False
    assert item["veto_candidate"] is False
    assert float(item["evidence_score"]) <= 0.24
    assert "as_of_date_violation" in item["verification_flags"]
    verification_summary = snapshot["verification_summary"]
    assert isinstance(verification_summary, dict)
    assert verification_summary["as_of_date_violation_count"] == 1


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


def test_collect_external_evidence_downgrades_benign_trading_halt_opendart(
    tmp_path: Path,
) -> None:
    snapshot = collect_external_evidence(
        company_name="테스트기업",
        stock_code="000001",
        as_of_date="2020-12-31",
        env={
            "CAS_ENABLE_EXTERNAL_EVIDENCE": "1",
            "OPENDART_API_KEY": "dummy",
            "CAS_OPENDART_CORP_CODE_CACHE_PATH": str(tmp_path / "corp_codes.csv"),
        },
        session=_BenignTradingHaltOpenDartSession(),
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


def test_collect_external_evidence_downgrades_low_materiality_litigation_opendart(
    tmp_path: Path,
) -> None:
    snapshot = collect_external_evidence(
        company_name="테스트기업",
        stock_code="000001",
        as_of_date="2020-12-31",
        env={
            "CAS_ENABLE_EXTERNAL_EVIDENCE": "1",
            "OPENDART_API_KEY": "dummy",
            "CAS_OPENDART_CORP_CODE_CACHE_PATH": str(tmp_path / "corp_codes.csv"),
        },
        session=_ProceduralDisclosureOpenDartSession(
            "소송등의판결ㆍ결정(자율공시:일정금액미만의청구)"
        ),
    )

    assert snapshot["status"] == "ready"
    assert snapshot["has_critical_risk"] is False
    assert snapshot["veto_candidate_count"] == 0
    item = snapshot["items"][0]
    assert item["source"] == "opendart"
    assert item["provider_relevance"] == "caution"
    assert item["disclosure_severity"] == "caution"
    assert item["disclosure_event_class"] == "low_materiality_litigation"
    assert item["disclosure_materiality"] == "procedural_or_one_off"
    assert item["critical_terms"] == []
    assert item["veto_candidate"] is False
    assert float(item["evidence_score"]) <= 0.68


def test_collect_external_evidence_downgrades_voluntary_contract_cancellation_opendart(
    tmp_path: Path,
) -> None:
    snapshot = collect_external_evidence(
        company_name="테스트기업",
        stock_code="000001",
        as_of_date="2020-12-31",
        env={
            "CAS_ENABLE_EXTERNAL_EVIDENCE": "1",
            "OPENDART_API_KEY": "dummy",
            "CAS_OPENDART_CORP_CODE_CACHE_PATH": str(tmp_path / "corp_codes.csv"),
        },
        session=_ProceduralDisclosureOpenDartSession("단일판매ㆍ공급계약해지(자율공시)"),
    )

    assert snapshot["status"] == "ready"
    assert snapshot["has_critical_risk"] is False
    assert snapshot["veto_candidate_count"] == 0
    item = snapshot["items"][0]
    assert item["provider_relevance"] == "caution"
    assert item["disclosure_severity"] == "caution"
    assert item["disclosure_event_class"] == "one_off_contract_cancellation"
    assert item["disclosure_materiality"] == "procedural_or_one_off"
    assert item["critical_terms"] == []
    assert item["veto_candidate"] is False


def test_collect_external_evidence_downgrades_spac_merger_halt_opendart(
    tmp_path: Path,
) -> None:
    snapshot = collect_external_evidence(
        company_name="테스트기업",
        stock_code="000001",
        as_of_date="2020-12-31",
        env={
            "CAS_ENABLE_EXTERNAL_EVIDENCE": "1",
            "OPENDART_API_KEY": "dummy",
            "CAS_OPENDART_CORP_CODE_CACHE_PATH": str(tmp_path / "corp_codes.csv"),
        },
        session=_ProceduralDisclosureOpenDartSession(
            "주권매매거래정지(SPAC 합병(예비심사청구대상))"
        ),
    )

    assert snapshot["status"] == "ready"
    assert snapshot["has_critical_risk"] is False
    assert snapshot["veto_candidate_count"] == 0
    item = snapshot["items"][0]
    assert item["provider_relevance"] == "caution"
    assert item["disclosure_severity"] == "caution"
    assert item["disclosure_event_class"] == "procedural_trading_halt"
    assert item["disclosure_materiality"] == "procedural_or_one_off"
    assert item["critical_terms"] == []
    assert item["veto_candidate"] is False


def test_collect_external_evidence_keeps_material_contract_cancellation_adverse(
    tmp_path: Path,
) -> None:
    snapshot = collect_external_evidence(
        company_name="테스트기업",
        stock_code="000001",
        as_of_date="2020-12-31",
        env={
            "CAS_ENABLE_EXTERNAL_EVIDENCE": "1",
            "OPENDART_API_KEY": "dummy",
            "CAS_OPENDART_CORP_CODE_CACHE_PATH": str(tmp_path / "corp_codes.csv"),
        },
        session=_ProceduralDisclosureOpenDartSession("단일판매ㆍ공급계약해지"),
    )

    assert snapshot["status"] == "ready"
    item = snapshot["items"][0]
    assert item["provider_relevance"] == "risk"
    assert item["disclosure_severity"] == "adverse"
    assert item["disclosure_event_class"] == "material_contract_cancellation"
    assert item["disclosure_materiality"] == "substantive_adverse"
    assert item["evidence_quality"] == "high"
    assert item["veto_candidate"] is False


def test_collect_external_evidence_downgrades_low_ratio_contract_cancellation(
    tmp_path: Path,
) -> None:
    document_text = """
<DOCUMENT>
  <TABLE>
    <TR><TD>계약해지금액</TD><TD>240,000,000</TD></TR>
    <TR><TD>최근매출액 대비</TD><TD>2.4%</TD></TR>
  </TABLE>
</DOCUMENT>
"""
    snapshot = collect_external_evidence(
        company_name="테스트기업",
        stock_code="000001",
        as_of_date="2020-12-31",
        env={
            "CAS_ENABLE_EXTERNAL_EVIDENCE": "1",
            "OPENDART_API_KEY": "dummy",
            "CAS_OPENDART_CORP_CODE_CACHE_PATH": str(tmp_path / "corp_codes.csv"),
        },
        session=_DetailMaterialityOpenDartSession(
            "단일판매ㆍ공급계약해지",
            document_text=document_text,
        ),
    )

    assert snapshot["status"] == "ready"
    assert snapshot["has_critical_risk"] is False
    assert snapshot["veto_candidate_count"] == 0
    item = snapshot["items"][0]
    assert item["provider_relevance"] == "caution"
    assert item["disclosure_severity"] == "caution"
    assert item["disclosure_event_class"] == "low_materiality_contract_cancellation"
    assert item["disclosure_materiality"] == "procedural_or_one_off"
    assert item["materiality_ratio"] == "0.0240"
    assert item["materiality_source"] == "opendart_document_xml"
    assert "2.40%" in item["materiality_basis"]
    assert item["critical_terms"] == []
    assert item["veto_candidate"] is False
    assert float(item["evidence_score"]) <= 0.68


def test_collect_external_evidence_keeps_high_ratio_contract_cancellation_adverse(
    tmp_path: Path,
) -> None:
    document_text = """
<DOCUMENT>
  <TABLE>
    <TR><TD>계약해지금액</TD><TD>1,520,000,000</TD></TR>
    <TR><TD>최근매출액 대비</TD><TD>15.2%</TD></TR>
  </TABLE>
</DOCUMENT>
"""
    snapshot = collect_external_evidence(
        company_name="테스트기업",
        stock_code="000001",
        as_of_date="2020-12-31",
        env={
            "CAS_ENABLE_EXTERNAL_EVIDENCE": "1",
            "OPENDART_API_KEY": "dummy",
            "CAS_OPENDART_CORP_CODE_CACHE_PATH": str(tmp_path / "corp_codes.csv"),
        },
        session=_DetailMaterialityOpenDartSession(
            "단일판매ㆍ공급계약해지",
            document_text=document_text,
        ),
    )

    item = snapshot["items"][0]
    assert item["provider_relevance"] == "risk"
    assert item["disclosure_severity"] == "adverse"
    assert item["disclosure_event_class"] == "material_contract_cancellation"
    assert item["disclosure_materiality"] == "substantive_adverse"
    assert item["materiality_ratio"] == "0.1520"
    assert item["evidence_quality"] == "high"


def test_collect_external_evidence_downgrades_low_ratio_financing(
    tmp_path: Path,
) -> None:
    document_text = """
<DOCUMENT>
  <TABLE>
    <TR><TD>발행금액</TD><TD>250,000,000</TD></TR>
    <TR><TD>자기자본</TD><TD>10,000,000,000</TD></TR>
  </TABLE>
</DOCUMENT>
"""
    snapshot = collect_external_evidence(
        company_name="테스트기업",
        stock_code="000001",
        as_of_date="2020-12-31",
        env={
            "CAS_ENABLE_EXTERNAL_EVIDENCE": "1",
            "OPENDART_API_KEY": "dummy",
            "CAS_OPENDART_CORP_CODE_CACHE_PATH": str(tmp_path / "corp_codes.csv"),
        },
        session=_DetailMaterialityOpenDartSession(
            "유상증자결정",
            document_text=document_text,
        ),
    )

    assert snapshot["status"] == "ready"
    assert snapshot["has_critical_risk"] is False
    assert snapshot["veto_candidate_count"] == 0
    item = snapshot["items"][0]
    assert item["provider_relevance"] == "caution"
    assert item["disclosure_severity"] == "caution"
    assert item["disclosure_event_class"] == "low_materiality_financing"
    assert item["disclosure_materiality"] == "procedural_or_one_off"
    assert item["materiality_ratio"] == "0.0250"
    assert "발행금액/자기자본" in item["materiality_basis"]
    assert item["critical_terms"] == []


def test_collect_external_evidence_keeps_high_dilution_financing_adverse(
    tmp_path: Path,
) -> None:
    document_text = """
<DOCUMENT>
  <TABLE>
    <TR><TD>발행금액</TD><TD>600,000,000</TD></TR>
    <TR><TD>자기자본</TD><TD>20,000,000,000</TD></TR>
    <TR><TD>증자비율</TD><TD>15.5%</TD></TR>
  </TABLE>
</DOCUMENT>
"""
    snapshot = collect_external_evidence(
        company_name="테스트기업",
        stock_code="000001",
        as_of_date="2020-12-31",
        env={
            "CAS_ENABLE_EXTERNAL_EVIDENCE": "1",
            "OPENDART_API_KEY": "dummy",
            "CAS_OPENDART_CORP_CODE_CACHE_PATH": str(tmp_path / "corp_codes.csv"),
        },
        session=_DetailMaterialityOpenDartSession(
            "유상증자결정",
            document_text=document_text,
        ),
    )

    item = snapshot["items"][0]
    assert item["provider_relevance"] == "risk"
    assert item["disclosure_severity"] == "adverse"
    assert item["disclosure_event_class"] == "material_financing"
    assert item["disclosure_materiality"] == "substantive_adverse"
    assert item["materiality_ratio"] == "0.1550"
    assert item["dilution_ratio"] == "0.1550"
    assert "희석률" in item["materiality_basis"]
    assert item["evidence_quality"] == "high"


def test_collect_external_evidence_downgrades_low_ratio_debt_guarantee(
    tmp_path: Path,
) -> None:
    document_text = """
<DOCUMENT>
  <TABLE>
    <TR><TD>채무보증금액</TD><TD>200,000,000</TD></TR>
    <TR><TD>자기자본</TD><TD>10,000,000,000</TD></TR>
  </TABLE>
</DOCUMENT>
"""
    snapshot = collect_external_evidence(
        company_name="테스트기업",
        stock_code="000001",
        as_of_date="2020-12-31",
        env={
            "CAS_ENABLE_EXTERNAL_EVIDENCE": "1",
            "OPENDART_API_KEY": "dummy",
            "CAS_OPENDART_CORP_CODE_CACHE_PATH": str(tmp_path / "corp_codes.csv"),
        },
        session=_DetailMaterialityOpenDartSession(
            "타인에대한채무보증결정",
            document_text=document_text,
        ),
    )

    item = snapshot["items"][0]
    assert item["provider_relevance"] == "caution"
    assert item["disclosure_severity"] == "caution"
    assert item["disclosure_event_class"] == "low_materiality_debt_guarantee"
    assert item["disclosure_materiality"] == "procedural_or_one_off"
    assert item["materiality_ratio"] == "0.0200"
    assert "채무보증금액/자기자본" in item["materiality_basis"]
    assert item["veto_candidate"] is False


def test_collect_external_evidence_keeps_high_ratio_litigation_adverse(
    tmp_path: Path,
) -> None:
    document_text = """
<DOCUMENT>
  <TABLE>
    <TR><TD>청구금액</TD><TD>1,200,000,000</TD></TR>
    <TR><TD>자기자본</TD><TD>10,000,000,000</TD></TR>
  </TABLE>
</DOCUMENT>
"""
    snapshot = collect_external_evidence(
        company_name="테스트기업",
        stock_code="000001",
        as_of_date="2020-12-31",
        env={
            "CAS_ENABLE_EXTERNAL_EVIDENCE": "1",
            "OPENDART_API_KEY": "dummy",
            "CAS_OPENDART_CORP_CODE_CACHE_PATH": str(tmp_path / "corp_codes.csv"),
        },
        session=_DetailMaterialityOpenDartSession(
            "소송등의제기",
            document_text=document_text,
        ),
    )

    item = snapshot["items"][0]
    assert item["provider_relevance"] == "risk"
    assert item["disclosure_severity"] == "adverse"
    assert item["disclosure_event_class"] == "material_litigation"
    assert item["disclosure_materiality"] == "substantive_adverse"
    assert item["materiality_ratio"] == "0.1200"
    assert "청구금액/자기자본" in item["materiality_basis"]
    assert item["evidence_quality"] == "high"


def test_collect_external_evidence_downgrades_low_ratio_subsidiary_business_suspension(
    tmp_path: Path,
) -> None:
    snapshot = collect_external_evidence(
        company_name="테스트기업",
        stock_code="000001",
        as_of_date="2020-12-31",
        env={
            "CAS_ENABLE_EXTERNAL_EVIDENCE": "1",
            "OPENDART_API_KEY": "dummy",
            "CAS_OPENDART_CORP_CODE_CACHE_PATH": str(tmp_path / "corp_codes.csv"),
        },
        session=_DetailMaterialityOpenDartSession(
            "영업정지(종속회사의주요경영사항)",
            bsn_sp_rows=[
                {
                    "rcept_no": "202012310001",
                    "bsnsp_amt": "250,000,000",
                    "rsl": "10,000,000,000",
                    "sl_vs": "2.5",
                    "bsnsp_cn": "종속회사 생산라인 일부 영업정지",
                }
            ],
        ),
    )

    assert snapshot["status"] == "ready"
    assert snapshot["has_critical_risk"] is False
    assert snapshot["veto_candidate_count"] == 0
    item = snapshot["items"][0]
    assert item["provider_relevance"] == "caution"
    assert item["disclosure_severity"] == "caution"
    assert item["disclosure_event_class"] == "subsidiary_business_suspension_low_materiality"
    assert item["disclosure_materiality"] == "procedural_or_one_off"
    assert item["business_suspension_scope"] == "subsidiary"
    assert item["materiality_ratio"] == "0.0250"
    assert item["materiality_source"] == "opendart_bsnSp"
    assert item["critical_terms"] == []


def test_collect_external_evidence_downgrades_business_suspension_from_document_fallback(
    tmp_path: Path,
) -> None:
    document_text = """
<DOCUMENT>
  <TABLE>
    <TR><TD>영업정지 내용</TD><TD>종속회사 생산라인 일부 영업정지</TD></TR>
    <TR><TD>영업정지금액</TD><TD>280,000,000</TD></TR>
    <TR><TD>최근매출액 대비</TD><TD>2.8%</TD></TR>
  </TABLE>
</DOCUMENT>
"""
    snapshot = collect_external_evidence(
        company_name="테스트기업",
        stock_code="000001",
        as_of_date="2020-12-31",
        env={
            "CAS_ENABLE_EXTERNAL_EVIDENCE": "1",
            "OPENDART_API_KEY": "dummy",
            "CAS_OPENDART_CORP_CODE_CACHE_PATH": str(tmp_path / "corp_codes.csv"),
        },
        session=_DetailMaterialityOpenDartSession(
            "영업정지(종속회사의주요경영사항)",
            document_text=document_text,
            bsn_sp_rows=[],
        ),
    )

    assert snapshot["status"] == "ready"
    assert snapshot["has_critical_risk"] is False
    assert snapshot["veto_candidate_count"] == 0
    item = snapshot["items"][0]
    assert item["provider_relevance"] == "caution"
    assert item["disclosure_severity"] == "caution"
    assert item["disclosure_event_class"] == "subsidiary_business_suspension_low_materiality"
    assert item["disclosure_materiality"] == "procedural_or_one_off"
    assert item["business_suspension_scope"] == "subsidiary"
    assert item["materiality_ratio"] == "0.0280"
    assert item["materiality_source"] == "opendart_document_xml"
    assert "2.80%" in item["materiality_basis"]
    assert item["critical_terms"] == []


def test_collect_external_evidence_keeps_high_ratio_business_suspension_from_document_fallback(
    tmp_path: Path,
) -> None:
    document_text = """
<DOCUMENT>
  <TABLE>
    <TR><TD>영업정지 내용</TD><TD>주요 사업부 영업정지</TD></TR>
    <TR><TD>영업정지금액</TD><TD>1,250,000,000</TD></TR>
    <TR><TD>최근매출액 대비</TD><TD>12.5%</TD></TR>
  </TABLE>
</DOCUMENT>
"""
    snapshot = collect_external_evidence(
        company_name="테스트기업",
        stock_code="000001",
        as_of_date="2020-12-31",
        env={
            "CAS_ENABLE_EXTERNAL_EVIDENCE": "1",
            "OPENDART_API_KEY": "dummy",
            "CAS_OPENDART_CORP_CODE_CACHE_PATH": str(tmp_path / "corp_codes.csv"),
        },
        session=_DetailMaterialityOpenDartSession(
            "영업정지",
            document_text=document_text,
            bsn_sp_rows=[],
        ),
    )

    item = snapshot["items"][0]
    assert item["provider_relevance"] == "risk"
    assert item["disclosure_severity"] == "adverse"
    assert item["disclosure_event_class"] == "substantive_adverse"
    assert item["disclosure_materiality"] == "substantive_adverse"
    assert item["business_suspension_scope"] == "parent_or_direct"
    assert item["materiality_ratio"] == "0.1250"
    assert item["materiality_source"] == "opendart_document_xml"
    assert item["evidence_quality"] == "high"


def test_collect_external_evidence_keeps_high_ratio_business_suspension_adverse(
    tmp_path: Path,
) -> None:
    snapshot = collect_external_evidence(
        company_name="테스트기업",
        stock_code="000001",
        as_of_date="2020-12-31",
        env={
            "CAS_ENABLE_EXTERNAL_EVIDENCE": "1",
            "OPENDART_API_KEY": "dummy",
            "CAS_OPENDART_CORP_CODE_CACHE_PATH": str(tmp_path / "corp_codes.csv"),
        },
        session=_DetailMaterialityOpenDartSession(
            "영업정지",
            bsn_sp_rows=[
                {
                    "rcept_no": "202012310001",
                    "bsnsp_amt": "1,520,000,000",
                    "rsl": "10,000,000,000",
                    "sl_vs": "15.2",
                    "bsnsp_cn": "주요 사업부 영업정지",
                }
            ],
        ),
    )

    item = snapshot["items"][0]
    assert item["provider_relevance"] == "risk"
    assert item["disclosure_severity"] == "adverse"
    assert item["disclosure_event_class"] == "substantive_adverse"
    assert item["disclosure_materiality"] == "substantive_adverse"
    assert item["business_suspension_scope"] == "parent_or_direct"
    assert item["materiality_ratio"] == "0.1520"
    assert item["evidence_quality"] == "high"


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


def _document_zip(text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("document.xml", text.encode("utf-8"))
    return buffer.getvalue()
