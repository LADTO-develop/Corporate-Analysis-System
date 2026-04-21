# Security Policy — Internal Information Channel

BFD는 "내부정보 채널"이라는 선택적 기능을 제공합니다. 기업 담당자가 공시되지 않은
내부 기밀(예: 미공개 실적 전망, 미공개 M&A 정보, 경영진 인터뷰 노트)을 LLM에
전달하여 판정에 반영하게 하는 경로입니다. 이 경로는 **기본 비활성**이며, 활성화
시에는 이 문서의 모든 규약을 준수해야 합니다.

## 위협 모델

| 위협 | 대응 |
|------|------|
| 학습 데이터로 재활용 | 학습에 사용되지 않는다고 명시한 API만 사용. 현재 Anthropic API는 기본적으로 요청을 학습에 사용하지 않음 ([policy](https://www.anthropic.com/legal/commercial-terms)). |
| 세션 간 누설 | 세션마다 별도 thread_id, `MemorySaver` 기본 — 프로세스 종료 시 사라짐. 영구 저장 금지. |
| 동일 LLM이 분석+검증을 모두 수행 | 위원회 LLM은 앙상블에 참여하지 않은 별도 모델로 지정 (`configs/agent/committee.yaml`의 `committee_llm`). |
| PII / 관계자 정보 유출 | `bfd.rag.internal_info.redaction`이 입력 전 RRN/BRN/CRN/전화/계좌/이메일/카드 마스킹. |
| 감사 추적 부재 | 모든 접근은 `bfd.rag.internal_info.policy._emit_audit`가 `BFD_AUDIT_SINK`로 로깅. |

## 모드

환경 변수 `BFD_INTERNAL_INFO_MODE`로 제어합니다.

| 값 | 의미 |
|----|------|
| `disabled` (기본) | 채널 전체 비활성. read/write 호출 시 `AccessDenied`. |
| `read_only` | 이미 ingest된 내부 정보를 조회만 가능. 새 입력 금지. |
| `full` | 세션 내에서 새 내부 정보 입력 허용. |

모드는 런타임에 바뀌지 않습니다 — 프로세스를 재기동해야 반영됩니다.

## 감사로그

`BFD_AUDIT_SINK` 로 세 가지 대상 중 하나를 선택합니다.

- `stdout` (기본) — structlog JSON으로 표준출력에 기록.
- `file:///path/to/audit.jsonl` — JSON Lines로 append.
- `syslog` — (미구현, 확장 지점)

모든 감사 이벤트는 다음 필드를 포함합니다.

```json
{
  "timestamp": "2026-04-21T08:12:09Z",
  "actor": "jdoe",
  "action": "read | write | refuse",
  "fact_class": "financial_forecast | m&a | personnel | ...",
  "corp_code": "005930",
  "mode": "full",
  "allowed": true,
  "reason": "",
  "extras": {}
}
```

거부 이벤트도 반드시 기록되므로, 정책 위반 시도를 사후에 식별할 수 있습니다.

## PII 마스킹

`bfd.rag.internal_info.redaction.redact(text)`는 아래 패턴을 `[라벨]`로 치환하고
치환 카운트를 dict로 돌려줍니다.

| 클래스 | 예시 | 라벨 |
|--------|------|------|
| 주민등록번호 | `900101-1234567` | `[RRN]` |
| 사업자등록번호 | `123-45-67890` | `[BRN]` |
| 법인등록번호 | `110111-0001234` | `[CRN]` |
| 휴대전화 | `010-1234-5678` | `[PHONE]` |
| 계좌번호 | `123-45-678901` | `[ACCOUNT]` |
| 이메일 | `a@b.com` | `[EMAIL]` |
| 카드번호 | `1234-5678-9012-3456` | `[CARD]` |

False positive를 허용하는 보수적 설계입니다 — "혹시 PII가 아닐 수도 있는 숫자"까지
마스킹됩니다. 정확도가 필요하다면 `redaction.py`의 정규식을 강화하세요 (단, 이 문서와
테스트를 함께 업데이트).

## 사용 예

```python
import os
os.environ["BFD_INTERNAL_INFO_MODE"] = "full"
os.environ["BFD_AUDIT_SINK"] = "file:///var/log/bfd/audit.jsonl"

from bfd.rag.internal_info import policy, redaction

policy.require_write_allowed()  # raises if mode != 'full'

raw = "900101-1234567 / 010-1234-5678 에서 내부 전망 공유..."
masked = redaction.redact(raw)

policy.record_access(
    action="write",
    fact_class="financial_forecast",
    corp_code="005930",
    extras={"n_masks": sum(masked.counts.values())},
)

# masked.text 를 LLM에 전달
```

## 변경 관리

이 문서의 규약을 바꾸려면:

1. PR 설명에 구체적인 위협/대응 변경을 기재.
2. `bfd/rag/internal_info/policy.py` 또는 `redaction.py`를 수정.
3. 해당 함수 단위 테스트를 추가/수정 (현재는 smoke 테스트만 존재 — 보강 여지).
4. 최소 한 명의 리뷰어가 보안 관점에서 승인.

**주의**: 본 시스템은 감사·리스크 관리·법무 리뷰를 *대체*하지 않습니다.
내부정보를 AI에 투입하는 행위 자체의 규제·계약상 의무는 별도로 확인해야 합니다.
