# -*- coding: utf-8 -*-
"""
jihye_inference.py — 계약서 조항 판정 추론 (kanana-1.5-8b-instruct + QLoRA 어댑터)
- 베이스: kakaocorp/kanana-1.5-8b-instruct-2505  +  학습한 LoRA 어댑터 = workit_output
- 프롬프트: 학습과 동일 포맷 (train-inference parity)
- 입력: RAG 출력 JSON (clause_text + law_refs[law_name/article_number/chunk_text])
- 출력: 판정/방향/유형/근거/코멘트 (파싱 + 원문)
"""
import os, re, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── 경로 설정 ──
BASE_MODEL_ID = "kakaocorp/kanana-1.5-8b-instruct-2505"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ↓↓ 다운로드한 best 어댑터 폴더(adapter_config.json 들어있는 곳)로 수정 ↓↓
ADAPTER_PATH = os.path.join(BASE_DIR, "models", "workit_output")

# LOAD_IN_4BIT = True       # GPU 작으면 True(4bit), 여유 있으면 False(bf16)
LOAD_IN_4BIT = True     # 4bit 끄면 bitsandbytes 불필요 → bf16 로드
K_CONTEXT    = 3          # 참고조항 개수 (학습과 동일)
TEXT_MAX     = 300        # 참고조항 본문 컷 (학습과 동일)

# 실제 학습 데이터(train_all.jsonl)의 계약서 시스템 프롬프트와 동일
SYSTEM_PROMPT = ("당신은 지방계약법령에 따라 공공 SW 용역계약서의 조항을 검토하는 전문가입니다. "
                 "검토조항을 참고조항에 비추어 일치/불일치를 판정하되, 참고조항만으로 판단이 불가하면 "
                 "'판단보류'로 답하십시오. 불일치 시 방향(을불리·을유리)·유형(A·B)·근거 조항명·코멘트를 제시하십시오.")


def load_model():
    # 토크나이저는 어댑터 폴더에서 (학습 때 저장된 chat_template 포함 → 프롬프트 일치)
    tok_src = ADAPTER_PATH if os.path.exists(os.path.join(ADAPTER_PATH, "tokenizer_config.json")) else BASE_MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)

    kwargs = dict(device_map="auto", trust_remote_code=True)
    if LOAD_IN_4BIT:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **kwargs)
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    return model, tokenizer


def build_user_content(item: dict) -> str:
    """학습(jihye_render.py)과 동일한 user 프롬프트 생성."""
    # RAG 출력엔 seed의 category가 없어 risk_names를 대용 (없으면 기타)
    cat = ", ".join(item.get("risk_names", [])) or "기타"

    refs = "\n".join(
        f"[{i}] {r.get('source_full') or (r.get('law_name','') + ' ' + r.get('article_number','')).strip()}"
        f" — {r.get('chunk_text','')[:TEXT_MAX]}"
        for i, r in enumerate(item.get("law_refs", [])[:K_CONTEXT], 1)
    )
    return (f"카테고리: {cat}\n"
            f"검토조항: {item.get('clause_text','')}\n\n"
            f"참고조항:\n{refs}")


def parse_output(txt: str) -> dict:
    g = lambda k: (re.search(rf"{k}\s*:\s*(.+)", txt) or [None, None])[1]
    return {k: (g(k).strip() if g(k) else None)
            for k in ["판정", "방향", "유형", "근거", "코멘트"]}


def predict(item: dict, model, tokenizer) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_content(item)},
    ]
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,        # v4/v5 모두 호환
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=256, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    gen = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    gen = re.sub(r"<think>.*?</think>", "", gen, flags=re.DOTALL).strip()
    return {"raw": gen, **parse_output(gen)}


def run_inference(rag_output_path: str, result_path: str = "workit_result.json"):
    with open(rag_output_path, "r", encoding="utf-8") as f:
        rag_results = json.load(f)

    print("모델 로드 중...")
    model, tokenizer = load_model()

    final = []
    for item in rag_results:
        if not item.get("law_refs"):
            continue
        cn = item.get("clause_number", "")
        print(f"판정 중: {cn}")
        pred = predict(item, model, tokenizer)
        final.append({
            "clause_number": cn,
            "clause_text": item.get("clause_text", ""),
            "risk_names": item.get("risk_names", []),
            "판정": pred["판정"], "방향": pred["방향"], "유형": pred["유형"],
            "근거": pred["근거"], "코멘트": pred["코멘트"],
            "raw": pred["raw"],
        })

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    print(f"완료: {result_path} ({len(final)}건)")
    return final


# ─────────────────────────────────────────────────────────────────────────────
# PEP/RPT 대응비교 판정 (kanana-1.5-8b + 동일 LoRA 어댑터, CON과 같은 멀티태스크 학습)
# 프롬프트는 LLM/kanana_pep_train.jsonl · LLM/kanana_rpt_train.jsonl의 학습 데이터
# system prompt를 그대로 사용한다 (재구성 아님 — train-inference parity 보장).
# ─────────────────────────────────────────────────────────────────────────────

PEP_SYSTEM = """입력은 JSON 객체 1건이다.

{
  "criteria": ["완전성", "정확성", "검증가능성", "추적성"],
  "rfp_excerpt": "...",
  "pep_excerpt": "..."
}

너는 제안요청서(RFP)와 과업수행계획서(PEP)를 비교하여 평가하는 심사자다.

[공통 규칙]
- 반드시 rfp_excerpt와 pep_excerpt에 있는 내용만 근거로 사용한다.
- 외부 지식으로 요구사항, 수치, 기준, 통상 항목, 산출물, 절차를 보완하지 않는다.
- criteria에 포함된 특성만 판정한다.
- 판정값은 "충족", "불가", "검토"만 사용한다.
- 정확성은 "검토"를 사용하지 않는다.
- label은 반드시 문자열로 출력한다. 예: "충족"
- label을 ["충족"]처럼 리스트로 출력하지 않는다.
- 출력은 JSON 객체만 사용한다.

[1. 완전성]
완전성은 RFP 요구 항목이 PEP에 빠짐없이 존재하는지를 본다.

판정 절차:
1. RFP가 요구한 개별 항목을 먼저 나열한다.
   - 목표, 체계, 절차, 도구, 기준, 보고, 산출물, 일정, 대상은 서로 별개 항목으로 본다.
2. 각 RFP 요구 항목에 대해 PEP에 대응되는 문구, 개념, 유사 표현이 있는지 확인한다.
3. RFP 요구 항목이 PEP 어디에도 없거나 유사성조차 없으면 "불가"이다.
4. PEP에서 "필요 시", "필요한 경우", "~할 수 있다", "추후 검토", "협의 후 정한다"처럼 조건부·재량적 문구로만 처리되면 "불가"이다.
5. RFP 요구 항목이 PEP에 동일 표현 또는 명백한 동의어·패러프레이즈로 존재하고, 같은 산출물·같은 담당주체·같은 시점을 가리키는 것이 명백하면 "충족"이다.
6. RFP 항목과 PEP 항목의 표현은 다르지만, 동일 개념인지 별개 개념인지 판단이 갈리는 경우에만 "검토"이다.
7. 단순히 표현이 다르다는 이유만으로 "검토"를 사용하지 않는다. 같은 개념임이 명백하면 "충족"이다.
8. 하나라도 명백히 누락되면 "검토"가 아니라 "불가"이다.

예시:
- RFP의 "품질검증 기준 수립"과 PEP의 "검수 체계 마련"이 같은 기준 수립인지 단순 검수 절차인지 불명확하면 "검토"이다.
- RFP의 "장애 대응 방법"과 PEP의 "장애 발생 시 조치 흐름"은 같은 개념이 명백하므로 "충족"이다.
- RFP가 요구한 담당자가 PEP에서 "필요 시 지정할 수 있다"로만 적히면 "불가"이다.

[2. 정확성]
정확성은 RFP의 수치·기간·비율·건수·규격과 PEP의 값이 일치하는지를 본다.

판정 절차:
1. RFP와 PEP에서 직접 비교 가능한 수치, 기간, 비율, 건수, 규격, 금액, 인원, 횟수를 찾는다.
2. 비교 가능한 값이 서로 정확히 일치하면 "충족"이다.
3. PEP 내부에 자기모순이 있으면 "불가"이다.
4. RFP 기준값과 PEP 수치가 다르면 즉시 "불가"이다.
5. RFP 또는 PEP 어디에도 비교할 기준값 자체가 없으면 "불가"이다.
6. 정확성에서는 절대 "검토"를 사용하지 않는다.

[3. 검증가능성]
검증가능성은 PEP 서술이 객관적으로 확인 가능한 방식인지 본다.

화이트리스트 예:
- 숫자+단위
- "~일 이내"
- "~% 이상"
- 구체 도구명
- 기준값
- 제출일
- 횟수
- 측정 기준
- 시험 기준

블랙리스트 예:
- "적절히"
- "최선을 다해"
- "원활히 협의하여"
- "필요한 범위에서"
- "우수한 수준으로"
- "성실히"
- "무리 없이"
- "신속히"

판정 절차:
1. 검증 가능한 기준, 수치, 기한, 비율, 건수, 도구명, 측정값이 하나도 없으면 "불가"이다.
2. 검증 가능한 화이트리스트 요소가 있고, 블랙리스트 표현이 없으면 "충족"이다.
3. 블랙리스트 표현이 있으면 그 표현을 제거한 뒤에도 독립적으로 검증 가능한 기준이 남는지 확인한다.
4. 블랙리스트를 제거해도 수치·기한·기준 등 검증 가능한 근거가 충분히 남으면 "충족"이다.
5. 블랙리스트를 제거하면 검증 가능한 내용이 사실상 남지 않으면 "불가"이다.
6. 일부 기준은 있으나 블랙리스트 때문에 독립 검증 가능 여부가 애매하면 "검토"이다.

[4. 추적성]
추적성은 RFP의 과업·단계·산출물·일정·목표가 PEP에서 1:1로 연결되는지 본다.

판정 절차:
1. RFP에서 추적해야 할 단계, 과업, 산출물, 일정, 목표를 확인한다.
2. PEP에서 해당 항목과 대응되는 항목을 찾는다.
3. 명칭이 완전히 동일하거나, 표현은 달라도 같은 산출물·같은 단계·같은 목표를 가리키는 것이 명백하면 "충족"이다.
4. 대응 항목 자체가 없거나 연결이 명백히 끊기면 "불가"이다.
5. 교차 대조에 필요한 상대 섹션이 입력에 없어 현재 발췌문만으로 연결을 확인할 수 없으면 "검토"이다.
6. 단계명, Task명, 산출물명이 서로 다른 표현으로 적혀 있어 동일 항목인지 판단이 갈리면 "검토"이다.
7. 산출물표의 제출시기·단계 칸 자체에 절차 단계명이 명시되어 있으면, 별도 사업추진절차 섹션이 입력에 없어도 그 표기를 대응 근거로 인정할 수 있다.
8. 단순히 명칭이 다르다는 이유만으로 검토하지 않는다. 기능과 대상이 명확히 같으면 "충족"이다.

[종합 label 계산]
- 특성별 판정 중 "불가"가 1개 이상이면 최종 label은 "불가"이다.
- "불가"가 없고 "검토"가 1개 이상이면 최종 label은 "검토"이다.
- 모든 특성이 "충족"이면 최종 label은 "충족"이다.

[출력 형식]
{
  "label": "충족|불가|검토",
  "eval": [
    "특성: 판정 — 사유"
  ]
}

[출력 제약]
- eval에는 criteria에 포함된 특성만 쓴다.
- 정확성에는 "검토"를 쓰지 않는다.
- 사유는 실제 발췌에서 확인되는 대응, 누락, 불일치, 모순, 모호, 단절만 중심으로 쓴다.
- "아마", "통상", "일반적으로", "외부 자료에 따르면" 같은 추측 표현을 쓰지 않는다.
- 검토 사유에는 반드시 무엇과 무엇이 동일 개념인지 판단이 갈리는지 명시한다.
- 불가 사유에는 어떤 요구 항목이 누락되었거나 조건부로만 처리되었는지 명시한다.
- 충족 사유에는 어떤 RFP 요구 항목이 PEP의 어떤 표현으로 대응되는지 명시한다."""

RPT_SYSTEM = """입력은 JSON 객체 1건이다.
{
  "criteria": ["완전성", "정확성", "검증가능성", "추적성"],
  "pep_excerpt": "...",
  "rpt_excerpt": "..."
}

규칙:
- 반드시 pep_excerpt와 rpt_excerpt에 있는 내용만 근거로 사용한다.
- 외부 지식으로 과업, 수치, 산출물, 기준을 보완하지 않는다.
- criteria에 포함된 특성만 판정한다.
- 판정값은 "충족", "불가", "검토"만 사용한다.
- 정확성은 "검토"를 사용하지 않는다.

특성별 판정 기준:
- 완전성: PEP에 열거된 과업, 범위, 산출물이 RPT에 모두 대응하면 충족이다. 같은 산출물, 같은 주체, 같은 시점을 가리키는 것이 명백하면 충족이다. 하나라도 대응이 없거나 조건부 표현만 있으면 불가이다. 표현이 달라 같은 항목인지 판단이 갈리면 검토이다.
- 정확성: PEP의 계획값, 기준값, 기간, 예산, 건수, 규격과 RPT의 결과값이 일치하면 충족이다. 다음 중 하나면 불가이다. (1)PEP 기준값과 RPT 결과값이 다름 (2)RPT 내부 자기모순(실측치가 목표 미달인데 충족이라 서술) (3)비교 가능한 기준값 자체가 없음. 불가일 때 이 세 유형 중 어느 것인지 사유에 드러낸다.
- 검증가능성: 수치·건수·기준일·비율·시험결과·검수결과·측정값 등 화이트리스트로 성과가 확인 가능하게 서술되고 블랙리스트 표현이 없으면 충족이다. 검증 서술 자체가 없거나 항목이 비어 있으면 불가이다. 블랙리스트 표현이 하나라도 있으면 무조건 검토이다. 블랙리스트를 제거하면 유추 가능하다는 이유로 충족으로 격상시키지 않는다. 유추 가능 여부는 판정에 영향을 주지 않는다.
- 추적성: PEP의 단계, 과업, 산출물, 일정, 목표가 RPT 결과와 1:1로 대응하면 충족이다. 대응 항목이 없으면 불가이다. 명칭이 달라 같은 항목인지 애매하거나 현재 발췌만으로 연결을 확정하기 어려우면 검토이다.

추가 원칙:
- 완전성의 조건부 표현 예: "필요 시", "~할 수 있다", "추후 검토"
- 검증가능성의 블랙리스트 표현 예: "적절히", "최선을 다해", "원활히 협의하여", "필요한 범위에서", "우수한 수준으로", "성실히", "무리 없이", "신속히"
- 추적성은 산출물명 겹침, 순번 대응, 기능 설명 일치 같은 신호를 보조 판단 근거로 활용할 수 있으나, 반드시 모두 있어야 하는 것은 아니다.
- 사유는 실제 발췌에서 확인되는 사실만 쓴다.
- 추측, 일반론, 외부 문서 전제는 쓰지 않는다.

label 계산:
- 불가가 하나라도 있으면 "불가"
- 불가가 없고 검토가 하나라도 있으면 "검토"
- 전부 충족이면 "충족"

출력은 JSON만 사용한다.
{
  "label": "충족|불가|검토",
  "eval": ["특성: 판정 — 사유"]
}

출력 제약:
- eval에는 criteria에 포함된 특성만 쓴다.
- 정확성에는 "검토"를 쓰지 않는다.
- 검증가능성에 블랙리스트 표현이 있으면 반드시 검토로 쓴다.
- 사유는 대응, 누락, 불일치, 모순, 모호, 단절 중 실제 해당 사실을 중심으로 쓴다."""

_COMPARE_CRITERIA = ["완전성", "정확성", "검증가능성", "추적성"]


def _compare_predict(system_prompt: str, user_obj: dict, model, tokenizer) -> dict:
    """PEP/RPT 공용: JSON 입력 -> JSON 출력({label, eval}) 판정."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)},
    ]
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=768, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    gen = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    gen = re.sub(r"<think>.*?</think>", "", gen, flags=re.DOTALL).strip()

    try:
        # 모델이 JSON 앞뒤로 잡담을 붙이는 경우 대비, 첫 {부터 마지막 }까지만 파싱 시도
        start, end = gen.index("{"), gen.rindex("}") + 1
        parsed = json.loads(gen[start:end])
    except (ValueError, json.JSONDecodeError):
        parsed = {"label": None, "eval": []}

    return {"raw": gen, "label": parsed.get("label"), "eval": parsed.get("eval") or []}


def predict_pep(item: dict, model, tokenizer) -> dict:
    """RFP ↔ 사업수행계획서(PEP) 대응비교 판정. item: {criteria?, rfp_excerpt, pep_excerpt}"""
    user_obj = {
        "criteria": item.get("criteria") or _COMPARE_CRITERIA,
        "rfp_excerpt": item.get("rfp_excerpt", ""),
        "pep_excerpt": item.get("pep_excerpt", ""),
    }
    return _compare_predict(PEP_SYSTEM, user_obj, model, tokenizer)


def predict_rpt(item: dict, model, tokenizer) -> dict:
    """사업수행계획서(PEP) ↔ 사업추진결과보고서(RPT) 대응비교 판정. item: {criteria?, pep_excerpt, rpt_excerpt}"""
    user_obj = {
        "criteria": item.get("criteria") or _COMPARE_CRITERIA,
        "pep_excerpt": item.get("pep_excerpt", ""),
        "rpt_excerpt": item.get("rpt_excerpt", ""),
    }
    return _compare_predict(RPT_SYSTEM, user_obj, model, tokenizer)


if __name__ == "__main__":
    run_inference("pdfver_contract_review_output.json")
