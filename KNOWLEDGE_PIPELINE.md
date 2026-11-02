# 출처 기반 말맥: 첫 실행 절차

이 코드는 **수집 도구와 학습 도구**다. 현재 저장소에는 수백 건의 검증된 문화 문서나 검수 완료 QLoRA 학습셋이 들어 있지 않다. 문서 수, 검색 품질, QLoRA 성능은 Seraph에서 실제 수집·평가한 뒤에만 발표한다.

현재 자동 수집기는 위키백과의 **임시 출처**만 다룬다. 최종 서비스의 우선 자료는 [공식 자료 도입 순서](KNOWLEDGE_PLAN.md#공식-자료-도입-순서)에 따라 한국어기초사전, 관련 관광·문화 공공데이터다. 모두의 말뭉치는 검수할 대화 사례를 찾는 데 사용한다. 각 자료의 실제 이용 조건을 확인하기 전에는 학습이나 공개 재배포에 넣지 않는다.

한국어기초사전의 `사전 검색` API에 대해서는 사용자 제공 공식 안내 화면에서 `https://krdict.korean.go.kr/api/search`, `key`·`q`·`start` 요청 변수 및 XML의 `target_code`·`word`·`sup_no`·`example`·`link` 필드를 확인했다. 이를 읽는 [검색 후보 수집기](scripts/collect_krdict.py)가 있다. 이 응답에는 **뜻풀이가 없으므로** 아직 검색 지식 문서로 합치지 않는다. `사전 내용` API 응답과 뜻풀이·용례 이용 조건을 확인한 뒤에만 본문 수집과 RAG 색인을 추가한다.

발급받은 키를 명령줄 인자나 코드에 적지 말고, 할당받은 계산 노드의 셸에서 입력한다. 이미 캡처에 노출한 개인 키라면 먼저 재발급한다.

```bash
cd /data/$USER/repos/swf/Forswf
git pull --ff-only
source /data/$USER/envs/malmaek/bin/activate
read -rsp '한국어기초사전 API 키: ' KRDICT_API_KEY; echo
export KRDICT_API_KEY
python -m scripts.collect_krdict --queries data/krdict_queries.txt \
  --output /data/$USER/datasets/malmaek-krdict-search
unset KRDICT_API_KEY
```

결과 `search_candidates.jsonl`은 표제어·용례·원문 링크를 담은 **미검수 후보**이며 모델 학습이나 앱 검색에는 자동으로 들어가지 않는다. API 키는 파일에 저장하지 않는다.

## 1. 계산 노드에서 공개 자료 수집

로그인 노드에서는 Python, 압축 해제, 대량 데이터 검색을 하지 않는다. `srun`으로 할당받은 계산 노드에서 아래를 실행한다. 저장소는 `codex/malmaek-seraph` 브랜치로 맞춘다. 문서 원본과 KoRMo 모델은 `/data/$USER`에 두고, **검색·학습·평가에 실제로 읽는 데이터는 `/local_datasets/$USER`**에 둔다.

```bash
cd /data/$USER/repos/swf/Forswf
git pull --ff-only
source /data/$USER/envs/malmaek/bin/activate
export HF_HOME=/data/$USER/models/malmaek/hf-cache
export KORMO_REVISION=547fa40eeb8c69d3c5e278daf906d315c4f8e7ea

python -m scripts.collect_wikimedia --language en \
  --titles data/wikipedia_en_titles.txt \
  --categories data/wikipedia_en_categories.txt \
  --output /data/$USER/datasets/malmaek-wiki-en
python -m scripts.collect_wikimedia --language ko \
  --titles data/wikipedia_ko_titles.txt \
  --categories data/wikipedia_ko_categories.txt \
  --output /data/$USER/datasets/malmaek-wiki-ko
python -m scripts.merge_knowledge \
  --inputs /data/$USER/datasets/malmaek-wiki-en/documents.jsonl /data/$USER/datasets/malmaek-wiki-ko/documents.jsonl \
  --output /data/$USER/datasets/malmaek-knowledge
```

수집기는 위키백과 API가 반환하는 **현재 사이트의 라이선스 이름·링크**, 개별 문서 제목·URL·수정 시각·본문을 보존한다. 카테고리는 한 단계의 문서만 가져오며 존재하지 않거나 빈 항목은 0건으로 표시된다. 카테고리 문서에는 주제와 무관한 항목이 섞일 수 있다. 실제 건수와 라이선스·관련성을 검토하기 전에는 모든 문서가 사용 가능하다고 주장하지 않는다. 위키백과의 일반 설명은 개인의 속뜻이나 지역별 생활 규칙을 입증하는 자료가 아니다. 공식 기관 자료를 추가할 때도 동일한 출처 필드를 갖춘 `documents.jsonl`로 합친다.

## 2. 계산 노드의 로컬 디스크에 지식 문서 준비

```bash
local_knowledge=/local_datasets/$USER/malmaek-knowledge-${SLURM_JOB_ID:-manual}
bash scripts/stage_knowledge.sh /data/$USER/datasets/malmaek-knowledge/knowledge.tar.gz "$local_knowledge"
python -m scripts.query_knowledge --knowledge "$local_knowledge/chunks.jsonl" \
  --query 'How should I greet a professor in Korea?'
python -m scripts.query_knowledge --knowledge "$local_knowledge/chunks.jsonl" \
  --query '반찬은 셀프입니다 식당'
```

기본 검색기는 다국어 E5-small을 CPU에서 실행한다. 처음 한 번 모델을 내려받고 임베딩을 계산하므로 시간이 걸린다. 벡터 캐시는 **같은 로컬 데이터 폴더**에 저장된다. KoRMo 가중치·LoRA는 계속 `/data/$USER/models`에 둔다. 진단용 `--retriever bm25`는 추가 모델 없이 작동하지만 영어 질문과 한국어 문서의 교차 언어 검색에는 약하다. 두 테스트 질문에서 가져온 문서 제목과 발췌가 실제로 질문에 맞는지 확인한다.

## 3. 출처가 보이는 시연

```bash
python demo.py --revision "$KORMO_REVISION" --knowledge "$local_knowledge/chunks.jsonl"
```

앱은 가져온 발췌를 KoRMo 입력에 붙이고, 답변 아래에 검색한 문서의 URL과 라이선스를 표시한다. 답변 속 `[1]` 같은 인용과 실제 문서 내용이 일치하는지는 아직 자동 검증하지 않는다. 무관한 문서가 검색되거나 근거 없이 답을 단정하는 실패 사례도 기록한다. 검색 결과가 없으면 출처를 꾸미지 않고 답을 보류한다.

## 4. QLoRA용 검수 자료

```bash
python -m scripts.prepare_review_queue --knowledge "$local_knowledge/chunks.jsonl" \
  --output /data/$USER/datasets/malmaek-review-64.csv
```

이 CSV의 64개 사례는 **모두 미검수 초안**이며 검색 결과는 정답 출처가 아닌 후보일 뿐이다. 검수자는 각 사례의 한국어 뜻·맥락·행동 제안·불확실성을 확인하고, 근거 문단이 실제로 해당 주장과 관련 있는지 읽는다. 관련이 없으면 출처를 빼거나 다른 자료를 찾는다. 영어·쉬운 한국어 답안을 작성하고 답에 `[1]`처럼 근거 번호를 넣는다. 출처가 없는 관습을 단정하거나 학습 자료로 승격하지 않는다.

Python으로 사례를 편집하려면 `prepare_dataset.py`의 `TRAIN`·`EVAL`을 사용한다. 각 레코드에 기존 필드 외에 아래 필드를 추가한다. `evidence=[]`는 근거가 부족한 질문의 보류 사례다. `reviewed=True`는 **실제로 사람이 검수한 뒤에만** 적는다.

```python
{
    "task": "ask",
    "expression": "How should I greet a professor?",
    "situation": "First meeting at university",
    "relationship": "Student and professor",
    "language": "en",
    "answer": "... [1] ...",
    "evidence": [{
        "source_id": "wikipedia:en:...",
        "title": "...",
        "url": "https://...",
        "license": "...",
        "text": "An exact, relevant excerpt from the source",
    }],
    "reviewed": True,
}
```

학습 목표는 200~500개의 **서로 다른 검수된 상황**과 별도 평가 50개 이상이다. 영어·한국어 번역본을 모두 두되 같은 상황의 양쪽 언어는 같은 분할에 둔다. 자료가 부족하면 더 작은 실험을 하고 데이터 규모와 일반화 한계를 정확히 표시한다. 공개 벤치마크는 학습에 섞지 않는다.

```bash
python prepare_dataset.py --output-dir ./prepared-data
# 형식 예시 2건으로는 학습이 시작되지 않는다. 검수한 50개 이상의 고유 표현/질문을 채운 뒤 실행한다.
cp ./prepared-data/malmaek.tar.gz /data/$USER/datasets/malmaek-reviewed.tar.gz
```

긴 학습은 Slurm 배치 노드에서 지식 문서와 학습 자료를 **같은 계산 노드**에 스테이징하도록 두 압축 파일을 전달한다. `MALMAEK_REQUIRE_EVIDENCE=1`이면 모든 학습/평가 사례의 검수 표시·근거 필드와 인용 번호를 확인한다.

```bash
mkdir -p logs
export MALMAEK_REQUIRE_EVIDENCE=1
sbatch -p <허용된_batch_partition> scripts/train.slurm.sh \
  /data/$USER/datasets/malmaek-reviewed.tar.gz \
  /data/$USER/datasets/malmaek-knowledge/knowledge.tar.gz
```

`outputs/comparison-<job-id>.jsonl`에는 KoRMo 단독, KoRMo+검색, QLoRA 단독, QLoRA+검색 결과가 들어간다. 특히 **KoRMo+검색과 QLoRA+검색**을 동일한 검색 결과로 비교한다. 평가는 뜻 정확성, 맥락 적합성, 실제 행동의 유용성, 근거 일치·근거 없는 단정 회피를 사람이 블라인드 채점한다. 학습 실행과 성능 개선은 아직 서버에서 검증되지 않았다.

```bash
python -m scripts.make_blind_eval \
  --comparison outputs/comparison-<job-id>.jsonl \
  --output outputs/scoring-<job-id>.csv \
  --key outputs/condition-key-<job-id>.csv
```

채점자에게는 `scoring` CSV만 주고 `condition-key`는 채점이 끝난 뒤 연다. 각 항목 0~2점을 쓰고, 검색 출처의 원문과 모델 인용이 맞는지 확인한다.
