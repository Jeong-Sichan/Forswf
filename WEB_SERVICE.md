# 말맥 FastAPI 웹 서비스

이 브랜치는 기존 `demo.py`와 별개로 실행하는 웹 챗봇이다. 첫 화면에서 전체 질문·한국 문화·예절과 표현·음식과 식사·교통과 이동·유학생 생활을 고를 수 있고, 갓을 쓴 말맥 캐릭터가 주제에 맞춰 먼저 인사한다. 최근 대화 최대 4개 발화를 문맥으로 전달하며, 카테고리별 대화는 **열려 있는 브라우저 탭 안에서만** 유지한다. 출처는 답변 아래에 작게 표시한다. 로그인·영구 저장·공개 인터넷 배포는 아직 구현하지 않았다.

영어 답변에는 **기본 KoRMo**를 사용한다. 기존 AI Hub 대화 LoRA는 영어 설명 품질이 떨어지는 사례가 있었으므로, 어댑터가 설정돼 있어도 영어 생성 시 비활성화한다. 한국어 답변에는 설정한 LoRA를 사용한다. 이는 한국어 생활 답변의 정확도가 개선됐다는 뜻은 아니며, 공식 안내 질문은 답변을 출처와 대조해야 한다. 어댑터가 지정되지 않으면 두 언어 모두 기본 KoRMo다.

생활·교통 질문에는 법제처 RAG, 전통 문양·유물 질문에는 전통문화 상징사전 RAG, 음식·예절·일반 문화 질문에는 선택적으로 위키백과 RAG를 사용한다. 해당 자료가 없으면 생활·교통 질문은 근거가 없다고 안내한다. 다른 주제는 일반 모델 답변이 가능하지만 공식 출처로 검증된 정보로 간주하지 않는다. 최신 비자·주거·학교 규정은 링크된 담당기관 안내를 재확인한다.

## Seraph 계산 노드에서 실행

코드는 `codex/malmaek-web` 브랜치에 있다. 기존 `codex/malmaek-seraph` 작업 내용은 덮어쓰지 않는다. 현재 브랜치에 수정 사항이 있다면 먼저 보존한 뒤 전환한다.

```bash
cd /data/$USER/repos/swf/Forswf
git fetch origin codex/malmaek-web
git switch --track -c codex/malmaek-web origin/codex/malmaek-web
source /data/$USER/envs/malmaek/bin/activate
python -m pip install -r requirements-web.txt
```

이미 로컬 브랜치가 있으면 `git switch codex/malmaek-web` 후 `git pull --ff-only`를 사용한다. Python 실행·자료 스테이징·패키지 설치는 Slurm으로 할당받은 **GPU 계산 노드**에서 한다.

법제처 확장 압축 파일은 이미 `/data/$USER/datasets/malmaek-easylaw-expanded/knowledge.tar.gz`에 있다. 같은 계산 노드에서 앞서 만든 `local_law`가 남아 있다면 아래 스테이징을 생략하고 그 경로를 사용한다. 새 Slurm 작업이라면 고유한 경로로 다시 스테이징한다.

```bash
local_law=/local_datasets/$USER/malmaek-easylaw-web-${SLURM_JOB_ID:-manual}
bash scripts/stage_knowledge.sh \
  /data/$USER/datasets/malmaek-easylaw-expanded/knowledge.tar.gz "$local_law"
```

기존 한국어·영어 위키백과 통합 자료가 있으면 함께 스테이징한다. 없어도 서비스는 실행되지만, 음식·예절·일반 문화 답변에 연결할 RAG 문서가 줄어든다.

```bash
local_wiki=/local_datasets/$USER/malmaek-wiki-web-${SLURM_JOB_ID:-manual}
bash scripts/stage_knowledge.sh \
  /data/$USER/datasets/malmaek-knowledge/knowledge.tar.gz "$local_wiki"
```

전통문화 상징사전 자료를 서버에 옮긴 경우에는 [별도 수집 절차](KNOWLEDGE_PIPELINE.md#전통문화-상징사전-rag)로 `chunks.jsonl`을 만든다. 이 색인을 설정하지 않으면 전통 문양 전용 RAG는 비활성화된다.

```bash
export HF_HOME=/data/$USER/models/malmaek/hf-cache
export KORMO_REVISION=547fa40eeb8c69d3c5e278daf906d315c4f8e7ea
export MALMAEK_ADAPTER=/data/$USER/models/malmaek/dialogue-adapter-118549
export MALMAEK_LAW_CHUNKS="$local_law/chunks.jsonl"
export MALMAEK_WIKI_CHUNKS="$local_wiki/chunks.jsonl"
# 전통문화 상징사전 색인을 만든 경우에만 설정
# export MALMAEK_CULTURE_CHUNKS="$local_culture/chunks.jsonl"
export MALMAEK_RETRIEVER=e5
mkdir -p /local_datasets/$USER/malmaek-web-cache
export MALMAEK_E5_CACHE=/local_datasets/$USER/malmaek-web-cache/combined-${SLURM_JOB_ID:-manual}.pt
python -m webapp --host 127.0.0.1 --port 8000
```

계산 노드에서 시작할 때 모델 5개 샤드와 검색 모델을 불러오므로 첫 화면이 나오기까지 시간이 걸릴 수 있다. `http://127.0.0.1:8000/api/health`에 `ready: true`가 표시되면 준비됐다. 개인 PC에서는 계산 노드에 연결한 VS Code의 **Ports** 기능 등 승인된 SSH 터널로 포트 8000을 전달해 `http://localhost:8000`을 연다. 기본 서버는 계산 노드의 `127.0.0.1`에만 바인딩한다. 인증 없는 공개 배포에는 적합하지 않다.

첫 확인 질문은 영어로 `What should I check before signing a housing lease in Korea?`와 한국어로 `버스 교통카드 없이 타도 되나요?`를 권장한다. 법제처 출처가 질문과 맞는지, 영어 답변에 한국어 설명 문장이 섞이지 않는지, 한국어 답변에 어댑터 표시가 뜨는지 확인한다. `MALMAEK_RETRIEVER=bm25`로 바꾸면 검색 모델 로딩 없이 빠르게 화면을 시험할 수 있지만 영어 질문에서 한국어 문서 검색 범위가 좁아질 수 있다.

## 개발 확인

GPU 없이도 첫 화면과 API 구조를 시험할 수 있다. `KORMO_REVISION`을 설정하지 않으면 `/api/health`는 `ready: false`를 반환하며 채팅은 503을 반환한다. 가짜 모델 답변을 실제 서비스 답변처럼 표시하지 않는다.

```bash
python -m pip install -r requirements-web.txt
python -m unittest discover -s tests -q
node --check webapp/static/app.js
python -m webapp --port 8000
```

API는 `POST /api/chat`에 `{ "message": "...", "category": "life", "language": "auto", "history": [] }`를 받는다. 답변에는 본문, 실제 선택된 출처, 출력 언어, 어댑터 사용 여부가 포함된다. 브라우저에 제공되는 모델 출력은 HTML로 해석하지 않고 일반 텍스트로 렌더링한다.
