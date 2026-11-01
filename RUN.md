# Seraph에서 말맥 실행

이 안내는 제공받은 Seraph 튜토리얼 3~6의 저장소·Slurm 규칙에 맞춘 것입니다. `aurora-g5`에서 RTX 3090 24GB 한 장과 PyTorch `2.5.1+cu121`의 CUDA 사용을 확인했습니다. GPU가 보이는 것과 Slurm 작업 할당은 별개이므로, 실제 작업은 할당된 계산 노드에서만 실행하세요.

## 저장 위치

- 코드·가상환경·압축된 원본 데이터: `/data/$USER/...` (NAS)
- 학습·평가 데이터만: `/local_datasets/$USER/...` (할당받은 계산 노드의 로컬 디스크)
- KoRMo 모델 캐시와 LoRA 어댑터: `/data/$USER/models/malmaek/...` (사용자 NAS 영역)
- 학습과 평가 스크립트는 `/data` 경로를 데이터로 받으면 오류를 내도록 설정했습니다.
- 로그인 노드에서는 Python 실행, 패키지 설치, 압축 해제, 대량 파일 검색을 하지 마세요.

## 코드를 서버로 옮기기

현재 사용자는 `/data/jsc66519075/repos/swf/Forswf`에 저장소를 복제했습니다. 이 경로에서 `codex/malmaek-seraph` 브랜치로 전환했다면 아래 복제·전송 대안은 건너뛰세요. 이후 명령은 이 실제 체크아웃 경로를 기준으로 합니다.

학교 서버에서 직접 복제할 수 있습니다. 오래된 VS Code `askpass.sh`가 Git 인증을 가로막는 경우, GitHub CLI가 설치돼 있다면 서버 셸에서 다음처럼 한 번 인증하세요. `gh auth login --web`이 표시하는 기기 인증 페이지를 **개인 PC 브라우저**에서 열어 승인합니다. 비밀번호나 토큰을 채팅에 보내지 마세요.

```bash
command -v gh
unset GIT_ASKPASS SSH_ASKPASS
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
cd /data/$USER/repos
git clone --branch codex/malmaek-seraph --single-branch https://github.com/Jeong-Sichan/Forswf.git malmaek-prototype
```

`command -v gh`가 실패하면 GitHub CLI 설치 또는 SSH 키 인증이 필요합니다. 별도 설치 없이 진행하려면 **개인 PC에서 로그인한 GitHub**의 `Jeong-Sichan/Forswf` → `codex/malmaek-seraph` 브랜치를 열고 `Code` → `Download ZIP`을 누르세요. PC에서 ZIP을 풀고 생성된 폴더 이름을 `malmaek-prototype`으로 바꾼 뒤, **개인 PC 터미널에서** 다음을 실행합니다.

```bash
scp -P 30080 -r malmaek-prototype jsc66519075@aurora.khu.ac.kr:/data/jsc66519075/repos/
```

전송 전에 서버의 `/data/$USER/repos/malmaek-prototype`이 이미 있는지 확인하세요. 존재한다면 기존 파일을 덮어쓰지 말고 별도 이름을 사용하세요. 이 단계는 코드 파일 몇 개를 NAS에 놓는 작업이며, 학습 데이터는 아래 절차대로 압축 파일을 계산 노드의 `/local_datasets`로 옮깁니다.

## 1. 데이터 준비

파이썬으로 사례를 작성하려면 `prepare_dataset.py`의 `TRAIN`과 `EVAL` 목록을 편집하세요. 개인 PC 또는 할당받은 계산 노드에서 아래 명령을 실행하면 학습용 JSONL 두 파일과 전송용 압축 파일이 생성됩니다. JSONL은 학습 프로그램의 데이터 저장 형식일 뿐, 사람이 직접 작성해야 하는 형식이 아닙니다. 현재 예시 2개씩은 형식 확인용입니다. 직접 검토한 학습 사례를 최소 50개 이상, 표현이 겹치지 않는 평가 사례를 별도로 준비하세요.

```bash
python prepare_dataset.py --output-dir ./prepared-data
```

생성된 `prepared-data/malmaek.tar.gz` **파일 하나**를 `/data/$USER/datasets/malmaek.tar.gz`로 업로드하세요. NAS의 압축 파일을 계산 노드에서 `/local_datasets`로 복사·압축 해제하는 단계는 `scripts/stage_dataset.sh`가 담당합니다. 데이터 로더에 NAS 경로를 전달하지 마세요.

## 2. Slurm 할당과 패키지 확인

로그인 노드에서 `show-assoc`, `slurm-gres-viz -i`로 본인에게 허용된 partition과 노드를 확인하세요. 튜토리얼의 `debug_ugrad`/`batch_ugrad`는 예시이며 계정별 권한이 다를 수 있습니다. 간단한 확인에는 `srun`으로 계산 노드를 받습니다.

```bash
srun --gres=gpu:1 --cpus-per-gpu=8 --mem-per-gpu=32G -p <허용된_debug_partition> --pty bash
```

**계산 노드 안에서** `nvidia-smi`와 `python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'`를 확인하세요. 기존 `mlclass` 환경을 보존하려면 그 환경을 활성화한 상태에서 별도 가상환경을 사용합니다. 학습 데이터만 `/local_datasets`로 옮깁니다.

```bash
cd /data/$USER/repos/swf/Forswf
python -m venv --system-site-packages /data/$USER/envs/malmaek
source /data/$USER/envs/malmaek/bin/activate
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
python -m pip install -r requirements-gpu.txt
```

가상환경에서 CUDA가 `False`이거나 서버의 PyTorch를 상속하지 못하면 설치를 멈추고 환경 경로를 확인하세요. KoRMo 모델 카드의 라이선스, 접근 조건, 사용 방법도 다운로드 전에 확인해야 합니다. 현재 클라우드에서는 모델 카드와 학교 서버에 직접 접근할 수 없어 모델 로딩은 아직 검증되지 않았습니다.

KoRMo 설정에는 모델 전용 코드(`auto_map`)가 있습니다. 모델 저장소 코드를 실행하기 전 해당 파일을 검토하고, 모델의 40자리 커밋 SHA를 확인해 `KORMO_REVISION`으로 고정하세요. 학습 배치 작업은 이 변수를 상속합니다.

## 3. 짧은 검증과 학습

`srun` 계산 노드에서 모델을 처음 불러올 때 모델 캐시는 사용자 `/data`에, 학습 데이터는 `/local_datasets`에 둡니다. 아래 예시의 `malmaek-debug-1`은 새 실행마다 다른 이름으로 바꾸세요.

```bash
cd /data/$USER/repos/swf/Forswf
export HF_HOME=/data/$USER/models/malmaek/hf-cache
export KORMO_REVISION=여기에_모델_40자리_커밋_SHA
bash scripts/stage_dataset.sh /data/$USER/datasets/malmaek.tar.gz /local_datasets/$USER/malmaek-debug-1/data
python validate_data.py --train /local_datasets/$USER/malmaek-debug-1/data/train.jsonl --eval /local_datasets/$USER/malmaek-debug-1/data/eval.jsonl
```

시연 코드는 체크아웃 디렉터리에서 `python demo.py --revision "$KORMO_REVISION"`로 실행합니다. 기본 바인딩은 계산 노드의 `127.0.0.1`이며, 자신의 SSH 터널이나 승인된 내부 경로로 접속해야 합니다. `--share`는 공개 링크가 필요할 때만 사용하세요.

긴 학습은 로그인 노드에서 `sbatch`로 제출합니다. 제출 전에 학습에 사용할 가상환경을 활성화하고, 작업 디렉터리에서 `logs`를 만드세요. 스크립트는 학습 데이터 압축 파일만 `/local_datasets`에 복사·해제합니다. 모델 캐시와 완성된 LoRA 어댑터는 사용자 `/data` 영역에 둡니다.

```bash
cd /data/$USER/repos/swf/Forswf
source /data/$USER/envs/malmaek/bin/activate
export KORMO_REVISION=여기에_모델_40자리_커밋_SHA
mkdir -p logs
sbatch -p <허용된_batch_partition> scripts/train.slurm.sh /data/$USER/datasets/malmaek.tar.gz
squeue -u "$USER"
```

`outputs/comparison-<job-id>.jsonl`은 자동 채점 결과가 아닙니다. [기획 범위](PROJECT.md)의 네 항목으로 사람이 원본과 조정 모델을 평가하고, 실패 사례도 기록하세요. 학습·평가 사례를 준비하지 않은 현재 ZIP만으로는 성능 개선을 주장할 수 없습니다.
