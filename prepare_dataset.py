"""Edit the Python examples below, then export files for the Seraph training job.

Run on your own computer or an allocated compute node, never the Seraph login node.
"""

import argparse
import json
import tarfile
from pathlib import Path


# Four scenarios in two languages demonstrate the format. Add reviewed cases.
TRAIN = [
    {
        "expression": "반찬은 셀프입니다",
        "language": "ko",
        "situation": "식당 벽에 붙은 안내문",
        "relationship": "손님과 식당",
        "answer": "1. 쉬운 뜻\n반찬은 손님이 직접 가져오라는 뜻입니다.\n2. 상황에 따른 해석\n식당 안의 별도 공간에서 반찬을 가져가는 방식일 수 있습니다.\n3. 적절한 말이나 행동\n반찬을 가져오는 곳을 찾거나 직원에게 물어보세요.\n4. 달라질 수 있는 점\n어떤 반찬까지 셀프인지는 식당마다 다릅니다.",
    },
    {
        "expression": "반찬은 셀프입니다",
        "situation": "A sign on the wall of a restaurant",
        "relationship": "Customer and restaurant",
        "language": "en",
        "answer": "1. Plain meaning\n'반찬은 셀프입니다' means you take the side dishes yourself.\n2. Meaning in this situation\nThe restaurant may have a separate area where customers can pick them up.\n3. What to say or do\nLook for the side-dish station or ask a staff member where it is.\n4. What could change the meaning\nWhich side dishes are self-service depends on the restaurant.",
    },
    {
        "expression": "시간 되실 때 확인 부탁드립니다",
        "language": "ko",
        "situation": "학생이 교수에게 과제 초안 검토를 부탁하는 메일",
        "relationship": "학생과 교수",
        "answer": "1. 쉬운 뜻\n가능한 때에 초안을 봐 달라는 정중한 요청입니다.\n2. 상황에 따른 해석\n상대 일정에 맞추겠다는 표현입니다.\n3. 적절한 말이나 행동\n검토가 필요한 날짜가 있다면 정중하게 덧붙이세요.\n4. 달라질 수 있는 점\n마감이 없다면 날짜를 꼭 넣을 필요는 없습니다.",
    },
    {
        "expression": "시간 되실 때 확인 부탁드립니다",
        "situation": "A student asks a professor to review a draft by email",
        "relationship": "Student and professor",
        "language": "en",
        "answer": "1. Plain meaning\nThis politely asks the professor to review the draft when they have time.\n2. Meaning in this situation\nThe student is trying to respect the professor's schedule. It does not set a clear deadline.\n3. What to say or do\nIf feedback is needed by a certain date, add that date politely.\n4. What could change the meaning\nIf there is no deadline, a date may not be needed.",
    },
]

EVAL = [
    {
        "expression": "생각해 볼게요",
        "language": "ko",
        "situation": "친구에게 주말 모임을 제안한 뒤 들은 답",
        "relationship": "친구",
        "answer": "1. 쉬운 뜻\n제안을 바로 결정하지 않고 생각하겠다는 뜻입니다.\n2. 상황에 따른 해석\n일정을 확인해야 할 수도 있고 답을 미루는 것일 수도 있습니다. 거절이라고 단정할 수 없습니다.\n3. 적절한 말이나 행동\n편할 때 알려 달라고 하고 시간을 주세요.\n4. 달라질 수 있는 점\n앞뒤 대화와 일정에 따라 의미가 달라집니다.",
    },
    {
        "expression": "생각해 볼게요",
        "situation": "A reply after inviting a friend to a weekend meetup",
        "relationship": "Friends",
        "language": "en",
        "answer": "1. Plain meaning\n'생각해 볼게요' means 'I'll think about it.'\n2. Meaning in this situation\nThe friend may need to check their schedule or may be delaying a decision. This alone is not a definite refusal.\n3. What to say or do\nGive them time. If you need a decision for a reservation, say when you need to know.\n4. What could change the meaning\nTheir tone, earlier conversation, and schedule may change the interpretation.",
    },
    {
        "expression": "편하실 때 연락 주세요",
        "language": "ko",
        "situation": "중고 거래 판매자가 구매자에게 보낸 메시지",
        "relationship": "판매자와 구매자",
        "answer": "1. 쉬운 뜻\n가능한 시간에 연락해 달라는 뜻입니다.\n2. 상황에 따른 해석\n상대의 시간을 배려하는 표현입니다.\n3. 적절한 말이나 행동\n거래를 원한다면 연락 가능한 시간을 제안하세요.\n4. 달라질 수 있는 점\n언제까지 연락해야 하는지는 이 문장만으로 알 수 없습니다.",
    },
    {
        "expression": "편하실 때 연락 주세요",
        "situation": "A seller sends this message to a potential buyer",
        "relationship": "Seller and buyer",
        "language": "en",
        "answer": "1. Plain meaning\n'편하실 때 연락 주세요' means 'Please contact me when it is convenient for you.'\n2. Meaning in this situation\nThe seller is leaving the timing flexible.\n3. What to say or do\nIf you want to proceed, suggest a time when you can contact them.\n4. What could change the meaning\nThis sentence alone does not tell you the latest time to respond.",
    },
]


def export(output_dir: Path) -> None:
    required = ("expression", "situation", "relationship", "language", "answer")
    for split, rows in (("train", TRAIN), ("eval", EVAL)):
        if not rows:
            raise ValueError(f"{split}: 사례가 없습니다")
        for row in rows:
            if any(not isinstance(row.get(field), str) or not row[field].strip() for field in required):
                raise ValueError(f"{split}: 다섯 필드를 모두 채워야 합니다")
            if row["language"] not in ("en", "ko"):
                raise ValueError(f"{split}: language는 en 또는 ko여야 합니다")
    train_keys = {row["expression"].strip() for row in TRAIN}
    eval_keys = {row["expression"].strip() for row in EVAL}
    if train_keys & eval_keys:
        raise ValueError("학습·평가 표현은 중복되면 안 됩니다")
    for split, rows in (("train", TRAIN), ("eval", EVAL)):
        keys = [(row["expression"].strip(), row["situation"].strip(), row["language"]) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{split}: 같은 사례·언어가 중복됐습니다")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in (("train", TRAIN), ("eval", EVAL)):
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
    with tarfile.open(output_dir / "malmaek.tar.gz", "w:gz") as archive:
        for name in ("train.jsonl", "eval.jsonl"):
            archive.add(output_dir / name, arcname=name)
    print(f"학습 {len(TRAIN)}개, 평가 {len(EVAL)}개 → {output_dir / 'malmaek.tar.gz'}")
    if len(train_keys) < 50:
        print("주의: 현재 예시는 형식 확인용입니다. 학습 전 검토한 사례를 더 추가하세요.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    export(parser.parse_args().output_dir)
