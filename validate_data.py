"""Check that training and evaluation cases are separate before fine-tuning."""

import argparse

from prompting import read_examples, require_local_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--grounded", action="store_true", help="Check reviewed, cited source-grounded examples")
    args = parser.parse_args()
    require_local_dataset(args.train)
    require_local_dataset(args.eval)
    train = read_examples(args.train)
    evaluation = read_examples(args.eval)
    if args.grounded:
        for split, rows in (("train", train), ("eval", evaluation)):
            for number, row in enumerate(rows, 1):
                if row.get("reviewed") is not True or "evidence" not in row:
                    raise ValueError(f"{split}:{number}: 검수 완료 표시와 evidence 목록이 필요합니다")
                if row["evidence"] and "[1]" not in row["answer"]:
                    raise ValueError(f"{split}:{number}: 근거 문서가 있다면 답변에 [1] 인용이 필요합니다")
                if not row["evidence"] and "[1]" in row["answer"]:
                    raise ValueError(f"{split}:{number}: 근거 문서 없이 [1]을 인용할 수 없습니다")
    train_phrases = {row["expression"].strip() for row in train}
    eval_phrases = {row["expression"].strip() for row in evaluation}
    overlap = train_phrases & eval_phrases
    if overlap:
        raise ValueError(f"학습·평가 표현 중복: {', '.join(sorted(overlap))}")
    print(f"학습 {len(train)}개, 평가 {len(evaluation)}개, 표현 중복 없음")


if __name__ == "__main__":
    main()
