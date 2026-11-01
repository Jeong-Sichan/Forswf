"""Check that training and evaluation cases are separate before fine-tuning."""

import argparse

from prompting import read_examples, require_local_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    args = parser.parse_args()
    require_local_dataset(args.train)
    require_local_dataset(args.eval)
    train = read_examples(args.train)
    evaluation = read_examples(args.eval)
    train_phrases = {row["expression"].strip() for row in train}
    eval_phrases = {row["expression"].strip() for row in evaluation}
    overlap = train_phrases & eval_phrases
    if overlap:
        raise ValueError(f"학습·평가 표현 중복: {', '.join(sorted(overlap))}")
    print(f"학습 {len(train)}개, 평가 {len(evaluation)}개, 표현 중복 없음")


if __name__ == "__main__":
    main()
