# Robot Action Video Generation

[DACON 2026 인하 인공지능 챌린지](https://dacon.io/competitions/official/236736/overview/description)를 활용한 첫 실제 프로젝트입니다.
초기 이미지와 로봇 행동 시퀀스를 입력으로 받아 미래 영상을 생성하는 과제입니다.

현재 단계는 **데이터 이해**입니다. 모델 학습·추론·제출은 아직 실행하지 않았습니다.

## 최소 환경

Python 3.12와 uv를 사용합니다.

```bash
uv sync --locked
uv run python scripts/inspect_data.py --output reports/local/inventory.json
```

## 데이터 위치

공식 `open.zip`을 참가 동의 후 직접 다운로드하여 `data/raw/` 아래에 압축 해제합니다.
`data/raw/data/train`, `data/raw/data/eval`, `data/raw/baseline`, `data/raw/submission_kit` 구조입니다.
원본 ZIP은 별도 보관합니다. 데이터·공식 코드·체크포인트·개별 샘플 출력은 Git에 포함하지 않습니다.

## Jupyter로 데이터 확인

```bash
uv sync --locked --group notebook
uv run --group notebook jupyter lab notebooks/01_data_overview.ipynb
```

노트북은 데이터 구성, 메타데이터, 평가 이미지·행동 시퀀스,
학습 Parquet 및 대응 영상을 확인합니다. 모델 학습이나 공식 코드를 실행하지 않습니다.
커밋 전에는 `Kernel > Restart Kernel and Clear Outputs of All Cells`로 실행 출력을 지워 주세요.

## 분석 및 다음 단계

[초기 데이터 분석](docs/data-overview.md)을 먼저 읽으세요.

1. 학습 에피소드 하나의 영상·행동·상태를 함께 확인하기
2. 에피소드 단위 검증 분리와 16-step 샘플링 정책 정의하기
3. 사용할 GPU 환경 확인 후 공식 베이스라인을 별도 환경에서 실행하기

평가 입력은 학습·정규화 통계·검증 분리에 사용하지 않습니다. 제출킷과 공식 체크포인트는 수정하지 않습니다.
이 초기 환경은 데이터 점검용이며, 공식 모델 실행 환경과는 별개입니다.
