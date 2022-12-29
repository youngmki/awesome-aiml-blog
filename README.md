#
## Research
### CV
* [Autoencoders and Diffusers: A Brief Comparison (Dec 2022)](https://eugeneyan.com//writing/autoencoders-vs-diffusers/)
* [*Data2vec 2.0*: Highly Efficient Self-Supervised Learning for Vision, Speech and Text (Dec 2022, Meta)](https://ai.facebook.com/blog/ai-self-supervised-learning-data2vec/)

### NLP
* [Accelerating Text Generation with Confident Adaptive Language Modeling (*CALM*) (Dec 2022, Google)](https://ai.googleblog.com/2022/12/accelerating-text-generation-with.html?m=1)
* [Illustrating Reinforcement Learning from Human Feedback (RLHF) (Dec 2022, HuggingFace)](https://huggingface.co/blog/rlhf)
① 정책 언어 모델과 ② 프롬프트 별 생성 텍스트에 대해 사람의 선호도 점수로 학습한 보상 모델 이용하여 언어 모델 튜닝(= PPO)

### RecSys, etc.
* [Mastering Stratego, the Classic Game of Imperfect Information (Dec 2022, DeepMind)](https://www.deepmind.com/blog/mastering-stratego-the-classic-game-of-imperfect-information)
* [Research @ MS 2022: A Look Back at a Year of Accelerating Progress in AI (Dec 2022, Microsoft)](https://www.microsoft.com/en-us/research/blog/2022-a-look-back-at-a-year-of-accelerating-progress-in-ai/): 이미지 모델 [Swin Transformer v2](https://www.microsoft.com/en-us/research/blog/swin-transformer-supports-3-billion-parameter-vision-models-that-can-train-with-higher-resolution-images-for-greater-task-applicability), 챗봇을 위한 언어 모델 [GODEL](https://www.microsoft.com/en-us/research/blog/godel-combining-goal-oriented-dialog-with-real-world-conversations/), 멀티모달 [BEiT](https://github.com/microsoft/unilm/tree/master/beit), 언어 모델 디버깅 도구 [AdaTest](https://www.microsoft.com/en-us/research/blog/partnering-people-with-large-language-models-to-find-and-fix-bugs-in-nlp-systems/), 유해 콘텐츠 데이터 생성 도구 [ToxiGen](https://www.microsoft.com/en-us/research/blog/detoxigen-leveraging-large-language-models-to-build-more-robust-hate-speech-detection-tools/), NN HPO기법 [µTransfer](https://www.microsoft.com/en-us/research/blog/%c2%b5transfer-a-technique-for-hyperparameter-tuning-of-enormous-neural-networks/), 모델 압축과 양자화를 위한 [DeepSpeed Compression](https://www.microsoft.com/en-us/research/blog/deepspeed-compression-a-composable-library-for-extreme-compression-and-zero-cost-quantization/) 등

## Business Cases and Open Source Libraries
### CV
* [When a Picture is Worth More Than Words (Dec 2022, Airbnb)](https://medium.com/airbnb-engineering/when-a-picture-is-worth-more-than-words-17718860dcc2): ① 미적 점수 레이블링과 예측 ② 자가 학습으로 이미지 임베딩 생성 → HNSW로 유사 이미지 검색 지원

### NLP
* [Build a Robust Text-Based Toxicity Predictor (Dec 2022, Amazon)](https://aws.amazon.com/ko/blogs/machine-learning/build-a-robust-text-based-toxicity-predictor/)
* [Luda Gen 1, 더 재미있고 자연스러운 대화로 돌아온 루다 1편 - 생성 기반 챗봇 (Dec 2022, 스캐터랩)](https://tech.scatterlab.co.kr/luda-gen-1/): 검색 대신 생성 모델(= GPT2) 기반 챗봇 도입 ← 관계 지향적, 안정성 위해 파인 튜닝 

### RecSys, etc.
* [How to Evaluate the Quality of the Synthetic Data – Measuring from the Perspective of Fidelity, Utility, and Privacy (Dec 2022)](https://aws.amazon.com/blogs/machine-learning/how-to-evaluate-the-quality-of-the-synthetic-data-measuring-from-the-perspective-of-fidelity-utility-and-privacy/): [이전 글](https://aws.amazon.com/blogs/machine-learning/augment-fraud-transactions-using-synthetic-data-in-amazon-sagemaker/)도 읽어보세요. [ydata-synthetic](https://github.com/ydataai/ydata-synthetic) 라이브러리의 WGAN-GP로 태뷸러 데이터 합성 → 충실도(통계량), 유용성(예측 성능), 정보 보호(암기 X) 기준으로 평가

### Training, Inference and MLOps
* [Automatically Retrain NNs with **Renate** (Dec 2022)](https://aws.amazon.com/blogs/machine-learning/automatically-retrain-neural-networks-with-renate/): 신경망 자동 재학습을 위한 라이브러리, [**Renate**](https://renate.readthedocs.io/en/latest/) ← 재학습 시 과거 데이터에 대한 예측 성능 하락하는 '파국적 망각' 현상 방지, 지속적 학습에 대한 HPO와 클라우드 백엔드 학습 지원
* [CLOVA MD 상품추천 솔루션 (Dec 2022, 네이버)](https://engineering.clova.ai/posts/2022/12/clova-md-solution): 적재(Hive 또는 Kafka) → 학습과 추론(Argo Workflows 또는 Kubeflow) → 결과 저장(MongoDB)의 일 단위 배치 작업
* [Introducing **Fortuna**: A Library for Uncertainty Quantification (Dec 2022)](https://aws.amazon.com/blogs/machine-learning/introducing-fortuna-a-library-for-uncertainty-quantification/): 딥러닝의 과신 문제 → [Fortuna](https://aws-fortuna.readthedocs.io/en/latest/) = 불확실성 정량화 라이브러리 ① Flax 모델 ② 출력이나 불확실성 추정값에서 시작 가능
* [ML 모델 서빙 비용 1/4로 줄이기 (Dec 2022, 하이퍼커넥트)](https://hyperconnect.github.io/2022/12/13/infra-cost-optimization-with-aws-inferentia.html)
* [Ready-to-go Sample Data Pipelines with **Dataflow** (Dec 2022, Netflix)](https://netflixtechblog.com/ready-to-go-sample-data-pipelines-with-dataflow-17440a9e141d)

## AWS ML Only
### RecSys, etc.
* [Power Recommendations and Search Using an IMDb Knowledge Graph (Dec 2022)](https://aws.amazon.com/blogs/machine-learning/part-2-power-recommendations-and-search-using-an-imdb-knowledge-graph/): [이전 글](https://aws.amazon.com/blogs/machine-learning/part-1-power-recommendation-and-search-using-an-imdb-knowledge-graph/)도 읽어보세요. ① 데이터 적재: S3에 IMDb 데이터 저장 → Glue 통해 Neptune Gremlin 포맷으로 가공 → Neptune 클러스터에 적재 → Gremlin 쿼리 가능 ② DGN의 GNN 학습: S3로 내보내기 작업 → 데이터 처리(모델 유형: 이기종 또는 지식 그래프) → 모델 훈련 → 임베딩 다운로드
