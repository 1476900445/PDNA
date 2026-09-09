# PDNA Paper Model Engineering Reproduction
This project implements Section 3 and Tables 4–6 of the paper *PDNA: An Automated Negotiation Dialogue Framework Integrating Interpretable Opponent Modeling and Reinforcement Learning Decision-Making*. By default, the project reads `../LLM_DATA` and `../NEGMAS_DATA` from the parent directory.

## Project Structure
```text
PDNA_Reproduction/
├── configs/pdna.yaml
├── pdna/
│   ├── bayesian/      # Equations (1)-(7): MLPenc, MLPlex, CPT, DAG calibration, additive product sensitivity
│   ├── expo/          # Equations (8)-(11): Intention layer, policy layer, Base/Edit/Q/On-the-fly Policy
│   ├── fql/           # Equations (12)-(17): 7-dimensional state, inverse utility mapping, Flow Q-Learning
│   ├── setformer/     # Equations (18)-(20): CNN/FC/BiLSTM, gated fusion, SeT, Transformer
│   ├── inspection/    # Equations (21)-(22): DeepSeek checking & correction interface and three-pass correction loop
│   ├── data/          # Strongly typed data models, JSONL datasets, 8:2 mixed replay buffer
│   ├── training/      # Separate optimizers and training procedures for each module
│   └── pipeline.py    # End-to-end orchestration of five modules
├── scripts/
├── tests/
└── docs/
```

## Paper-to-Code Mapping
| Paper Content | Code Location |
|---|---|
| History encoding and lexical encoding, Eq.(1) | pdna/bayesian/model.py |
| C→I→S→O joint probability and CPT, Eq.(2) | pdna/bayesian/cpt.py |
| Posterior update, targeted calibration, Bayesian posterior, Eqs.(3)-(6) | pdna/bayesian/model.py |
| Additive product sensitivity 0.8/0.1/0.05/0.05, Eq.(7) | pdna/bayesian/model.py |
| EXPO 40/45-dimensional state, Eqs.(8), (10) | pdna/expo/state.py |
| Binary selection among Base/Edit/Q-Critic/On-the-fly, Eq.(11) | pdna/expo/policy.py |
| 7-dimensional utility state of the most recent three rounds, Eq.(12) | pdna/fql/state.py |
| Inverse utility interval search and opponent utility maximization, Eq.(13) | pdna/fql/offer.py |
| Flow Matching, twin Q networks, one-step distillation, Eqs.(15)-(17) | pdna/fql/model.py |
| Triple encoders, gated fusion, Nyström RBF, Sinkhorn | pdna/setformer/ |
| Five joint losses with weights from the paper, Eq.(20) | pdna/setformer/model.py |
| Thresholds 0.85/0.80, politeness bias 0.05, maximum three corrections | pdna/inspection/corrector.py |

## Data Directory Layout
```text
Parent Directory/
├── LLM_DATA/
│   ├── train.jsonl
│   ├── validation.jsonl
│   ├── test.jsonl
│   ├── tokenizer/
│   ├── terminology.json
│   ├── knowledge_base.json
│   └── deepseek-14b-negotiation/
├── NEGMAS_DATA/
│   ├── offline.jsonl
│   ├── online.jsonl
│   ├── scenarios/
│   └── utility_functions/
└── PDNA_Reproduction/
```

# PDNA Paper Model Engineering Reproduction
PDNA targets bilateral multi-issue natural language negotiation for cloud computing service trading and consists of five modules. The Bayesian feature recognition module infers opponent characteristics and additive product sensitivity via a personality-intention-strategy directed acyclic graph, combining current utterances and historical interactions. The EXPO module adopts hierarchical reinforcement learning to sequentially select self-intentions and negotiation strategies. The FQL module determines configurations, prices and add-on product schemes within the continuous utility space, and generates offers through the inverse utility function. SeTformer integrates domain terminology, opponent features, decision outputs and negotiation history to generate natural language responses. The inspection and correction module finally audits content and politeness, and revises abnormal responses.

