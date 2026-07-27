# Immuno2Hit — triagem QSAR em TREM2, CD28 e IDO1

Todos os modelos por trás dos três manuscritos, num formato `.pkl` único e
auto-contido. Montado em 2026-07-27 por `build_models.py`.

Auto-contido significa que os pickles só dependem de `scikit-learn`, `xgboost` e
`numpy` — não é preciso ter o repositório do ScreenSAR/QSAR_curadoria no `sys.path`,
como acontecia com os `.joblib` originais.

```
models/
  TREM2/  RDKit_SVM.pkl   RDKit_LogisticRegression.pkl   Morgan_SVM.pkl
  CD28/   Morgan_XGBoost.pkl   Morgan_KNN.pkl
  IDO1/   RDKit_XGBoost.pkl    Morgan_RandomForest.pkl    MACCS_SVM.pkl
```

## Plataforma de predição

```bash
cd backend && python app.py    # ou duplo clique em Launch_Immuno2Hit.command
```

Abre `http://127.0.0.1:8765`: cola SMILES (até 100 por rodada), devolve estrutura,
probabilidade por modelo, status do domínio de aplicabilidade e o veredito de consenso
de cada alvo. Deep link: `#smiles=CCO|c1ccccc1` já preenche e roda.

## Deploy

O frontend estático está no Vercel; o Python **não** roda lá. Só as dependências somam ~295 MB no
macOS e passam de 1 GB no Linux, contra um teto de 500 MB por função — e mais da metade disso é
`nvidia-nccl-cu12`, que o wheel `xgboost` arrasta no Linux e que inferência em CPU nunca usa
(por isso o `requirements.txt` usa `xgboost-cpu` fora do macOS).

Duas armadilhas custaram várias builds e ficam registradas aqui:

- O **Framework Preset fica gravado** nas configurações do projeto no momento da importação. Como
  havia `requirements.txt` na raiz, o Vercel salvou um preset Python e continuou rodando o builder
  Python mesmo depois de todo o código ir para `backend/`. O `"framework": null` no `vercel.json`
  sobrescreve isso por implantação, sem depender do painel.
- O **`vercel.json` rejeita a chave `"//"`** usada como comentário — o schema é estrito. O sintoma
  não é óbvio: a build falha apontando para a documentação de project-configuration.

Por isso a raiz do repositório contém apenas `static/` e `vercel.json`: sem `requirements.txt` e sem
nenhum nome de entrypoint Python (`app.py`, `index.py`, `server.py`, `main.py`, `wsgi.py`, `asgi.py`),
a detecção de runtime não tem como disparar.

A página publicada carrega e explica os alvos, mas **não prediz** enquanto não houver uma API Python
viva: os modelos são `.pkl`, e pickle só reconstrói os objetos com scikit-learn e xgboost carregados.
O `Dockerfile` na raiz sobe essa API em qualquer host de container.

## Censo de modelos — conferido contra os papers

| Alvo | No paper | Empacotados aqui |
|---|---|---|
| TREM2 | 3 (ensemble, voto ≥ 2/3) | 3, originais |
| CD28 | 2 no screening (Morgan/XGBoost + Morgan/k-NN aumentados) | 2, originais |
| IDO1 | **9** = 3 regimes × 3 fingerprints | 3, retreinados (só o Baseline) |

**CD28 são 2 mesmo.** O §2.4 diz que "os modelos Morgan/XGBoost e Morgan/k-NN aumentados
pontuaram 66.089 compostos". O Random Forest que aparece no §2.3 (400 árvores, class
weights balanceados) não é um terceiro membro do consenso: é o experimento antes/depois
que demonstra o colapso na classe majoritária (sensibilidade 0 → 0,77 com augmentação),
treinado duas vezes e nunca salvo. Os números dele estão em `CD28/reinvent_before_after_metrics.csv`.

**IDO1 são 9.** O paper define o baseline como "o consenso do melhor algoritmo por
fingerprint" e depois re-roda a grade 3×9 inteira em cada um dos três regimes (Baseline, A,
C) — 3 regimes × 3 fingerprints = 9 modelos implantados. Os 49 compostos do núcleo de
consenso são os aprovados pelos três regimes simultaneamente.

Só o consenso Baseline está empacotado, de propósito: a conclusão do próprio paper é que
os regimes A e C inflam a validação cruzada em 0,08–0,10 sem melhorar a predição real, e
que o C é auto-confirmatório. Implantá-los num preditor propagaria justamente o artefato
que o paper denuncia. Se quiser reconstruí-los mesmo assim, dá: as moléculas do REINVENT4
foram preservadas em `~/anaconda_projects/REINVENT/ido_gen/ido_generated.csv` (regime A,
2.502 moléculas) e `ido_gen_C/ido_C_generated.csv` (regime C, 1.566). Falta saber quais 450
foram escolhidas e quantos inativos de biblioteca entraram no C — o script original sumiu.

## Proveniência dos arquivos

| Alvo | Modelos | Origem |
|---|---|---|
| TREM2 | 3 | **originais do paper**, copiados de `TREM2/Models/TREM2_D3_*.pkl` |
| CD28 | 2 | **originais do paper**, convertidos dos `.joblib` exportados pelo ScreenSAR |
| IDO1 | 3 | **retreinados em 2026-07-27** — os originais nunca existiram em disco |

Os do TREM2 e do CD28 foram validados por regressão contra as probabilidades gravadas nos
CSVs de screening da época: diferença máxima de 5×10⁻⁷ em 800 compostos amostrados, ou
seja, reproduzem exatamente o que gerou os hits publicados.

Os modelos do IDO1 usados em `IDO_Paper_CS.docx` foram treinados em memória e nunca
salvos: `run_qsar_curadoria_IDO1.py` não tem nenhum `joblib.dump`, e nenhum modelo em
`~/Downloads` tem domínio de aplicabilidade compatível com o dataset do IDO. Foram
reconstruídos a partir do **mesmo** dataset curado (`IDO/QSAR_ChEMBL/01_curated.csv`,
623 compostos, 200 ativos / 423 inativos, corte IC50 1 µM) e dos **mesmos**
hiperparâmetros do ScreenSAR (`src/core/modeling.py::_build_model_constructors`).

O AUC por fold difere um pouco do publicado, dentro do ruído de validação cruzada:

| Modelo | AUC no paper | AUC reproduzido (CV 5-fold) | Holdout 20% |
|---|---|---|---|
| RDKit::XGBoost | 0,9165 | 0,9204 ± 0,0343 | 0,9097 |
| Morgan::RandomForest | 0,9091 | 0,9065 ± 0,0368 | 0,8971 |
| MACCS::SVM | 0,9080 | 0,8896 ± 0,0463 | 0,8800 |

Os três valores do paper caem dentro do intervalo obtido variando só a semente do
split (testado com `random_state` 0, 1, 7, 42 e 123), então a diferença é variância de
fold, e não hiperparâmetro divergente. Ainda assim, **estes não são bit-a-bit os
modelos que geraram os hits do paper** — se o objetivo for defender números
publicados, cite os reproduzidos aqui, não os originais.

## Como cada consenso funciona

- **TREM2** — voto majoritário, ≥ 2 de 3 membros positivos no *próprio* threshold de
  Youden (gravado em `decision_threshold`: 0,107 / 0,032 / 0,230). O score do ensemble
  é a média aritmética das três probabilidades.
- **CD28** — consenso de 2, treinados no conjunto aumentado com REINVENT4
  (120 ativos reais + 2.014 análogos sintéticos + 6.403 inativos da Enamine). O paper
  filtra pela média das probabilidades em P ≥ 0,6, dentro do domínio de aplicabilidade.
- **IDO1** — consenso do melhor algoritmo de cada fingerprint, também com corte
  P ≥ 0,6 sobre a média.

## Uso

```python
from predict import load, screen

trem2 = [load(f"models/TREM2/{m}.pkl")
         for m in ("RDKit_SVM", "RDKit_LogisticRegression", "Morgan_SVM")]
df = screen(trem2, ["CCOc1ccccc1C(=O)N", "c1ccc2[nH]ccc2c1"], vote="majority")

ido1 = [load(f"models/IDO1/{m}.pkl")
        for m in ("RDKit_XGBoost", "Morgan_RandomForest", "MACCS_SVM")]
df = screen(ido1, smiles_list, vote="mean")   # gate P >= 0.6
```

Ou pela linha de comando:

```bash
cd backend && python predict.py models/IDO1/*.pkl --smiles biblioteca.csv --out scored.csv
```

SMILES que o RDKit não consegue parsear voltam como `NaN`/`<NA>`, nunca como negativo.

## Esquema do `.pkl`

Cada arquivo é um `dict`:

| Chave | Conteúdo |
|---|---|
| `target`, `fp`, `algorithm`, `model_key` | identificação (ex.: `RDKit::SVM`) |
| `n_bits`, `radius` | parâmetros do fingerprint (1024 bits, raio 2 em todos) |
| `model` | o estimador treinado (nos do TREM2, um `Pipeline` com `StandardScaler`) |
| `decision_threshold` | corte de Youden por modelo; `None` quando o paper decide só no consenso |
| `ad` | domínio de aplicabilidade: `NearestNeighbors` ajustado, `threshold_AD`, distâncias de treino |
| `training` | dataset, nº de amostras, ativos/inativos, em que subconjunto foi ajustado |
| `metrics` | só no IDO1: AUC de CV, holdout e o valor publicado |
| `provenance`, `is_original` | de onde veio; `is_original=False` marca os retreinados |

## Arquivos de apoio

- `backend/app.py` + `static/index.html` — a plataforma web local (servidor da stdlib, sem Streamlit).
  O Python fica todo em `backend/` porque o Vercel detecta projeto Python pela presença de
  `requirements.txt` na raiz e de nomes como `app.py`; com a raiz limpa, ele só publica o site estático.
- `Launch_Immuno2Hit.command` — launcher de duplo clique.
- `backend/build_models.py` — reconstrói tudo do zero (idempotente).
- `backend/fingerprints.py` — geração de MACCS / Morgan / RDKit, idêntica ao ScreenSAR.
- `backend/qsar_ad.py` — cópia autônoma da classe de domínio de aplicabilidade.
- `backend/predict.py` — carregar, pontuar e aplicar o consenso.

## Modelos que ficaram de fora, de propósito

- `TREM2/Models/{MACCS_CatBoost-2, MACCS_Gradient_Boosting, RDKit_CatBoost}.joblib` —
  exports exploratórios do ScreenSAR, não entraram no paper.
- `CD28/1/Results/*` — rodada de maio (UUID `be349ce3`, 5.120 amostras de treino),
  anterior à augmentação com REINVENT4.
