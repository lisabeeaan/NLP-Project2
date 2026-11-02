# Modern Models, Historical isiXhosa

A reproducible evaluation of how five pretrained language models score nineteenth-century isiXhosa newspaper text relative to a modern comparison corpus.

## Research question

Do selected modern pretrained models assign worse language-modelling scores to isiXhosa newspaper text from 1870-1894 than to modern NCHLT isiXhosa text?

The experiment evaluates frozen checkpoints. It does not train, fine-tune or optimise model hyperparameters. The historical and modern corpora differ in register, topic, orthography and processing, so the comparison cannot isolate historical language change as the cause of the observed gap.

**Main finding:** XLM-R, Afro-XLMR and InkubaLM assign consistently worse character-normalised language-modelling scores to the sampled historical text, including on a source-disjoint held-out test. mBERT and AfriBERTa do not show a clear directional difference. This is evidence of a historical-modern corpus mismatch, not proof of semantic failure or that the models classify historical isiXhosa as a foreign dialect.

## Repository layout

```text
nlp-project-2
├── data/
│   ├── XhTxt/
│   ├── EnTxt/                          # not used in the historical core corpus
│   ├── LanguageSplitTracker70-94.csv
│   ├── historical_clean/
│   │   └── historical_clean.jsonl      # file, sentences
│   ├── modern_corpus/
│   │   ├── NCHLT/Corpora/              # exact clean input selected in script
│   │   └── modern_sentences.jsonl      # text, doc
│   ├── samples/                        # original exploratory samples
│   └── heldout/                        # frozen manifests, protocol and samples
├── isixhosa-env
│   ├── bin/activate
├── scripts/ # 19 scripts in total
│   ├── load_models.py
│   ├── scoring.py
│   ├── extend_analysis.py
│   └── ...                             # preprocessing and exploratory scripts
├── results/
│   ├── filtering/
│   ├── perplexity/                     # exploratory results
│   ├── tokenization/
│   ├── noise/
│   └── heldout/                        # final evaluation outputs
├── run_script.sh                      # export from the successful environment
└── README.md
```


## 1. Environment

A CUDA-capable GPU is strongly recommended due to the scripts being cuda accelerated. Also log into hugging-face (it saves your access token to your local cache)

To create a new environment when needed:

```bash
hf auth login # setup according to your token
mkdir nlp-project-2
python3 -m venv isixhosa-env
source isixhosa-env/bin/activate
python3 -m pip install -r requirements.txt
```

## Reproduction

### 1. Build and inspect the corpora as well as reproducing the exploratory experiments

The following names match the supplied implementation pipeline:

```bash
chmod +x run_script.sh
./run_script.sh
```

### 2. Prepare the fresh held-out evaluation once

```bash
python3 scripts/extend_analysis.py prepare --n 300 --seed 20261031
```

Preparation groups all pages of a historical issue together and all sentences from a modern document together. Previously evaluated source groups are assigned to development. Normalised exact duplicates are removed within corpora; exact cross-corpus overlaps are removed from both. Original sample text is also excluded from validation/test. This does not remove unknown near-duplicates or identify overlap with model pretraining.

Approximately 15% of all source groups are reserved for each of validation and test. Sentence ratios need not be 70/15/15 because source groups vary in size. Character-length bin boundaries come from development historical text; each evaluation corpus contributes 60 sentences to each of five bins. The script fails rather than silently reducing an infeasible requested sample size.

Archive the generated `protocol.json`, `sha256.json`, split manifest and samples. Preparation intentionally refuses to overwrite an existing `data/heldout/`. Do not delete/regenerate the test after inspecting scores.

### 3. Validate the implementation

```bash
python3 scripts/extend_analysis.py score --split validation
python3 scripts/extend_analysis.py analyze --split validation --boots 2000
```

### 4. Evaluate the final test

```bash
python3 scripts/extend_analysis.py score --split test
python3 scripts/extend_analysis.py analyze --split test --boots 2000
```

Scoring refuses to overwrite an existing score file. An interruption can leave a partial file; preserve it separately before a deliberate complete rerun. Automatic resume is not implemented. Analysis rejects missing models, missing scores and duplicate score rows. Do not rerun a modified method on the same test and describe it as a fresh held-out result.


## Final held-out results

Each test corpus contains 300 sentences. Historical sentences span 101 newspaper issues; modern sentences span 112 source documents. Positive gaps indicate worse historical scores within the model.

| Model | Median gap per character | 95% source-group bootstrap CI | Cliff's delta | Approx. Holm-adjusted bootstrap p |
|---|---:|---|---:|---:|
| mBERT | -0.056 | [-0.128, 0.024] | -0.093 | 0.3118 |
| AfriBERTa | -0.026 | [-0.067, 0.020] | -0.047 | 0.3118 |
| XLM-R | 0.796 | [0.696, 0.872] | 0.669 | 0.0025 |
| Afro-XLMR | 0.815 | [0.725, 0.885] | 0.837 | 0.0025 |
| InkubaLM | 0.627 | [0.578, 0.682] | 0.825 | 0.0025 |

The primary statistic is `median(historical NLL per character) - median(modern NLL per character)`. Masked-model NLL denotes negative pseudo-log-likelihood; causal-model NLL denotes negative autoregressive log-likelihood. Compare corpora within each model. Character normalisation does not make different model objectives a common performance ranking.

The 95% intervals use 2,000 bootstrap draws resampling source groups independently within each corpus. Cliff's delta is a descriptive sentence-level order effect, not a causal effect. Approximate centred-bootstrap p-values are adjusted across the five models for each metric. The minimum unadjusted value with this implementation and 2,000 draws is 1/2001, approximately 0.0005; it should not be reported as zero.

### Validation consistency

Validation uses separate source groups: 100 historical issues and 114 modern documents, again with 300 sentences per corpus.

| Model | Validation gap | Test gap |
|---|---:|---:|
| mBERT | -0.039 | -0.056 |
| AfriBERTa | -0.060 | -0.026 |
| XLM-R | 0.857 | 0.796 |
| Afro-XLMR | 0.842 | 0.815 |
| InkubaLM | 0.629 | 0.627 |

Validation means are 115.03 historical versus 111.93 modern characters. These are validation statistics, not test statistics. Sampling matches broad character-length bins rather than exact sentence lengths.

## Models

| Key | Hugging Face checkpoint | Scoring objective |
|---|---|---|
| `mbert` | `bert-base-multilingual-cased` | Masked-token pseudo-log-likelihood |
| `afriberta` | `castorini/afriberta_large` | Masked-token pseudo-log-likelihood |
| `xlmr` | `FacebookAI/xlm-roberta-large` | Masked-token pseudo-log-likelihood |
| `afroxlmr` | `Davlan/afro-xlmr-large` | Masked-token pseudo-log-likelihood |
| `inkubalm` | `lelapa/InkubaLM-0.4B` | Causal next-token log-likelihood |

mBERT and AfriBERTa are comparison baselines: isiXhosa is absent from their documented pretraining language lists. This does not prove that no incidental isiXhosa text occurred in pretraining. Afro-XLMR and InkubaLM explicitly document isiXhosa coverage; XLM-R is a multilingual web-pretrained model. The checkpoint set is not a controlled experiment on the causal effect of training-language inclusion.

Exact model and tokenizer revisions belong in `scripts/load_models.py`; preserve resolved revisions in the experiment outputs. Current model cards may describe later repository states, so archive the revisions actually evaluated.

## Data and provenance

### Historical corpus

The isiXhosa Intellectual Traditions Digital Archive provides digitised historical texts. This experiment uses the instructor-supplied isiXhosa blocks from **Imvo Zabantsundu** and **Isigidimi sama-Xosa**, with tracker coverage from 1870-1894. It does not evaluate the full 1870-1914 archive.

- Archive paper: [Schoots et al. (2023)](https://aclanthology.org/2023.rail-1.7/).
- Isigidimi collection: [permanent DOI](https://doi.org/10.25375/uct.22332271).
- Imvo collection: [permanent DOI](https://doi.org/10.25375/uct.22332268).

`XhTxt/` and `EnTxt/` contain language-separated blocks from pages; they are not parallel translations. Only `XhTxt/` feeds the historical corpus. The instructor-provided tracker is an input; do not assume a public download contains exactly the same filenames, labels or version.

The 100-isiXhosa-word minimum retains 3,592 of 3,608 pages. Automated preprocessing yields 327,265 sentences from 3,574 pages with retained sentences. Minimum sentence thresholds are three whitespace words and 25 characters; the maximum is 80 whitespace words. Other automated rules remove selected fragments, numerical strings, all-capital material and apparent English contamination.

A proposed 90% page-level isiXhosa word-share rule was disabled. Its diagnostic would additionally exclude 895 substantial pages even though English blocks had already been separated. Keep this diagnostic distinct from the actual filter used.

No manual correction or modernisation was performed. The researcher lacked the isiXhosa proficiency needed to distinguish authentic historical spelling from errors reliably. The absence of manual edits does not mean that no automated cleaning occurred.

### Modern corpus

Use the **NCHLT Clean isiXhosa Text Corpus**, not the lexicon, frequency list or raw corpus:

- [Permanent dataset record](https://hdl.handle.net/20.500.12185/314).
- [Eiselen and Puttkammer (2014)](https://aclanthology.org/L14-1106/).
- [Corpus Portal provenance](https://rmaservices.nwu.ac.za/corpusportal/explore/corpus).

The supplied clean file identifies itself as v2.0. Preserve the exact downloaded version and its licence header. Source-document boundaries are identified using `<fn>...</fn>` markers. After automated sentence filters, exact duplicate removal and a maximum of 200 sentences per document, 41,188 modern sentences remain. Government-domain provenance makes register a material limitation of this comparison; do not assume genre labels are present in the selected concatenated clean file.

### Held-out partition pools

After the extension's exact deduplication/overlap handling, the observed manifest contains:

| Corpus | Partition | Available sentences | Source groups |
|---|---|---:|---:|
| Historical | Development (`development_train`) | 184,945 | 502 |
| Historical | Validation | 39,206 | 108 |
| Historical | Test | 39,597 | 108 |
| Modern | Development (`development_train`) | 32,351 | 1,076 |
| Modern | Validation | 4,386 | 231 |
| Modern | Test | 4,451 | 231 |

These are partition pools, not the numbers scored. Validation and test each score 300 sentences per corpus. Actual group counts in the scored subsets are smaller. `development_train` is a partition label; no project-specific training occurs.

## OCR and noise analysis

The original sensitivity experiment corrupts 100 modern sentences at nominal letter rates of 0%, 1%, 2% and 5%. The generic perturbations show model sensitivity; they are not calibrated measurements of historical OCR errors. Reported shifts relative to the original mean per-token gaps must not be relabelled as causal percentages of the final median character-normalised gap.

Actual residual transcription errors and their score effects were not measured. The optional `ocr-template` and `ocr` commands require independently scan-verified reference passages. They were not completed for the reported study. Do not make guessed corrections or modernise authentic historical spelling. Obvious-symbol counts miss plausible-looking errors and are not an actual OCR error rate.

## Interpretation and limitations

- Frozen language-modelling scores are proxies for corpus fit, not direct measures of semantic understanding, representation quality on a labelled task, or downstream harm.
- NCHLT and historical newspapers differ in register, topic, orthography and processing. The experiment cannot identify a pure temporal effect.
- Small baseline gaps do not establish historical robustness, and intervals containing zero do not prove equivalence.
- Standard single-subword PLL reveals other subwords of the same word; this can affect scores of fragmented words. See [Kauf and Ivanova (2023)](https://aclanthology.org/2023.acl-short.80/).
- Character normalisation does not remove all tokenisation effects or make masked and causal objectives directly comparable.
- Source-group splitting improves evaluation separation, but preprocessing was developed before the fresh split. The final test is a held-out assessment of that developed procedure, not a wholly untouched raw-data study.
- Exact deduplication does not identify near-duplicates or model-pretraining contamination.
- No manual linguistic correction, isiXhosa-speaker validation, true OCR audit or downstream task was completed.
- Newspaper differences do not establish a temporal trend, editorial cause, or cultural judgement.

## Responsible use and attribution

Treat historical forms as evidence rather than automatically as errors. Findings concern evaluated systems and sampled corpora; they do not imply linguistic deficiency in historical isiXhosa. Any use in archival access tools should involve isiXhosa readers, historians and relevant communities, with direct evaluation of the intended task.

NCHLT's dataset record specifies CC BY 2.5 South Africa. Model and archive terms are separate: for example, the InkubaLM checkpoint card specifies CC BY-NC 4.0. This README does not grant redistribution rights to source texts, weights or third-party code. Attribute each resource and preserve its applicable terms; do not assign a new blanket licence to third-party materials.

Generative AI assisted implementation, methodological review and report drafting. The researcher is responsible for the final code, claims, references and submission. The project report must include the course-required disclosures in their prescribed format.
