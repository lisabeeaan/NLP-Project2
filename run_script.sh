#!/bin/bash
#-----------------------------------------------------------------------
source isixhosa-env/bin/activate
#------------------------------------------------
echo "Phase 1 - Building the historical corpus"

echo "1.1 Building the Historical Corpus"
python3 scripts/validate_tracker.py
echo "1.2 Load isiXhosa .txt file files, filtered by the given tracker"
python3 scripts/load_txt.py
echo "1.3 Check what the filter dropped"
python3 scripts/inspect_filtering.py
echo "1.4 How clean?"
python3 scripts/check_contamination.py
echo "1.5 Clean the text and split into sentences"
python3 scripts/clean_corpus.py
echo "1.5 Check cleaned corpus"
python3 scripts/corpus_stats.py

#-----------------------------------------------------------------------

#-----------------------------------------------------------------------

echo "Phase 2 - Assembling the Modern Comparison Corpus"
python3 scripts/get_modern_corpus.py
echo "2.1 Make samples"
python3 scripts/make_samples.py

#-----------------------------------------------------------------------

#-----------------------------------------------------------------------

echo "Phase 3 - Setting up the models"
python3 scripts/load_models.py
echo "sanity-check each model on one sentence"
python3 scripts/test_model_load.py

#-----------------------------------------------------------------------

#-----------------------------------------------------------------------

echo "Phase 4 - Computing Perplexity / Pseudo-Log-Likelihood"

python3 scripts/scoring.py
python3 scripts/run_scoring.py
echo "OCR-noise sensitivity check"
python3 scripts/noise_injection.py
echo "Visible junk left"
python3 scripts/junk_rate.py
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
echo "Phase 5 - Tokenization Analysis"
python3 scripts/tokenization_analysis.py
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
echo "Phase 6 - Statistical Comparison and Visualization"
python3 scripts/analyze_results.py
python3 scripts/analyze_extra.py
#-----------------------------------------------------------------------
