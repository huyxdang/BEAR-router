#!/bin/bash
set -e

echo "============================================"
echo "  Step 1: Prepare Data"
echo "============================================"
python scripts/01_prepare_data.py

echo ""
echo "============================================"
echo "  Step 2: Smoke Test (Validate)"
echo "============================================"
python scripts/00_validate.py

echo ""
echo "============================================"
echo "  Step 3: Grid Search"
echo "============================================"
python scripts/02_grid_search.py

echo ""
echo "============================================"
echo "  Step 4: LLM Judge (submit batch)"
echo "============================================"
python scripts/06_llm_judge_batch.py submit

echo ""
echo "=========================================="
echo "  PAUSE: Wait for judge batch to complete."
echo "  Check status:  python scripts/06_llm_judge_batch.py status"
echo "  Then download:  python scripts/06_llm_judge_batch.py download"
echo "  Then run:       bash run.sh post-judge"
echo "=========================================="

if [ "$1" = "post-judge" ]; then
    echo ""
    echo "============================================"
    echo "  Step 5: Build Router"
    echo "============================================"
    python scripts/03_build_router.py

    echo ""
    echo "============================================"
    echo "  Step 6: K-Tuning + Cross-Validation"
    echo "============================================"
    python scripts/08_tune_and_cv.py

    echo ""
    echo "============================================"
    echo "  Step 7: Evaluate"
    echo "============================================"
    python scripts/04_evaluate.py

    echo ""
    echo "============================================"
    echo "  Step 8: Visualize"
    echo "============================================"
    python scripts/05_visualize.py

    echo ""
    echo "============================================"
    echo "  DONE"
    echo "============================================"
fi
