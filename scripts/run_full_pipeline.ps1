# Kaggle Indoor Location - Full Pipeline Automated Script
# 一键式自动化提分流水线

Write-Host "🚀 Starting Full Site Training (Phase 1)..." -ForegroundColor Cyan
python models/train_lgbm_baseline.py

Write-Host "📝 Generating Initial Baseline Submission..." -ForegroundColor Cyan
python scripts/generate_submission.py --out submission_baseline.csv

Write-Host "🏗️  Mining Topological Grids (Phase 2)..." -ForegroundColor Cyan
python data_processing/build_topological_grids.py

Write-Host "✨ Applying Viterbi Snap-to-Grid Optimization..." -ForegroundColor Cyan
python scripts/postprocess_viterbi.py --sub submission_baseline.csv --out submission_viterbi_final.csv

Write-Host "✅ All processes completed! Please submit 'submission_viterbi_final.csv' to Kaggle." -ForegroundColor Green
