# Folders

$folders = @(

"app",
"app/pages",

"artifacts",
"artifacts/extracted_json",
"artifacts/shap_plots",
"artifacts/generated_reports",
"artifacts/logs",
"artifacts/temp",

"config",

"data",
"data/lendingclub",
"data/lendingclub/raw",
"data/lendingclub/processed",

"data/raw_documents",
"data/raw_documents/bank_statements",
"data/raw_documents/loan_forms",

"data/processed",
"data/processed/features",

"models",

"notebooks",

"reports",
"reports/figures",

"sample_documents",
"sample_documents/bank_statements",
"sample_documents/loan_forms",

"src",

"src/data",
"src/documents",
"src/document_ai",
"src/schemas",
"src/features",
"src/risk_models",
"src/fraud",
"src/rules",
"src/explainability",
"src/reports",
"src/pipelines",
"src/utils",

"tests"

)


foreach ($folder in $folders)
{
    New-Item -ItemType Directory -Force -Path $folder
}



# Files

$files = @(

"app/Home.py",

"app/pages/01_Document_Extraction.py",
"app/pages/02_Risk_Assessment.py",
"app/pages/03_Fraud_Analysis.py",
"app/pages/04_Explainability.py",
"app/pages/05_Report_Generation.py",

"config/rules.yaml",
"config/model_config.yaml",
"config/prompts.yaml",

"models/metadata.json",

"src/data/__init__.py",
"src/data/preprocess.py",

"src/documents/__init__.py",
"src/documents/pdf_parser.py",
"src/documents/ocr_engine.py",
"src/documents/document_router.py",

"src/document_ai/__init__.py",
"src/document_ai/loan_extractor.py",
"src/document_ai/statement_extractor.py",
"src/document_ai/transaction_classifier.py",

"src/schemas/__init__.py",
"src/schemas/loan_schema.py",
"src/schemas/statement_schema.py",
"src/schemas/transaction_schema.py",

"src/features/__init__.py",
"src/features/feature_engineering.py",

"src/risk_models/__init__.py",
"src/risk_models/train.py",
"src/risk_models/predict.py",

"src/fraud/__init__.py",
"src/fraud/income_checks.py",
"src/fraud/employer_checks.py",
"src/fraud/behavior_checks.py",
"src/fraud/fraud_engine.py",

"src/rules/__init__.py",
"src/rules/rule_engine.py",

"src/explainability/__init__.py",
"src/explainability/shap_utils.py",

"src/reports/__init__.py",
"src/reports/llm_report.py",
"src/reports/pdf_report.py",

"src/pipelines/__init__.py",
"src/pipelines/document_pipeline.py",
"src/pipelines/risk_pipeline.py",
"src/pipelines/report_pipeline.py",

"src/utils/__init__.py",
"src/utils/config.py",
"src/utils/logger.py",

"tests/test_documents.py",
"tests/test_features.py",
"tests/test_models.py",
"tests/test_fraud.py",

".env",
".gitignore",
"requirements.txt",
"README.md",
"main.py"

)


foreach ($file in $files)
{
    New-Item -ItemType File -Force -Path $file
}

Write-Host ""
Write-Host "Project structure created successfully!"