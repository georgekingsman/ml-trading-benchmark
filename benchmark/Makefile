.PHONY: all data features split train backtest report clean help

CONFIG ?= config/settings.yaml

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

all: benchmark  ## Run full pipeline

benchmark:  ## Run full benchmark (data → features → train → backtest → report)
	python run_all.py --config $(CONFIG)

benchmark-cached:  ## Run benchmark with cached data & features
	python run_all.py --config $(CONFIG) --skip-download --skip-features

data:  ## Download and process data only
	python scripts/data_pipeline.py --config $(CONFIG)

features:  ## Compute features only
	python scripts/feature_engineering.py --config $(CONFIG)

split:  ## Run train/val/test split only
	python scripts/split.py --config $(CONFIG)

clean:  ## Remove generated data & results
	rm -rf data/raw/*.parquet data/processed/*.parquet
	rm -rf features/*.parquet features/splits/
	rm -rf backtest/*.parquet
	rm -rf reports/tables/* reports/figures/*
	rm -rf models/*.pkl

install:  ## Install dependencies
	pip install -r requirements.txt
