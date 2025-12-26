.PHONY: help install data train predict clean

help:
	@echo "Potential Talents - Makefile Commands"
	@echo "======================================"
	@echo "make install       - Install all dependencies"
	@echo "make notebook      - Start Jupyter notebook server"
	@echo "make train         - Train the ranking model"
	@echo "make test          - Run tests (when available)"
	@echo "make clean         - Clean generated files"
	@echo "make structure     - Show project structure"

install:
	pip install -r requirements.txt
	pip install -e .
	@echo "Installation complete!"

notebook:
	jupyter notebook notebooks/

train:
	python -c "from src.dataset import load_raw_data, preprocess_candidates; \
	from src.modeling.train import train_ranking_model, save_model; \
	df = load_raw_data('potential-talents - Aspiring human resources - seeking human resources.csv'); \
	df = preprocess_candidates(df); \
	model = train_ranking_model(df, 'aspiring human resources seeking human resources'); \
	save_model(model, 'hr_ranking_model.pkl'); \
	print('Model trained and saved!')"

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	@echo "Cleaned up temporary files"

structure:
	tree -I '__pycache__|*.pyc|.git' -L 3
