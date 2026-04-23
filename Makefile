.PHONY: install nltk-data

install:
	pip install -r requirements.txt
	python -m nltk.downloader wordnet

nltk-data:
	python -m nltk.downloader wordnet