PYTHON := python

install:
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) main.py --data data/sales.csv --out artifacts

test:
	$(PYTHON) -m unittest discover -v
