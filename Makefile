PYTHON ?= python
DATA ?= data/raw/telco_customer_churn.xlsx

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	pytest -q

train:
	$(PYTHON) src/train_model.py --input "$(DATA)"

evaluate:
	$(PYTHON) src/evaluate_model.py

pipeline: test train evaluate

app:
	streamlit run app.py
