install:
    pip install -r requirements.txt

run-notebook:
    jupyter notebook

test:
    pytest tests/

clean:
    rm -rf __pycache__ data/processed models/
