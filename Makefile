.PHONY: db schema sample ingest bench eval calib test api ui all
db:      ; docker compose up -d db
schema:  ; PYTHONPATH=src python -c "from finrag.store import db; from finrag.config import CFG; \
           conn=db.connect().__enter__(); db.init_schema(conn, CFG.embed_dim); print('schema ready')"
sample:  ; python tools/make_sample_pdf.py
ingest:  ; PYTHONPATH=src python -m finrag.ingest.run_ingest --pdf data/raw/*.pdf \
           --entity "$(ENTITY)" --fiscal-year "$(FY)" --also-naive --init
bench:   ; PYTHONPATH=src python eval/build_benchmark.py
eval:    ; PYTHONPATH=src python eval/run_eval.py --arms all --split test
calib:   ; PYTHONPATH=src python eval/calibrate.py
test:    ; PYTHONPATH=src pytest tests -q
api:     ; PYTHONPATH=src uvicorn finrag.api:app --reload --port 8000
ui:      ; streamlit run ui/app.py
all: sample ingest bench eval
