DOC := main
.DEFAULT_GOAL := all
DATA_DIR := literature_recent_discrete_models/replication_code_sources
DATA_FILE := $(DATA_DIR)/kuairec_caption_category.csv
DATA_MD5 := 31bc38cdccdf75a71df137779035f8cb

$(DATA_FILE):
	mkdir -p $(DATA_DIR)
	curl -L --fail --retry 3 -o $(DATA_FILE) 'https://zenodo.org/records/18164998/files/kuairec_caption_category.csv?download=1'

data: $(DATA_FILE)
	python3 -c "import hashlib,pathlib; p=pathlib.Path('$(DATA_FILE)'); h=hashlib.md5(p.read_bytes()).hexdigest(); assert h == '$(DATA_MD5)', h; print('verified', p, h)"

solve: data
	MPLCONFIGDIR=/tmp/matplotlib-dynamic-paper python3 scripts/solve_dynamic_equilibria.py

all:
	xelatex -interaction=nonstopmode -synctex=1 $(DOC).tex
	bibtex $(DOC)
	xelatex -interaction=nonstopmode -synctex=1 $(DOC).tex
	xelatex -interaction=nonstopmode -synctex=1 $(DOC).tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.nav *.snm *.vrb *.synctex.gz

.PHONY: data solve all clean
