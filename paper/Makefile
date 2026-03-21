TEX = pdflatex
BIB = bibtex
MAIN = main

.PHONY: all clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib
	$(TEX) $(MAIN)
	$(BIB) $(MAIN)
	$(TEX) $(MAIN)
	$(TEX) $(MAIN)

clean:
	rm -f $(MAIN).{aux,bbl,blg,log,out,pdf,toc,fls,fdb_latexmk,synctex.gz}
