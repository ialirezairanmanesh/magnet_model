# Makefile برای کامپایل مقاله پژوهشی MAGNET

# متغیرها
PAPER = MAGNET_Research_Paper
LATEX = xelatex
BIBTEX = bibtex

# هدف اصلی
all: $(PAPER).pdf

# کامپایل مقاله
$(PAPER).pdf: $(PAPER).tex
	$(LATEX) $(PAPER).tex
	$(LATEX) $(PAPER).tex
	$(LATEX) $(PAPER).tex

# کامپایل با مراجع (در صورت استفاده از BibTeX)
paper-with-bib: $(PAPER).tex
	$(LATEX) $(PAPER).tex
	$(BIBTEX) $(PAPER)
	$(LATEX) $(PAPER).tex
	$(LATEX) $(PAPER).tex

# پاکسازی فایل‌های موقت
clean:
	rm -f $(PAPER).aux $(PAPER).log $(PAPER).out $(PAPER).toc $(PAPER).lof $(PAPER).lot
	rm -f $(PAPER).bbl $(PAPER).blg $(PAPER).fdb_latexmk $(PAPER).fls
	rm -f $(PAPER).synctex.gz

# پاکسازی کامل (شامل PDF)
distclean: clean
	rm -f $(PAPER).pdf

# نمایش راهنما
help:
	@echo "استفاده:"
	@echo "  make            - کامپایل مقاله"
	@echo "  make paper-with-bib - کامپایل با مراجع BibTeX"
	@echo "  make clean      - پاکسازی فایل‌های موقت"
	@echo "  make distclean  - پاکسازی کامل"
	@echo "  make help       - نمایش این راهنما"

# اهداف که فایل نیستند
.PHONY: all clean distclean help paper-with-bib 