# Makefile for Windows Thesis Compilation
# Usage: make or make all

MAIN = SBUKThesis-main
TEX = xelatex
BIB = biber

.PHONY: all clean help

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	$(TEX) $(MAIN).tex
	$(BIB) $(MAIN)
	$(TEX) $(MAIN).tex
	$(TEX) $(MAIN).tex

clean:
	del /Q *.aux *.log *.out *.toc *.lof *.lot *.loa *.bbl *.bcf *.blg *.run.xml *.synctex.gz 2>nul

help:
	@echo Usage:
	@echo   make     - Compile the thesis
	@echo   make clean - Clean auxiliary files
	@echo   make help - Show this help
