#!/bin/bash

# Remove auxiliary files
rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.loa

# First compilation
xelatex -interaction=nonstopmode SBUKThesis-main.tex

# Run bibtex
bibtex SBUKThesis-main

# Second compilation
xelatex -interaction=nonstopmode SBUKThesis-main.tex

# Third compilation
xelatex -interaction=nonstopmode SBUKThesis-main.tex

# Clean up auxiliary files
rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.loa 