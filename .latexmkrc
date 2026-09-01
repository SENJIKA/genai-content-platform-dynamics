$pdf_mode = 5; # 5 => xelatex
$pdflatex = 'xelatex -interaction=nonstopmode -synctex=1 %O %S';
$bibtex = 'bibtex %O %S';
$clean_ext = "aux bbl bcf blg dvi fdb_latexmk fls log lot out toc bcf run.xml synctex.gz";
