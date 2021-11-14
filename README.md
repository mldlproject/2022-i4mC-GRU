# i4mC-GRU: Identifying DNA N4-Methylcytosine Sites in Mice using Bidirectional GRU and Sequence-embedded Features


#### T-H Nguyen-Vo, Q. H. Trinh, L. Nguyen, P-U. Nguyen-Hoang, [S. Rahardja](http://www.susantorahardja.com/)*, [B. P. Nguyen](https://homepages.ecs.vuw.ac.nz/~nguyenb5/about.html)∗

![alt text](https://github.com/mldlproject/2022-i4mC-GRU/blob/main/i4mC_GRU_abs0.svg)

## Motivation
N4-methylcytosine (4mC) is one of the most common DNA methylation modifications found in both prokaryotic and eukaryotic genomes
besides N6-methyladenine and N5-methylcytosine. 4mC is catalyzed by the N4 cytosine-specific DNA methyltransferase (DNMT) to form a new bond attaching
a methyl group to the amino group at the C4 position of cytosine. Numerous studies have confirmed the essential biological roles of 4mC in DNA replication and 
repair, cell cycle, controlling gene expression levels, epigenetic inheritance, genome stabilization, recombination, and evolution. Therefore, determining the
location of 4mC is critical for the investigation of physiological and pathological mechanisms. In this study, we propose a more effective computational method
called 4mC-GRU using a gated recurrent unit and duplet sequence-embedded features to predict 4mC sites in DNA sequences. The sequence samples were collected from 
the MethSMRT Database and then refined to build a benchmark dataset. The sequence samples are of *Mus musculus* (mice) species. 

## Results
Results indicate that i4mC-GRU achieves both AUC-ROC and AUC-PR values of 0.98 for prediction tasks on standard sequence samples. The model also works
well with non-standard sequence samples with AUC-ROC values ranging from 067 to 0.74 and AUC-PR values ranging from 0.72 to 0.83. .i4mC-GRU is an efficient, robust, 
and stable prediction model for detect N4-methylcytosine sites in DNA sequences of mice. Also, the comparative analysis on the independent test set indicates that our
method outperforms other state-of-the-art methods.

## Availability and implementation
Source code and data are available upon request. 

## Web-based Application
[Click here](http://103.130.219.193:8003/)

## Contact 
[Go to contact information](https://homepages.ecs.vuw.ac.nz/~nguyenb5/contact.html)
