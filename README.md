# i4mC-GRU: Identifying DNA N4-Methylcytosine Sites in Mouse Genomes using Bi-GRU and Sequence-embedded Features


#### T-H Nguyen-Vo, Q. H. Trinh, L. Nguyen, P-U. Nguyen-Hoang, [S. Rahardja](http://www.susantorahardja.com/)*, [B. P. Nguyen](https://homepages.ecs.vuw.ac.nz/~nguyenb5/about.html)∗

![alt text](https://github.com/mldlproject/2022-i4mC-GRU/blob/main/i4mC_GRU_abs0.svg)

## Motivation
N4-methylcytosine (4mC) is one of the most common DNA methylation modifications found in both prokaryotic and eukaryotic
genomes besides N6-methyladenine and N5-methylcytosine. This methylation type has essential biological roles comprising of
replication and repair, cell cycle, controlling gene expression levels, epigenetic inheritance, genome stabilization, recombination,
and evolution. Determining the location of 4mC is therefore critical to investigating physiological and pathological mechanisms.
In this study, we propose an effective computational method called i4mC-GRU using a gated recurrent unit and duplet sequenceembedded features to 
predict 4mC sites in DNA sequences. The sequence samples of Mus musculus (mice) were collected from the
MethSMRT database and then were refined to build a benchmark dataset.

## Results
To fairly assess the model performance, we compared
our method with several state-of-the-art methods. Our results showed that i4mC-GRU had achieved the area under the receiver
operating characteristic curve of 0.97 and the area under the precision-recall curve of 0.98. Comparative analysis also indicated
that our method had performed better than other methods. As an efficient, robust, and stable method, i4mC-GRU contributes to the
extension of computational tools predicting N4-methylcytosine sites in mouse genomes. 

## Availability and implementation
Source code and data are available upon request. 

## Web-based Application
- Source 1: [Click here](http://124.197.54.240:5002/)
- Source 2: [Click here](http://14.177.208.167:5002/)

## Citation
Thanh-Hoang Nguyen-Vo, Quang H. Trinh, Loc Nguyen, Phuong-Uyen Nguyen-Hoang, Susanto Rahardja*, Binh P. Nguyen* (2023). i4mC-GRU: Identifying DNA N4-Methylcytosine sites in mouse genomes using bidirectional gated 
recurrent unit and sequence-embedded features. *Computational and Structural Biotechnology Journal, 21, 3045-3053.* [DOI: 10.1016/j.csbj.2023.05.014](https://doi.org/10.1016/j.csbj.2023.05.014).

## Contact 
[Go to contact information](https://homepages.ecs.vuw.ac.nz/~nguyenb5/contact.html)
