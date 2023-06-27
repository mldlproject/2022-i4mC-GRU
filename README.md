# i4mC-GRU: Identifying DNA N4-Methylcytosine Sites in Mouse Genomes using Bi-GRU and Sequence-embedded Features


#### T-H Nguyen-Vo, Q. H. Trinh, L. Nguyen, P-U. Nguyen-Hoang, [S. Rahardja](http://www.susantorahardja.com/)*, [B. P. Nguyen](https://homepages.ecs.vuw.ac.nz/~nguyenb5/about.html)∗

![alt text](https://github.com/mldlproject/2022-i4mC-GRU/blob/main/i4mC_GRU_abs0.svg)

## Motivation
N4-methylcytosine (4mC) is one of the most common DNA methylation modifications found in both prokaryotic and eukaryotic genomes. Since the 4mC has various essential biological roles, 
determining its location helps reveal unexplored physiological and pathological pathways. In this study, we propose an effective computational method called i4mC-GRU using a gated 
recurrent unit and duplet sequence-embedded features to predict potential 4mC sites in mouse (Mus musculus) genomes. To fairly assess the performance of the model, we compared our method 
with several state-of-the-art methods using two different benchmark datasets.

## Results
Our results showed that i4mC-GRU achieved area under the receiver operating characteristic curve values of 0.97 and 0.89 and area under the precision-recall curve values of 0.98 
and 0.90 on the first and second benchmark datasets, respectively. Briefly, our method outperformed existing methods in predicting 4mC sites in mouse genomes. Also, we deployed i4mC-GRU 
as an online web server, supporting users in genomics studies.

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
