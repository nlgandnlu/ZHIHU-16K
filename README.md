### ZHIHU-16K Dataset

#### Overview
The ZHIHU-16K dataset comprises 16,381 articles collected from 13 diverse topics on the ZHIHU platform, with labels provided by ZHIHU. It includes questions, answers, author details, and comments. Among these articles, 2,526 are advertorials and 13,855 are normal articles. The dataset is split into training, validation, and test sets, with detailed statistics provided in the dataset_information.xlsx.

#### Data preprocessing

unzip data.zip files (https://drive.google.com/file/d/10GyQd8At0uuQ_e6fcS8zaQsyDIosbaTs/view?usp=drive_link) first, and move these files to corresponding folders (We have encrypted the sensitive information in these files). 
Move constructor_graph.py to the same directory in the entire data folder and:

python construct_graph.py

The features required by the gnn module are processed and saved in edge/ (structural information) and feature/ (node ​​feature information)

#### The data for robustness testing is in robu_test.zip

