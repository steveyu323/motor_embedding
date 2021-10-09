# motor_embedding
Fall 2020 Thomson Lab Rotation ProjectCancel changes

## Folder Structure of /code folder



#### ./code/data_process

Include the code for getting all motor proteins from pFAM domains and corresponding full-protein sequences from Uniprot, with annotation from a variety of databases

#### ./code/first_try

Include the code using bidirectional LSTM for embedding motor proteins and classifying the motor proteins into different classes of motor

#### ./code/evo_tune_201023

Code for training first on the Uniprot database sequence then evotune specifically on motor proteins

#### ./code/tape

a abandoned try of using tape package from Yun Song lab, UC Berkeley

#### ./code/kif

code for using ESM pretrained transformer model to learn kinesin superfamilies instead of the entire motor protein set

#### ./code/thermo

code for using ESM pretrained transformer model to learn thermo properties of motor proteins.

Since we stick to ESM pre-trained weights, which performed the best as it was trained on the entire Uniprot with much larger computational power than the training we performed on Uniprot, ./code/data_process, ./code/kif, ./code/thermo are the most relevant to what is present and finalized in the report. 



### Data Accession





