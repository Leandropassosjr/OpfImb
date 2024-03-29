## OpfImb

Code from:
["Passos, L. A., Jodas, D. S., Ribeiro, L. C., Akio, M., de Souza, A. N., & Papa, J. P. (2022). **Handling imbalanced datasets through Optimum-Path Forest.** Knowledge-Based Systems, 108445.](https://doi.org/10.1016/j.knosys.2022.108445)

["Passos, L. A., Jodas, D. S., Ribeiro, L. C., Moreira, T., & Papa, J. P. (2020). **O²PF: Oversampling via Optimum-Path Forest for Breast Cancer Detection.** IEEE 33rd International Symposium on Computer-Based Medical Systems (CBMS), 2020, pp. 498-503](https://ieeexplore.ieee.org/abstract/document/9183010)



OPFImb is an Optimum-Path Forest-based library created for handling the problem of imbalanced datasets. OPFImb was developed to be simple and easy-to-use for users who are familiarized with existing frameworks which tackle the same research issue that often represent a problem in the context of classification tasks. Our library is composed of functions that handle either oversampling or undersampling of imbalanced datasets by using a range of variants designed to specific aspects of the data distribution under analysis. 

For the oversampling procedure, synthetic samples are created using a Gaussian distribution computed through the mean value and the covariance of the samples within the clusters of the minority class samples generated using the Unsupervised Optimum-Path Forest (OPF) model. Regarding the undersampling, Supervised learning by OPF is employed to assign a score for each training sample that correct conquers an instance of the testing set. Training samples with zeros or negative scores are candidates to be removed from the training set.

The following methods are so far available in the OPFImb:

Overampling:
 - O2PF: it represents the standard Oversampling Optimum-Path Forest method;
 - O2PF_RI: O2PF Radius Interpolation;
 - O2PF_MI: O2PF Mean Interpolation;
 - O2PF_P: O2PF Prototype;
 - O2PF_WI: O2PF Weight Interpolation.

Undersampling:
 - OPF-US: represents the standard Undersampling Optimum-Path Forest method. Removes low-ranked samples from majority class until the dataset is balanced;
 - OPF-US1: removes samples from the majority class with negative scores;
 - OPF-US2: removes samples from the majority class with scores lower or equal to zero;
 - OPF-US3: removes all samples with negative scores.

Besides the above-mentioned methods, OPFImb provides three hybrid approaches that firstly apply an undersampling method followed by the oversampling performed by the standard O2PF. These hybrid methods are described as follows:
 - OPF-US1-O2PF: undersampling by using OPF-US1 followed by oversampling performed by O2PF;
 - OPF-US2-O2PF: undersampling by using OPF-US2 followed by oversampling performed by O2PF;
 - OPF-US3-O2PF: undersampling by using OPF-US3 followed by oversampling performed by O2PF.

Examples:
 - Overampling;
 	- python oversampling.py
 	- python oversampling_example_simple.py data/vertebral_column/1/train.txt 20
 - Undersampling;
 	- python undersampling.py
 	- python undersampling_example_simple.py data/vertebral_column/1/train.txt data/vertebral_column/1/valid.txt
 - Hybrid;
 	- python hybrid.py;
 	- python hybrid_example_simple.py data/vertebral_column/1/train.txt 20 data/vertebral_column/1/valid.txt

Results:
 - result_folder/OpfImb_variant_name/dataset_name/fold_number/pred.txt
	- testing set predicted labels over fold
 - result_folder/OpfImb_variant_name/dataset_name/fold_number/results.txt
	- testing set results over fold. Follows the pattern: best_k,accuracy,recall,f1,execution_time
 - result_folder/OpfImb_variant_name/dataset_name/fold_number/validation.txt
	- validation set results over fold, where each line denotes a evaluated k_max. Follows the pattern: best_k,accuracy,recall,f1,execution_time

OPF code inspired in the OPFYTHON lib:
 - https://github.com/gugarosa/opfython

Paper describing the code:
 - Handling Imbalanced Datasets Through Optimum-Path Forest (https://doi.org/10.1016/j.knosys.2022.108445).
