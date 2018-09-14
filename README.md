# Kernel-Perturbation-based-Boosting-of-SVMs
The KPBoost-SVM method can be used to effectively perform boosting with SVMs. SVMs are stable learner which cannot be readily boosted by resampling/reweighting the data. Therefore, KPBoost-SVM uses Kernel Perturbation to boost SVMs. The idea is to increase, in each round of boosting, the resolution of the RBF-kernel-induced feature space around the data points which are misclassified in the previous round. The resolution is increased by using a conformal transformation. The KPBoostROI-SVM method is a variant of KPBoost-SVM which calculates the resolution around the individual test points, unlike KPBoost-SVM (which assumes maximum possible resolution around all test data points).

**This code bundle contains codes for IDENTIFYING the DISJUNCTS in a DATASET (findDisjunts.m) and also contains an implementation of the Geometric Small Disjunct Index for MEASURING the PERFORMANCE on the SMALL DISJUNCTS for ANY CLASSIFIER (GSDI.m).**

LICENSE:

1. This software is provided free of charge to the research community as an academic software package with no commitment in terms of support or maintenance.
2. Users interested in commercial applications should contact Shounak Datta (shounak.jaduniv@gmail.com) or Dr. Swagatam Das (swagatamdas19@yahoo.co.in). 
3. Any copy shall bear an appropriate copyright notice specifying the above-mentioned authors.
4. Licensee acknowledges that this software is a research tool, still in the development stage. Hence, it is not presented as error–free, accurate, complete, useful, suitable for any specific application or free from any infringement of any rights. The Software is licensed AS IS, entirely at the Licensee's own risk.
5. The Researchers and/or ISI Kolkata shall not be liable for any damage, claim, demand, cost or expense of whatsoever kind or nature directly or indirectly arising out of or resulting from or encountered in connection with the use of this software.

INSTRUCTIONS:

1. Load the required data (see 'sampleWorkspace.mat' for an example) in the MATLAB workspace.
2. Run 'KPBoostCode.m' with appropriate choice of parameters (see our article on arXiv for recommended settings: https://arxiv.org/abs/1712.08493).
3. To only identify the disjuncts in a dataset, use the function 'findDisjuncts.m'.
4. To only measure the performance on the small disjuncts for a given classification, use the function 'GSDI.m'.
