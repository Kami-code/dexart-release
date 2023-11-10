
# Bucket
## Eval Bucket No Pre-train
```
python evaluate_policy.py --task bucket --checkpoint_path /PATH/TO/CHECKPOINT/bucket/bucket_nopretrain_0.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task bucket --checkpoint_path /PATH/TO/CHECKPOINT/bucket/bucket_nopretrain_1.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task bucket --checkpoint_path /PATH/TO/CHECKPOINT/bucket/bucket_nopretrain_2.zip --eval_per_instance 100 --seed 100 --use_test_set
```
| split | Seed 0 | Seed 1 | Seed 2 | Avg                  | Std                  |
|-------|--------|--------|--------|----------------------|----------------------|
| train | 0.36   | 0.58   | 0.52   | 0.48666666666666664  | 0.09285592184789412  |
| test  | 0.545  | 0.6875 | 0.49   | 0.5741666666666666   | 0.08322492949164265  |

## Eval Bucket Segmentation on PMM
```
python evaluate_policy.py --task bucket --checkpoint_path /PATH/TO/CHECKPOINT/bucket/bucket_seg_pmm_0.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task bucket --checkpoint_path /PATH/TO/CHECKPOINT/bucket/bucket_seg_pmm_1.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task bucket --checkpoint_path /PATH/TO/CHECKPOINT/bucket/bucket_seg_pmm_2.zip --eval_per_instance 100 --seed 100 --use_test_set
```
| split | Seed 0             | Seed 1                | Seed 2             | Avg                  | Std                 |
|-------|--------------------|-----------------------|--------------------|----------------------|---------------------|
| train | 0.6190909090909091 | 0.0009090909090909091 | 0.4036363636363636 | 0.34121212121212124  | 0.25620275774514356 |
| test  | 0.6175             | 0.00125               | 0.41125            | 0.3433333333333333   | 0.25612564733392523 |

## Eval Bucket Classification on PMM
```
python evaluate_policy.py --task bucket --checkpoint_path /PATH/TO/CHECKPOINT/bucket/bucket_cls_pmm_0.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task bucket --checkpoint_path /PATH/TO/CHECKPOINT/bucket/bucket_cls_pmm_1.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task bucket --checkpoint_path /PATH/TO/CHECKPOINT/bucket/bucket_cls_pmm_2.zip --eval_per_instance 100 --seed 100 --use_test_set
```
| split | Seed 0             | Seed 1              | Seed 2             | Avg                | Std                 |
|-------|--------------------|---------------------|--------------------|--------------------|---------------------|
| train | 0.5545454545454546 | 0.49727272727272726 | 0.6672727272727272 | 0.573030303030303  | 0.07062231572541762 |
| test  | 0.46625            | 0.50625             | 0.725              | 0.5658333333333333 | 0.11372634064083639 |

## Eval Bucket Reconstruction on DAM
```
python evaluate_policy.py --task bucket --checkpoint_path /PATH/TO/CHECKPOINT/bucket/bucket_recon_dam_0.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task bucket --checkpoint_path /PATH/TO/CHECKPOINT/bucket/bucket_recon_dam_1.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task bucket --checkpoint_path /PATH/TO/CHECKPOINT/bucket/bucket_recon_dam_2.zip --eval_per_instance 100 --seed 100 --use_test_set
```
| split | Seed 0              | Seed 1             | Seed 2             | Avg                 | Std                 |
|-------|---------------------|--------------------|--------------------|---------------------|---------------------|
| train | 0.48727272727272725 | 0.5754545454545454 | 0.4009090909090909 | 0.48787878787878786 | 0.07125917207730058 |
| test  | 0.48625             | 0.46125            | 0.5875             | 0.5116666666666667  | 0.05458492364095501 |

## Eval Bucket SimSiam on DAM
```
python evaluate_policy.py --task bucket --checkpoint_path /PATH/TO/CHECKPOINT/bucket/bucket_sim_dam_0.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task bucket --checkpoint_path /PATH/TO/CHECKPOINT/bucket/bucket_sim_dam_1.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task bucket --checkpoint_path /PATH/TO/CHECKPOINT/bucket/bucket_sim_dam_2.zip --eval_per_instance 100 --seed 100 --use_test_set
```
| split | Seed 0                | Seed 1             | Seed 2             | Avg                  | Std                  |
|-------|-----------------------|--------------------|--------------------|----------------------|----------------------|
| train | 0.0009090909090909091 | 0.5290909090909091 | 0.7290909090909091 | 0.4196969696969697   | 0.3071779783390928   |
| test  | 0.0                   | 0.3825             | 0.78125            | 0.3879166666666667   | 0.31896697408282815  |

## Eval Bucket Segmentation on DAM
```
python evaluate_policy.py --task bucket --checkpoint_path /PATH/TO/CHECKPOINT/bucket/bucket_seg_dam_0.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task bucket --checkpoint_path /PATH/TO/CHECKPOINT/bucket/bucket_seg_dam_1.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task bucket --checkpoint_path /PATH/TO/CHECKPOINT/bucket/bucket_seg_dam_2.zip --eval_per_instance 100 --seed 100 --use_test_set
```
| split | Seed 0             | Seed 1   | Seed 2             | Avg                 | Std                   |
|-------|--------------------|----------|--------------------|---------------------|-----------------------|
| train | 0.7018181818181818 | 0.7      | 0.7927272727272727 | 0.7315151515151515  | 0.04328987012952383   |
| test  | 0.6775             | 0.735    | 0.8475             | 0.7533333333333334  | 0.07060256526658379   |


