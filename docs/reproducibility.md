# Faucet
## Eval Faucet No Pre-train
```
python evaluate_policy.py --task faucet --checkpoint_path /PATH/TO/CHECKPOINT/faucet/faucet_nopretrain_0.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task faucet --checkpoint_path /PATH/TO/CHECKPOINT/faucet/faucet_nopretrain_1.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task faucet --checkpoint_path /PATH/TO/CHECKPOINT/faucet/faucet_nopretrain_2.zip --eval_per_instance 100 --seed 100 --use_test_set
```
| split | Seed 0              | Seed 1                | Seed 2              | Avg                    | Std                 |
|-------|---------------------|-----------------------|---------------------|------------------------|---------------------|
| train | 0.5218181818181818  | 0.0                   | 0.44                | 0.3206060606060606     | 0.22915022480461883 |
| test  | 0.3414285714285714  | 0.0014285714285714286 | 0.42714285714285716 | 0.25666666666666665    | 0.1838415960176147  |

## Eval Faucet Segmentation on PMM
```
python evaluate_policy.py --task faucet --checkpoint_path /PATH/TO/CHECKPOINT/faucet/faucet_seg_pmm_0.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task faucet --checkpoint_path /PATH/TO/CHECKPOINT/faucet/faucet_seg_pmm_1.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task faucet --checkpoint_path /PATH/TO/CHECKPOINT/faucet/faucet_seg_pmm_2.zip --eval_per_instance 100 --seed 100 --use_test_set
```
| split | Seed 0             | Seed 1              | Seed 2               | Avg                    | Std                   |
|-------|--------------------|---------------------|----------------------|------------------------|-----------------------|
| train | 0.42               | 0.2545454545454545  | 0.14181818181818182  | 0.27212121212121215    | 0.11424523748646122   |
| test  | 0.3757142857142857 | 0.1457142857142857  | 0.11428571428571428  | 0.21190476190476193    | 0.11653928906463953   |

## Eval Faucet Classification on PMM
```
python evaluate_policy.py --task faucet --checkpoint_path /PATH/TO/CHECKPOINT/faucet/faucet_cls_pmm_0.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task faucet --checkpoint_path /PATH/TO/CHECKPOINT/faucet/faucet_cls_pmm_1.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task faucet --checkpoint_path /PATH/TO/CHECKPOINT/faucet/faucet_cls_pmm_2.zip --eval_per_instance 100 --seed 100 --use_test_set
```
| split | Seed 0              | Seed 1               | Seed 2               | Avg                  | Std                  |
|-------|---------------------|----------------------|----------------------|----------------------|----------------------|
| train | 0.3990909090909091  | 0.18727272727272729  | 0.07272727272727272  | 0.21969696969696972  | 0.13519567154736256  |
| test  | 0.32857142857142857 | 0.13714285714285715  | 0.08857142857142856  | 0.18476190476190477  | 0.10360399050264696  |

## Eval Faucet Reconstruction on DAM
```
python evaluate_policy.py --task faucet --checkpoint_path /PATH/TO/CHECKPOINT/faucet/faucet_recon_dam_0.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task faucet --checkpoint_path /PATH/TO/CHECKPOINT/faucet/faucet_recon_dam_1.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task faucet --checkpoint_path /PATH/TO/CHECKPOINT/faucet/faucet_recon_dam_2.zip --eval_per_instance 100 --seed 100 --use_test_set
```
| split | Seed 0              | Seed 1              | Seed 2              | Avg                 | Std                  |
|-------|---------------------|---------------------|---------------------|---------------------|----------------------|
| train | 0.26545454545454544 | 0.3663636363636364  | 0.3618181818181818  | 0.33121212121212124 | 0.04653464205221157  |
| test  | 0.17                | 0.2985714285714286  | 0.2057142857142857  | 0.2247619047619048  | 0.05418955560352879  |

## Eval Faucet SimSiam on DAM
```
python evaluate_policy.py --task faucet --checkpoint_path /PATH/TO/CHECKPOINT/faucet/faucet_sim_dam_0.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task faucet --checkpoint_path /PATH/TO/CHECKPOINT/faucet/faucet_sim_dam_1.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task faucet --checkpoint_path /PATH/TO/CHECKPOINT/faucet/faucet_sim_dam_2.zip --eval_per_instance 100 --seed 100 --use_test_set
```
| split | Seed 0              | Seed 1              | Seed 2              | Avg                 | Std                  |
|-------|---------------------|---------------------|---------------------|---------------------|----------------------|
| train | 0.7990909090909091  | 0.4                 | 0.7272727272727273  | 0.6421212121212122  | 0.17369796358276837  |
| test  | 0.6042857142857143  | 0.24142857142857144 | 0.5285714285714286  | 0.4580952380952381  | 0.15629352001632368  |

## Eval Faucet Segmentation on DAM
```
python evaluate_policy.py --task faucet --checkpoint_path /PATH/TO/CHECKPOINT/faucet/faucet_seg_dam_0.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task faucet --checkpoint_path /PATH/TO/CHECKPOINT/faucet/faucet_seg_dam_1.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task faucet --checkpoint_path /PATH/TO/CHECKPOINT/faucet/faucet_seg_dam_2.zip --eval_per_instance 100 --seed 100 --use_test_set
```
| split | Seed 0              | Seed 1              | Seed 2              | Avg                | Std                  |
|-------|---------------------|---------------------|---------------------|--------------------|----------------------|
| train | 0.8027272727272727  | 0.7572727272727273  | 0.8218181818181818  | 0.793939393939394  | 0.0270733452656873   |
| test  | 0.5685714285714286  | 0.5328571428571428  | 0.6614285714285715  | 0.5876190476190476 | 0.054189555603528804 |

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


# Toilet
## Eval Toilet No Pre-train
```
python evaluate_policy.py --task toilet --checkpoint_path /PATH/TO/CHECKPOINT/toilet/toilet_nopretrain_0.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task toilet --checkpoint_path /PATH/TO/CHECKPOINT/toilet/toilet_nopretrain_1.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task toilet --checkpoint_path /PATH/TO/CHECKPOINT/toilet/toilet_nopretrain_2.zip --eval_per_instance 100 --seed 100 --use_test_set
```
| split | Seed 0              | Seed 1              | Seed 2               | Avg                 | Std                  |
|-------|---------------------|---------------------|----------------------|---------------------|----------------------|
| train | 0.798235294117647   | 0.7452941176470588  | 0.6311764705882353   | 0.7249019607843138  | 0.06970912267364476  |
| test  | 0.47454545454545455 | 0.5054545454545455  | 0.43272727272727274  | 0.4709090909090909  | 0.029801917219743637 |

## Eval Toilet Segmentation on PMM
```
python evaluate_policy.py --task toilet --checkpoint_path /PATH/TO/CHECKPOINT/toilet/toilet_seg_pmm_0.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task toilet --checkpoint_path /PATH/TO/CHECKPOINT/toilet/toilet_seg_pmm_1.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task toilet --checkpoint_path /PATH/TO/CHECKPOINT/toilet/toilet_seg_pmm_2.zip --eval_per_instance 100 --seed 100 --use_test_set
```
| split | Seed 0              | Seed 1              | Seed 2              | Avg                 | Std                   |
|-------|---------------------|---------------------|---------------------|---------------------|-----------------------|
| train | 0.78                | 0.6182352941176471  | 0.64                | 0.6794117647058825  | 0.07167947365517743   |
| test  | 0.41545454545454547 | 0.45545454545454545 | 0.47363636363636363 | 0.4481818181818182  | 0.024302954734258687  |

## Eval Toilet Classification on PMM
```
python evaluate_policy.py --task toilet --checkpoint_path /PATH/TO/CHECKPOINT/bucket/toilet_cls_pmm_0.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task toilet --checkpoint_path /PATH/TO/CHECKPOINT/bucket/toilet_cls_pmm_1.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task toilet --checkpoint_path /PATH/TO/CHECKPOINT/bucket/toilet_cls_pmm_2.zip --eval_per_instance 100 --seed 100 --use_test_set
```
| split | Seed 0              | Seed 1              | Seed 2             | Avg                 | Std                   |
|-------|---------------------|---------------------|--------------------|---------------------|-----------------------|
| train | 0.7811764705882352  | 0.6458823529411765  | 0.6558823529411765 | 0.6943137254901961  | 0.06155676168911205   |
| test  | 0.33090909090909093 | 0.4263636363636364  | 0.44               | 0.3990909090909091  | 0.048532173872869594  |

## Eval Toilet Reconstruction on DAM
```
python evaluate_policy.py --task toilet --checkpoint_path /PATH/TO/CHECKPOINT/toilet/toilet_recon_dam_0.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task toilet --checkpoint_path /PATH/TO/CHECKPOINT/toilet/toilet_recon_dam_1.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task toilet --checkpoint_path /PATH/TO/CHECKPOINT/toilet/toilet_recon_dam_2.zip --eval_per_instance 100 --seed 100 --use_test_set
```
| split | Seed 0              | Seed 1              | Seed 2              | Avg                  | Std                  |
|-------|---------------------|---------------------|---------------------|----------------------|----------------------|
| train | 0.7817647058823529  | 0.7288235294117648  | 0.7511764705882353  | 0.753921568627451    | 0.02170013385452063  |
| test  | 0.5827272727272728  | 0.47545454545454546 | 0.4890909090909091  | 0.5157575757575757   | 0.04768083358797024  |

## Eval Toilet SimSiam on DAM
```
python evaluate_policy.py --task toilet --checkpoint_path /PATH/TO/CHECKPOINT/toilet/toilet_sim_dam_0.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task toilet --checkpoint_path /PATH/TO/CHECKPOINT/toilet/toilet_sim_dam_1.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task toilet --checkpoint_path /PATH/TO/CHECKPOINT/toilet/toilet_sim_dam_2.zip --eval_per_instance 100 --seed 100 --use_test_set
```
| split | Seed 0              | Seed 1              | Seed 2              | Avg                 | Std                 |
|-------|---------------------|---------------------|---------------------|---------------------|---------------------|
| train | 0.8405882352941176  | 0.8076470588235294  | 0.8441176470588235  | 0.8307843137254901  | 0.0164238365422525  |
| test  | 0.5436363636363636  | 0.4909090909090909  | 0.45181818181818184 | 0.4954545454545454  | 0.03762216098584951 |

## Eval Toilet Segmentation on DAM
```
python evaluate_policy.py --task toilet --checkpoint_path /PATH/TO/CHECKPOINT/toilet/toilet_seg_dam_0.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task toilet --checkpoint_path /PATH/TO/CHECKPOINT/toilet/toilet_seg_dam_1.zip --eval_per_instance 100 --seed 100 --use_test_set
python evaluate_policy.py --task toilet --checkpoint_path /PATH/TO/CHECKPOINT/toilet/toilet_seg_dam_2.zip --eval_per_instance 100 --seed 100 --use_test_set
```
| split | Seed 0              | Seed 1              | Seed 2              | Avg                  | Std                   |
|-------|---------------------|---------------------|---------------------|----------------------|-----------------------|
| train | 0.8635294117647059  | 0.8388235294117647  | 0.8588235294117647  | 0.8537254901960784   | 0.01071098061470402   |
| test  | 0.5445454545454546  | 0.5272727272727272  | 0.5627272727272727  | 0.5448484848484848   | 0.014475843530298432  |

