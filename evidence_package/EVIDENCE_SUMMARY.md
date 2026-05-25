# Cross-Market Evidence Package

The evidence package combines the no-training baselines, supervised neural models, compact modern time-series models, and Chronos-T5-small zero-shot audit.

## Main finding

Latest-close persistence remains a hard gate. Supervised neural models almost never beat it, and Chronos-T5-small zero-shot has more local wins but still has median RMSE ratio above 1.0.

## Full-origin Chronos subset verification

The manuscript also cites a 60-row full-origin Chronos subset verification. The corresponding row-level evidence is stored in `chronos_full_origin_subset_60_rows.csv`, with grouped summaries in `full_origin_subset_by_dataset.csv` and `full_origin_subset_by_horizon.csv`. This subset uses two assets per dataset, both price modes, all three horizons, and all available test origins rather than the sampled-origin protocol. It has 60 rows, 2 persistence-gate passes, and a median RMSE ratio of approximately 1.028, which is consistent with the main sampled-origin Chronos conclusion.

## Model summary

```
                     model  rows  passes  pass_rate  median_rmse_ratio  median_ci_low  median_ci_high  p_wilcoxon_ratio_gt_1
Chronos-T5-small zero-shot   990     133   0.134343           1.034709       1.032606        1.037472          5.513058e-126
                       GRU  5940       5   0.000842           2.221390       2.144060        2.315136           0.000000e+00
           TimeMixer-style  5940       3   0.000505           2.327998       2.287791        2.369617           0.000000e+00
                      LSTM  5940       8   0.001347           2.640621       2.510700        2.754118           0.000000e+00
                   DLinear  5940       0   0.000000           3.149812       3.016526        3.257681           0.000000e+00
        iTransformer-style  5940       4   0.000673           3.722963       3.611596        3.945537           0.000000e+00
            PatchTST-style  5940       3   0.000505           3.766382       3.644904        3.898635           0.000000e+00
```

## Dataset-model pass summary

```
      dataset_label                      model  rows  passes  pass_rate  median_rmse_ratio
             Dow 30 Chronos-T5-small zero-shot   180      21   0.116667           1.036235
             Dow 30                    DLinear  1080       0   0.000000           2.671105
             Dow 30                        GRU  1080       1   0.000926           2.390109
             Dow 30                       LSTM  1080       2   0.001852           2.793853
             Dow 30             PatchTST-style  1080       0   0.000000           4.078310
             Dow 30            TimeMixer-style  1080       0   0.000000           2.324484
             Dow 30         iTransformer-style  1080       0   0.000000           4.160895
         ETF basket Chronos-T5-small zero-shot    72      13   0.180556           1.031853
         ETF basket                    DLinear   432       0   0.000000           3.675586
         ETF basket                        GRU   432       1   0.002315           4.363805
         ETF basket                       LSTM   432       0   0.000000           5.233067
         ETF basket             PatchTST-style   432       0   0.000000           5.552849
         ETF basket            TimeMixer-style   432       0   0.000000           3.038687
         ETF basket         iTransformer-style   432       0   0.000000           6.223629
Hang Seng large-cap Chronos-T5-small zero-shot   258      40   0.155039           1.032956
Hang Seng large-cap                    DLinear  1548       0   0.000000           4.746876
Hang Seng large-cap                        GRU  1548       2   0.001292           1.408018
Hang Seng large-cap                       LSTM  1548       4   0.002584           1.638625
Hang Seng large-cap             PatchTST-style  1548       3   0.001938           2.084948
Hang Seng large-cap            TimeMixer-style  1548       3   0.001938           1.970080
Hang Seng large-cap         iTransformer-style  1548       4   0.002584           1.958612
  Nasdaq large-tech Chronos-T5-small zero-shot   180      19   0.105556           1.035055
  Nasdaq large-tech                    DLinear  1080       0   0.000000           2.291023
  Nasdaq large-tech                        GRU  1080       0   0.000000           3.995578
  Nasdaq large-tech                       LSTM  1080       0   0.000000           4.572106
  Nasdaq large-tech             PatchTST-style  1080       0   0.000000           5.108434
  Nasdaq large-tech            TimeMixer-style  1080       0   0.000000           2.403611
  Nasdaq large-tech         iTransformer-style  1080       0   0.000000           5.364039
     S&P 100 subset Chronos-T5-small zero-shot   300      40   0.133333           1.036146
     S&P 100 subset                    DLinear  1800       0   0.000000           3.000215
     S&P 100 subset                        GRU  1800       1   0.000556           2.412370
     S&P 100 subset                       LSTM  1800       2   0.001111           2.868273
     S&P 100 subset             PatchTST-style  1800       0   0.000000           4.189315
     S&P 100 subset            TimeMixer-style  1800       0   0.000000           2.358853
     S&P 100 subset         iTransformer-style  1800       0   0.000000           4.228747
```
