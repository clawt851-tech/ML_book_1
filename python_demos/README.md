# Python Demos for "Advances in Financial Machine Learning"

Each file is a runnable, self-contained demonstration of the key snippets and concepts of one chapter.

## Dependencies

```bash
pip install numpy pandas scipy scikit-learn matplotlib statsmodels
```

Optional (only specific chapters):
- `mpi4py` (Ch 22 only — falls back to serial)

## File List

| File | Chapter | Key Snippets |
|------|---------|--------------|
| `chapter_01_financial_ml_intro.py` | 1 | Sisyphus vs meta-strategy, common pitfalls |
| `chapter_02_financial_data_structures.py` | 2 | Time/Tick/Volume/Dollar bars, TIBs, PCA weights, CUSUM filter |
| `chapter_03_labeling.py` | 3 | Daily volatility, triple-barrier, getEvents/getBins, meta-labeling |
| `chapter_04_sample_weights.py` | 4 | Concurrent labels, average uniqueness, sequential bootstrap, time decay |
| `chapter_05_fractional_differentiation.py` | 5 | get_weights, frac_diff, FFD, ADF-based d* search |
| `chapter_06_ensemble_methods.py` | 6 | Bagging variance formula, Condorcet accuracy, RF setup variants |
| `chapter_07_cross_validation.py` | 7 | getTrainTimes (purging), getEmbargoTimes, PurgedKFold class |
| `chapter_08_feature_importance.py` | 8 | MDI, MDA (permutation), SFI, orthogonal features |
| `chapter_09_hyperparameter_tuning.py` | 9 | MyPipeline, log-uniform RV, GridSearchCV with PurgedKFold |
| `chapter_10_bet_sizing.py` | 10 | getSignal, avgActiveSignals, discreteSignal, sigmoid sizing |
| `chapter_11_dangers_of_backtesting.py` | 11 | Seven sins, CSCV / Probability of Backtest Overfitting |
| `chapter_12_backtesting_via_cv.py` | 12 | Walk-forward, CombinatorialPurgedKFold, multi-path backtest |
| `chapter_13_synthetic_data_backtesting.py` | 13 | O-U Monte Carlo, optimal trading rule (pt, sl) heatmap |
| `chapter_14_backtest_statistics.py` | 14 | Bet timing, holding period, HHI, DD/TuW, PSR, DSR |
| `chapter_15_strategy_risk.py` | 15 | Symmetric/asymmetric SR, required precision, P[failure] |
| `chapter_16_ml_asset_allocation.py` | 16 | HRP: correlDist, getQuasiDiag, getRecBipart |
| `chapter_17_structural_breaks.py` | 17 | SADF inner loop, CS-White CUSUM, SMT polynomial |
| `chapter_18_entropy_features.py` | 18 | Plug-in / LZ / Kontoyiannis entropy, encoding schemes |
| `chapter_19_microstructural_features.py` | 19 | Tick rule, Roll, Parkinson, Corwin-Schultz, Kyle/Amihud/Hasbrouck λ, VPIN |
| `chapter_20_multiprocessing.py` | 20 | linParts, nestedParts, mpPandasObj |
| `chapter_21_brute_force_quantum.py` | 21 | Pigeonhole partitions, brute-force portfolio |
| `chapter_22_hpc_intelligence.py` | 22 | VPIN calibration, NUFFT, streaming moments, MPI stub |

## Run Any Demo

```bash
python python_demos/chapter_05_fractional_differentiation.py
```

Each `main()` prints results to stdout — no plotting required (book figures omitted to keep demos lean).

## Limitations

These are **didactic** implementations:

- Synthetic data only (no real market feeds).
- Some snippets compress book pseudocode for clarity.
- The book's `mpEngine` parallel infrastructure is approximated by single-thread loops; for production parallelism see Ch 20.
- For production-grade code, look at [`mlfinlab`](https://github.com/hudson-and-thames/mlfinlab) (open-source community implementation).

## Citation

```
López de Prado, Marcos (2018). Advances in Financial Machine Learning.
Wiley. ISBN 978-1-119-48208-6.
```
