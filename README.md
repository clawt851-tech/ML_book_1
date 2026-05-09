# Advances in Financial Machine Learning - Study Materials

Based on the book by **Marcos López de Prado** (Wiley, 2018).

This directory contains:

- **`AiFML.pdf`** — the original textbook (395 pages, 22 chapters)
- **`python_demos/`** — Self-contained Python demonstration scripts, one per chapter
- **`chinese_notes/`** — Chinese-language detailed notes, one Markdown file per chapter

## Chapter Index

| Ch | Title | Python Demo | Chinese Notes |
|----|-------|-------------|---------------|
| 1 | Financial ML as a Distinct Subject | [ch01](python_demos/chapter_01_financial_ml_intro.py) | [第01章](chinese_notes/第01章_金融机器学习独立学科.md) |
| 2 | Financial Data Structures | [ch02](python_demos/chapter_02_financial_data_structures.py) | [第02章](chinese_notes/第02章_金融数据结构.md) |
| 3 | Labeling | [ch03](python_demos/chapter_03_labeling.py) | [第03章](chinese_notes/第03章_标注.md) |
| 4 | Sample Weights | [ch04](python_demos/chapter_04_sample_weights.py) | [第04章](chinese_notes/第04章_样本权重.md) |
| 5 | Fractionally Differentiated Features | [ch05](python_demos/chapter_05_fractional_differentiation.py) | [第05章](chinese_notes/第05章_分数差分特征.md) |
| 6 | Ensemble Methods | [ch06](python_demos/chapter_06_ensemble_methods.py) | [第06章](chinese_notes/第06章_集成方法.md) |
| 7 | Cross-Validation in Finance | [ch07](python_demos/chapter_07_cross_validation.py) | [第07章](chinese_notes/第07章_金融中的交叉验证.md) |
| 8 | Feature Importance | [ch08](python_demos/chapter_08_feature_importance.py) | [第08章](chinese_notes/第08章_特征重要性.md) |
| 9 | Hyper-Parameter Tuning | [ch09](python_demos/chapter_09_hyperparameter_tuning.py) | [第09章](chinese_notes/第09章_交叉验证调超参.md) |
| 10 | Bet Sizing | [ch10](python_demos/chapter_10_bet_sizing.py) | [第10章](chinese_notes/第10章_下注大小.md) |
| 11 | The Dangers of Backtesting | [ch11](python_demos/chapter_11_dangers_of_backtesting.py) | [第11章](chinese_notes/第11章_回测的危险.md) |
| 12 | Backtesting through Cross-Validation | [ch12](python_demos/chapter_12_backtesting_via_cv.py) | [第12章](chinese_notes/第12章_交叉验证回测.md) |
| 13 | Backtesting on Synthetic Data | [ch13](python_demos/chapter_13_synthetic_data_backtesting.py) | [第13章](chinese_notes/第13章_合成数据回测.md) |
| 14 | Backtest Statistics | [ch14](python_demos/chapter_14_backtest_statistics.py) | [第14章](chinese_notes/第14章_回测统计指标.md) |
| 15 | Understanding Strategy Risk | [ch15](python_demos/chapter_15_strategy_risk.py) | [第15章](chinese_notes/第15章_理解策略风险.md) |
| 16 | Machine Learning Asset Allocation | [ch16](python_demos/chapter_16_ml_asset_allocation.py) | [第16章](chinese_notes/第16章_机器学习资产配置.md) |
| 17 | Structural Breaks | [ch17](python_demos/chapter_17_structural_breaks.py) | [第17章](chinese_notes/第17章_结构性突变.md) |
| 18 | Entropy Features | [ch18](python_demos/chapter_18_entropy_features.py) | [第18章](chinese_notes/第18章_熵特征.md) |
| 19 | Microstructural Features | [ch19](python_demos/chapter_19_microstructural_features.py) | [第19章](chinese_notes/第19章_市场微结构特征.md) |
| 20 | Multiprocessing and Vectorization | [ch20](python_demos/chapter_20_multiprocessing.py) | [第20章](chinese_notes/第20章_多进程与向量化.md) |
| 21 | Brute Force and Quantum Computers | [ch21](python_demos/chapter_21_brute_force_quantum.py) | [第21章](chinese_notes/第21章_暴力搜索与量子计算.md) |
| 22 | High-Performance Computational Intelligence | [ch22](python_demos/chapter_22_hpc_intelligence.py) | [第22章](chinese_notes/第22章_HPC计算智能与预测.md) |

## Book Structure

The book is organized into 5 parts:

- **Preamble** (Ch 1) — Why financial ML is distinct
- **Part 1: Data Analysis** (Ch 2-5) — Bars, labeling, sample weights, fracdiff
- **Part 2: Modelling** (Ch 6-9) — Ensembles, purged CV, feature importance, tuning
- **Part 3: Backtesting** (Ch 10-16) — Bet sizing, dangers, CPCV, synthetic data, statistics, risk, HRP
- **Part 4: Useful Financial Features** (Ch 17-19) — Structural breaks, entropy, microstructure
- **Part 5: HPC Recipes** (Ch 20-22) — Multiprocessing, quantum, national-lab perspective

## Three Laws of Backtesting

> **First Law:** "Backtesting is not a research tool. Feature importance is."
>
> **Second Law:** "Backtesting while researching is like drinking and driving. Do not research under the influence of a backtest."
>
> **Third Law:** "Every backtest result must be reported in conjunction with all the trials involved in its production."

## Running the Demos

Most demos require `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `statsmodels`. Install:

```bash
pip install numpy pandas scipy scikit-learn matplotlib statsmodels
```

Then run any chapter directly:

```bash
python python_demos/chapter_03_labeling.py
```

## Notes

- The demos are **simplified, self-contained reproductions** of the book's snippets — they prioritize clarity over production-grade implementation.
- For production use, see the open-source [`mlfinlab`](https://github.com/hudson-and-thames/mlfinlab) library.
- Mathematical notation in the Chinese notes follows the book's conventions.
