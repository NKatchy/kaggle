import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. データの読み込み
# ※手元で実行する場合はパスを書き換えてください
try:
    train = pd.read_csv("/kaggle/input/playground-series-s6e1/train.csv")
    print(f"Train shape: {train.shape}")
except:
    print("CSVファイルが見つかりません。パスを確認してください。")

# 2. グラフ作成用の関数（EDAツール）
def plot_features_dual_axis(
    df, cols, target_col, n_bins=10, n_wide=3, 
    figsize_per_plot=(5, 4), int_as_cat_unique_max=20, cat_order=None
):
    BAR_COLOR = "tab:blue"
    LINE_COLOR = "tab:orange"

    def is_categorical(s):
        return s.dtype == "object" or pd.api.types.is_string_dtype(s)

    n_cols = len(cols)
    n_rows = int(np.ceil(n_cols / n_wide))

    fig, axes = plt.subplots(
        n_rows, n_wide, 
        figsize=(figsize_per_plot[0] * n_wide, figsize_per_plot[1] * n_rows)
    )
    axes = np.atleast_1d(axes).ravel()

    for i, col in enumerate(cols):
        ax1 = axes[i]
        n_nan = df[col].isna().sum()
        n_unique = df[col].nunique(dropna=True)

        tmp = df[[col, target_col]].dropna()
        if tmp.empty:
            ax1.set_title(f"{col}\n(empty; {n_unique} unique)")
            continue

        x = tmp[col]
        y = tmp[target_col]

        # --- カテゴリ変数の処理 ---
        if is_categorical(x):
            counts = x.value_counts()
            mean_y = tmp.groupby(col)[target_col].mean()

            if cat_order is not None and col in cat_order:
                desired = list(cat_order[col])
                ordered = [c for c in desired if c in counts.index]
                remaining = [c for c in counts.index if c not in ordered]
                final_order = ordered + remaining
            else:
                final_order = sorted(counts.index)

            counts = counts.loc[final_order]
            mean_y = mean_y.loc[final_order]
            xpos = np.arange(len(final_order))

            ax1.bar(xpos, counts.values, alpha=0.6, color=BAR_COLOR)
            ax1.set_xticks(xpos)
            ax1.set_xticklabels(final_order, rotation=45, ha="right")

            ax2 = ax1.twinx()
            ax2.plot(xpos, mean_y.values, marker="o", color=LINE_COLOR)
            
        # --- 数値変数の処理 ---
        else:
            if int_as_cat_unique_max is not None and n_unique <= int_as_cat_unique_max:
                # 数値だが種類が少ない場合（Ageなど）はカテゴリ風に表示
                unique_vals = np.sort(x.unique())
                counts = x.value_counts().sort_index()
                mean_y = tmp.groupby(col)[target_col].mean().sort_index()
                xpos = np.arange(len(unique_vals))
                
                ax1.bar(xpos, counts.values, alpha=0.6, color=BAR_COLOR)
                ax1.set_xticks(xpos)
                ax1.set_xticklabels(unique_vals)
                
                ax2 = ax1.twinx()
                ax2.plot(xpos, mean_y.values, marker="o", color=LINE_COLOR)
            else:
                # 通常の数値（ヒストグラム）
                bins = np.linspace(x.min(), x.max(), n_bins + 1)
                bin_centers = 0.5 * (bins[:-1] + bins[1:])
                counts, _ = np.histogram(x, bins=bins)
                bin_idx = np.digitize(x, bins) - 1
                mean_y = [tmp[bin_idx == j][target_col].mean() for j in range(n_bins)]

                ax1.bar(bin_centers, counts, width=(bins[1]-bins[0]), alpha=0.6, color=BAR_COLOR)
                ax2 = ax1.twinx()
                ax2.plot(bin_centers, mean_y, marker="o", color=LINE_COLOR)

        ax1.set_title(f"{col}\n({n_unique} unique, {n_nan} nan)")
        ax1.set_ylabel("Count", color=BAR_COLOR)
        ax2.set_ylabel(f"Mean {target_col}", color=LINE_COLOR)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

# 3. 実行
if 'train' in locals():
    FEATURES = list(train.columns[1:-1])
    ordinal_order = {
        "sleep_quality": ["poor", "average", "good"],
        "facility_rating": ["low", "medium", "high"],
        "exam_difficulty": ["easy", "moderate", "hard"],
    }

    plot_features_dual_axis(
        train,
        cols=FEATURES,
        target_col="exam_score",
        cat_order=ordinal_order
    )
