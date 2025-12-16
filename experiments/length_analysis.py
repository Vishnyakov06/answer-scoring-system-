import os
import sys
from pathlib import Path
os.environ["WANDB_DISABLED"] = "true"
sys.path.append(str(Path(__file__).parent.parent))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import json
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

from src.datasets import SemanticSimilarityDataset
from src.utils import read_csv

def comprehensive_length_analysis(model_path, data_path, output_dir="./comprehensive_length_analysis"):
    print("=" * 70)
    print("КОМПЛЕКСНЫЙ АНАЛИЗ: ДЛИНА ОТВЕТА vs ОЦЕНКА")
    print("=" * 70)

    test_df = read_csv(data_path).dropna(
        subset=["Вопрос", "Эталонный ответ преподавателя", "Ответ студента", "Оценка"]
    )

    label_encoder = LabelEncoder()
    test_df["label"] = label_encoder.fit_transform(test_df["Оценка"].astype(str))

    print(f"Тестовые данные: {len(test_df)} примеров")
    print(test_df["Оценка"].value_counts().sort_index())

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    test_df = test_df.copy()
    test_df["длина_ответа"] = test_df["Ответ студента"].apply(len)
    test_df["длина_эталона"] = test_df["Эталонный ответ преподавателя"].apply(len)
    test_df["отношение_длин"] = test_df["длина_ответа"] / test_df["длина_эталона"]
    test_df["разность_длин"] = test_df["длина_ответа"] - test_df["длина_эталона"]

    test_dataset = SemanticSimilarityDataset(test_df, tokenizer)
    tester = Trainer(model=model)
    test_preds = tester.predict(test_dataset)
    test_df["предсказанная_оценка"] = test_preds.predictions.argmax(-1)
    test_df["предсказанная_оценка_текст"] = label_encoder.inverse_transform(test_df["предсказанная_оценка"])
    test_df["правильно"] = test_df["label"] == test_df["предсказанная_оценка"]

    probabilities = torch.softmax(torch.tensor(test_preds.predictions), dim=1)
    test_df["уверенность"] = torch.max(probabilities, dim=1)[0].numpy()

    length_bins = [0, 50, 100, 200, 300, 500, 1000, float('inf')]
    length_labels = ['0-50', '50-100', '100-200', '200-300', '300-500', '500-1000', '>1000']
    test_df["диапазон_длины"] = pd.cut(test_df["длина_ответа"], bins=length_bins, labels=length_labels)

    print("\nРаспределение истинных оценок по диапазонам длин:")
    distribution_table = pd.crosstab(test_df["диапазон_длины"], test_df["Оценка"], normalize='index') * 100
    print(distribution_table.round(1))

    print("\nКоличество примеров по диапазонам длин:")
    count_table = pd.crosstab(test_df["диапазон_длины"], test_df["Оценка"])
    print(count_table)

    accuracy_by_length = test_df.groupby("диапазон_длины")["правильно"].mean()
    print("\nТочность модели по диапазонам длин:")
    for length_range, acc in accuracy_by_length.items():
        count = len(test_df[test_df["диапазон_длины"] == length_range])
        print(f"  {length_range}: {acc:.3f} (n={count})")

    relation_bins = [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, float('inf')]
    relation_labels = ['0-10%', '10-25%', '25-50%', '50-75%', '75-100%', '100-150%', '150-200%', '>200%']
    test_df["категория_отношения"] = pd.cut(test_df["отношение_длин"], bins=relation_bins, labels=relation_labels)

    print("\nРаспределение оценок по отношению длин:")
    relation_distribution = pd.crosstab(test_df["категория_отношения"], test_df["Оценка"], normalize='index') * 100
    print(relation_distribution.round(1))

    print("\nТочность модели по отношению длин:")
    accuracy_by_relation = test_df.groupby("категория_отношения")["правильно"].mean()
    for relation, acc in accuracy_by_relation.items():
        count = len(test_df[test_df["категория_отношения"] == relation])
        print(f"  {relation}: {acc:.3f} (n={count})")

    correlation_length_grade = test_df["длина_ответа"].corr(test_df["label"].astype(float))
    correlation_relation_grade = test_df["отношение_длин"].corr(test_df["label"].astype(float))

    print(f"\nКорреляция длина-оценка: {correlation_length_grade:.3f}")
    print(f"Корреляция отношение_длин-оценка: {correlation_relation_grade:.3f}")

    groups = [group["длина_ответа"].values for name, group in test_df.groupby("Оценка")]
    f_stat, p_value = stats.f_oneway(*groups)
    print(f"ANOVA (различие длин по оценкам): F={f_stat:.2f}, p={p_value:.4f}")

    print("\nСтатистика длин ответов по оценкам:")
    stats_by_grade = test_df.groupby("Оценка").agg({
        "длина_ответа": ["mean", "std", "min", "max"],
        "отношение_длин": ["mean", "std"],
        "label": "count"
    }).round(1)
    print(stats_by_grade)

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(20, 18))

    heatmap_data = pd.crosstab(test_df["диапазон_длины"], test_df["Оценка"])
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0, 0])
    axes[0, 0].set_title('Распределение оценок по диапазонам длин\n(количество примеров)')
    axes[0, 0].set_xlabel('Оценка')
    axes[0, 0].set_ylabel('Диапазон длины ответа')

    accuracy_by_length.plot(kind='bar', ax=axes[0, 1], color='skyblue')
    axes[0, 1].set_title('Точность модели по диапазонам длин ответа')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].set_ylim(0, 1)

    test_df.boxplot(column='длина_ответа', by='Оценка', ax=axes[1, 0])
    axes[1, 0].set_title('Распределение длин ответов по оценкам')
    axes[1, 0].set_ylabel('Длина ответа (символы)')

    test_df.boxplot(column='отношение_длин', by='Оценка', ax=axes[1, 1])
    axes[1, 1].set_title('Отношение длина_ответа/длина_эталона по оценкам')
    axes[1, 1].set_ylabel('Отношение длин')

    confidence_by_length = test_df.groupby("диапазон_длины")["уверенность"].mean()
    confidence_by_length.plot(kind='bar', ax=axes[2, 0], color='lightgreen')
    axes[2, 0].set_title('Средняя уверенность модели по диапазонам длин')
    axes[2, 0].set_ylabel('Уверенность')
    axes[2, 0].tick_params(axis='x', rotation=45)
    axes[2, 0].set_ylim(0, 1)

    for grade in sorted(test_df["Оценка"].unique()):
        grade_data = test_df[test_df["Оценка"] == grade]
        axes[2, 1].scatter(grade_data["длина_ответа"], grade_data["предсказанная_оценка"],
                           alpha=0.6, label=f"Оценка {grade}", s=30)
    axes[2, 1].set_title('Длина ответа vs Предсказанная оценка')
    axes[2, 1].set_xlabel('Длина ответа')
    axes[2, 1].set_ylabel('Предсказанная оценка')
    axes[2, 1].legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/comprehensive_length_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nСамые короткие ответы (топ-5):")
    shortest = test_df.nsmallest(5, "длина_ответа")[
        ["Ответ студента", "длина_ответа", "Оценка", "предсказанная_оценка_текст", "правильно"]]
    for idx, row in shortest.iterrows():
        print(
            f"  Длина: {row['длина_ответа']}, Истинная: {row['Оценка']}, Предсказанная: {row['предсказанная_оценка_текст']}, Правильно: {row['правильно']}")
        print(f"  Ответ: {row['Ответ студента'][:80]}...")

    print("\nСамые длинные ответы (топ-5):")
    longest = test_df.nlargest(5, "длина_ответа")[
        ["Ответ студента", "длина_ответа", "Оценка", "предсказанная_оценка_текст", "правильно"]]
    for idx, row in longest.iterrows():
        print(
            f"  Длина: {row['длина_ответа']}, Истинная: {row['Оценка']}, Предсказанная: {row['предсказанная_оценка_текст']}, Правильно: {row['правильно']}")
        print(f"  Ответ: {row['Ответ студента'][:80]}...")

    results = {
        "accuracy_by_length": accuracy_by_length.to_dict(),
        "accuracy_by_relation": accuracy_by_relation.to_dict(),
        "correlation_length_grade": correlation_length_grade,
        "correlation_relation_grade": correlation_relation_grade,
        "anova_f": f_stat,
        "anova_p": p_value,
        "grade_statistics": test_df.groupby("Оценка")["длина_ответа"].describe().to_dict()
    }

    with open(f'{output_dir}/length_analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nРезультаты сохранены в: {output_dir}/")
    print("  - comprehensive_length_analysis.png (6 графиков)")
    print("  - length_analysis_results.json (статистика)")

    return test_df, results


def main():
    MODEL_PATH = "../models/rubert_4class_enhanced_v3"
    DATA_PATH = "../datasets/test_data.csv"


    if not os.path.exists(MODEL_PATH):
        print(f"Ошибка: Модель не найдена по пути {MODEL_PATH}")
        return

    if not os.path.exists(DATA_PATH):
        print(f"Ошибка: Данные не найдены по пути {DATA_PATH}")
        return

    test_df, results = comprehensive_length_analysis(MODEL_PATH, DATA_PATH)

    best_length = max(results['accuracy_by_length'].items(), key=lambda x: x[1])
    worst_length = min(results['accuracy_by_length'].items(), key=lambda x: x[1])

    print(f"\n1. Лучшая точность: {best_length[0]} ({best_length[1]:.3f})")
    print(f"2. Худшая точность: {worst_length[0]} ({worst_length[1]:.3f})")
    print(f"3. Корреляция длина-оценка: {results['correlation_length_grade']:.3f}")

    if results['anova_p'] < 0.05:
        print("4. Статистически значимая связь длины и оценки: ДА")
    else:
        print("4. Статистически значимая связь длины и оценки: НЕТ")

    print(f"5. Отношение длин влияет на оценку: {abs(results['correlation_relation_grade']):.3f}")


if __name__ == "__main__":
    main()