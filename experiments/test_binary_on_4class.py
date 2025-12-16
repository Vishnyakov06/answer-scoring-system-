# -*- coding: utf-8 -*-
import os
os.environ["WANDB_DISABLED"] = "true"
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import json
from datetime import datetime

from src.datasets import SemanticSimilarityDataset
from src.utils import read_csv

def test_binary_on_multiclass(binary_model_path, multiclass_csv_path, output_dir="./results"):

    df_multiclass = read_csv(multiclass_csv_path).dropna(
        subset=["Вопрос", "Эталонный ответ преподавателя", "Ответ студента", "Оценка"]
    )

    print("Распределение оценок в 4-классовом датасете:")
    grade_distribution = df_multiclass["Оценка"].value_counts().sort_index()
    print(grade_distribution)

    tokenizer = AutoTokenizer.from_pretrained(binary_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(binary_model_path)

    print(f"Количество классов в бинарной модели: {model.config.num_labels}")
    print(f"ID to Label mapping: {model.config.id2label}")

    strategies = {
        "strict": ("Строгая (2,3->0; 4,5->1)", lambda x: 0 if str(x) in ["2", "3"] else 1),
        "liberal": ("Либеральная (2->0; 3,4,5->1)", lambda x: 0 if str(x) == "2" else 1),
        "middle": ("Средняя (2,3,4->0; 5->1)", lambda x: 0 if str(x) in ["2", "3", "4"] else 1)
    }

    all_results = {}

    for strategy_name, (strategy_desc, label_func) in strategies.items():
        print(f"\nСТРАТЕГИЯ: {strategy_desc}")

        test_df = df_multiclass.copy()
        test_df["binary_label"] = test_df["Оценка"].apply(label_func)

        print(f"Распределение бинарных меток:")
        binary_dist = test_df["binary_label"].value_counts().sort_index()
        for label, count in binary_dist.items():
            print(f"  Класс {label}: {count} примеров")

        test_df["binary_label_encoded"] = test_df["binary_label"]

        test_dataset = SemanticSimilarityDataset(test_df, tokenizer)
        tester = Trainer(model=model)
        test_preds = tester.predict(test_dataset)

        test_labels = test_df["binary_label_encoded"].values
        test_predictions = test_preds.predictions.argmax(-1)

        print(f"\nРаспределение истинных меток: {np.bincount(test_labels)}")
        print(f"Распределение предсказаний: {np.bincount(test_predictions)}")

        accuracy = accuracy_score(test_labels, test_predictions)
        f1_macro = f1_score(test_labels, test_predictions, average="macro")
        f1_weighted = f1_score(test_labels, test_predictions, average="weighted")

        print("\nДетальный отчёт:")
        print(classification_report(
            test_labels,
            test_predictions,
            target_names=["Класс_0", "Класс_1"],
            digits=4
        ))

        cm = confusion_matrix(test_labels, test_predictions)
        print("\nConfusion Matrix:")
        print("Истина \\ Предсказание |   Класс 0  |   Класс 1  |")
        print("-" * 55)
        if cm.shape[0] == 2:
            print(f"      Класс 0        |    {cm[0, 0]:5d}    |    {cm[0, 1]:5d}    |")
            print(f"      Класс 1        |    {cm[1, 0]:5d}    |    {cm[1, 1]:5d}    |")

        print("\nАНАЛИЗ ПО ИСХОДНЫМ ОЦЕНКАМ:")
        original_grades = test_df["Оценка"].values
        for grade in ["2", "3", "4", "5"]:
            grade_mask = original_grades == grade
            if sum(grade_mask) > 0:
                grade_predictions = test_predictions[grade_mask]
                grade_true_binary = test_labels[grade_mask]
                grade_accuracy = accuracy_score(grade_true_binary, grade_predictions)
                pred_as_0 = sum(grade_predictions == 0)
                pred_as_1 = sum(grade_predictions == 1)
                total = len(grade_predictions)
                print(f"  Оценка {grade}:")
                print(f"    Точность: {grade_accuracy:.3f}")
                print(f"    Распределение: {pred_as_0}/{total} как '0' ({pred_as_0 / total * 100:.1f}%), "
                      f"{pred_as_1}/{total} как '1' ({pred_as_1 / total * 100:.1f}%)")

        all_results[strategy_name] = {
            'description': strategy_desc,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'confusion_matrix': cm.tolist(),
            'true_distribution': np.bincount(test_labels).tolist(),
            'pred_distribution': np.bincount(test_predictions).tolist(),
            'binary_distribution': binary_dist.to_dict()
        }

        print(f"\nИТОГИ ДЛЯ СТРАТЕГИИ '{strategy_desc}':")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Macro: {f1_macro:.4f}")
        print(f"F1 Weighted: {f1_weighted:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_summary = {
        "timestamp": timestamp,
        "binary_model_path": binary_model_path,
        "multiclass_dataset": multiclass_csv_path,
        "original_distribution": grade_distribution.to_dict(),
        "strategies": all_results,
        "summary": {
            "total_samples_multiclass": len(df_multiclass),
            "best_strategy": max(all_results.items(), key=lambda x: x[1]['accuracy'])[0] if all_results else "none",
            "best_accuracy": max(result['accuracy'] for result in all_results.values()) if all_results else 0
        }
    }

    output_file = f"{output_dir}/binary_on_multiclass_results_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"\nРезультаты сохранены в: {output_file}")

    for strategy_name, result in all_results.items():
        print(f"\n{result['description']}:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  F1 Macro: {result['f1_macro']:.4f}")

    return all_results


def main():
    BINARY_MODEL_PATH = "../models/rubert_binary_classifier"
    MULTICLASS_CSV_PATH = "../datasets/test.csv"
    from src.utils import check_paths
    if not check_paths(BINARY_MODEL_PATH, MULTICLASS_CSV_PATH):
        print("Ошибка: не все файлы найдены")
        return
    results = test_binary_on_multiclass(BINARY_MODEL_PATH, MULTICLASS_CSV_PATH)
    print("ИТОГОВЫЙ АНАЛИЗ:")

    if results:
        best_strategy = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"Лучшая стратегия: {best_strategy[1]['description']}")
        print(f"Лучшая точность: {best_strategy[1]['accuracy']:.4f}")

        accuracy = best_strategy[1]['accuracy']
        if accuracy > 0.7:
            print("Модель показывает хорошую обобщающую способность")
        elif accuracy > 0.5:
            print("Модель показывает среднюю обобщающую способность")
        else:
            print("Модель показывает низкую обобщающую способность")
    else:
        print("Нет результатов для анализа")
if __name__ == "__main__":
    main()