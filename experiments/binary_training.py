# -*- coding: utf-8 -*-
import os
os.environ["WANDB_DISABLED"] = "true"
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
    EarlyStoppingCallback
import json
from datetime import datetime
import random

from src.datasets import AugmentedSemanticSimilarityDataset
from src.trainers import SimpleTrainer
from src.utils import read_excel

def main():
    df = read_excel("../datasets/big_data_binary.xlsx").dropna(
        subset=["Вопрос", "Эталонный ответ преподавателя", "Ответ студента", "Оценка"]
    )
    print("Распределение оценок:")
    print(df["Оценка"].value_counts().sort_index())
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["Оценка"].astype(str))
    print(f"Классы: {label_encoder.classes_}")

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

    print(f"\nРазмеры выборок:")
    print(f"Train: {len(train_df)}")
    print(f"Val: {len(val_df)}")
    print(f"Test: {len(test_df)}")

    MODEL_NAME = "DeepPavlov/rubert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = AugmentedSemanticSimilarityDataset(train_df, tokenizer, is_training=True)
    val_dataset = AugmentedSemanticSimilarityDataset(val_df, tokenizer)
    test_dataset = AugmentedSemanticSimilarityDataset(test_df, tokenizer)

    num_labels = 2

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        hidden_dropout_prob=0.4,
        attention_probs_dropout_prob=0.4,
        classifier_dropout=0.5
    )

    training_args = TrainingArguments(
        output_dir="../models/rubert_binary_classifier",
        num_train_epochs=12,
        learning_rate=8e-6,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        warmup_ratio=0.1,
        weight_decay=0.15,
        logging_steps=30,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_drop_last=False,
        remove_unused_columns=False,
        max_grad_norm=0.5,
        report_to="none",
        lr_scheduler_type="linear",
        warmup_steps=50,
        save_total_limit=2,
        seed=42,
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        print(f"\n--- ДИАГНОСТИКА ---")
        print(f"Истинное распределение: {np.bincount(labels)}")
        print(f"Предсказанное распределение: {np.bincount(preds)}")

        if len(np.bincount(preds)) == 1 or min(np.bincount(preds)) / max(np.bincount(preds)) < 0.1:
            print("ВОЗМОЖНОЕ ЗАВИСАНИЕ НА ОДНОМ КЛАССЕ!")

        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
            "f1_weighted": f1_score(labels, preds, average="weighted")
        }



    trainer = SimpleTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    print("ФИНАЛЬНАЯ ОЦЕНКА НА ТЕСТЕ")

    test_preds = trainer.predict(test_dataset)
    test_labels = test_df["label"].values
    test_predictions = test_preds.predictions.argmax(-1)

    print(f"Распределение истинных меток: {np.bincount(test_labels)}")
    print(f"Распределение предсказаний: {np.bincount(test_predictions)}")

    print("\nДетальный отчёт:")
    print(classification_report(
        test_labels,
        test_predictions,
        target_names=[str(cls) for cls in label_encoder.classes_]
    ))

    probabilities = torch.softmax(torch.tensor(test_preds.predictions), dim=1)
    confidence_scores = torch.max(probabilities, dim=1)[0].numpy()

    print(f"\nАнализ уверенности:")
    print(f"Средняя уверенность: {np.mean(confidence_scores):.3f}")

    trainer.save_model("./models/rubert_binary_classifier")
    tokenizer.save_pretrained("./models/rubert_binary_classifier")

    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    distribution_matrix = {
        "timestamp": timestamp,
        "confusion_matrix": confusion_matrix(test_labels, test_predictions).tolist(),
        "class_names": label_encoder.classes_.tolist(),
        "summary": {
            "total_samples": len(test_labels),
            "accuracy": float(accuracy_score(test_labels, test_predictions)),
            "f1_macro": float(f1_score(test_labels, test_predictions, average='macro')),
            "confidence_stats": {
                "mean": float(np.mean(confidence_scores)),
                "min": float(np.min(confidence_scores)),
                "max": float(np.max(confidence_scores))
            }
        }
    }

    with open(f"{results_dir}/binary_results_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(distribution_matrix, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()