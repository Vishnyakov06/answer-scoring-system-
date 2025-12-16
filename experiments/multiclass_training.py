# -*- coding: utf-8 -*-
import os
os.environ["WANDB_DISABLED"] = "true"
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, \
    EarlyStoppingCallback
from sklearn.utils.class_weight import compute_class_weight

from src.datasets import AugmentedSemanticSimilarityDataset, SemanticSimilarityDataset
from src.trainers import WeightedTrainer
from src.utils import read_csv

def main():
    train_df_full = read_csv("../datasets/train.csv").dropna(
        subset=["Вопрос", "Эталонный ответ преподавателя", "Ответ студента", "Оценка"]
    )

    test_df = read_csv("../datasets/test.csv").dropna(
        subset=["Вопрос", "Эталонный ответ преподавателя", "Ответ студента", "Оценка"]
    )
    label_encoder = LabelEncoder()
    train_df_full["label"] = label_encoder.fit_transform(
        train_df_full["Оценка"].astype(str)
    )

    test_df["label"] = label_encoder.transform(
        test_df["Оценка"].astype(str)
    )
    print("Распределение классов:")
    print(train_df_full["Оценка"].value_counts().sort_index())
    print(f"Закодированные классы: {label_encoder.classes_}")

    train_df, val_df = train_test_split(
        train_df_full,
        test_size=0.1,
        random_state=42,
        stratify=train_df_full["label"]
    )

    print(f"\nРазмеры выборок:")
    print(f"Train: {len(train_df)}")
    print(f"Val: {len(val_df)}")
    print(f"Test: {len(test_df)}")

    MODEL_NAME = "DeepPavlov/rubert-base-cased"
    MODEL_PATH = "../models/rubert_backup_rubert_4class_enhanced_v3"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = AugmentedSemanticSimilarityDataset(train_df, tokenizer)
    train_dataset.is_training = True
    val_dataset = SemanticSimilarityDataset(val_df, tokenizer)
    test_dataset = SemanticSimilarityDataset(test_df, tokenizer)

    num_labels = len(label_encoder.classes_)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_df["label"]),
        y=train_df["label"]
    )
    print(f"\nИсходные веса классов: {class_weights}")

    class_weights_adjusted = class_weights.copy()
    class_weights_adjusted[2] = class_weights[2] * 1.4
    class_weights_adjusted[3] = class_weights[3] * 1.5

    print(f"Скорректированные веса: {class_weights_adjusted}")
    class_weights_tensor = torch.tensor(class_weights_adjusted, dtype=torch.float)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=num_labels,
        hidden_dropout_prob=0.25,
        attention_probs_dropout_prob=0.25,
        classifier_dropout=0.35
    )

    training_args = TrainingArguments(
        output_dir="../models/rubert_4class_enhanced_v4",
        num_train_epochs=6,
        learning_rate=1e-6,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_weighted",
        greater_is_better=True,
        dataloader_drop_last=False,
        remove_unused_columns=False,
        max_grad_norm=0.8,
        report_to="none",
        lr_scheduler_type="linear",
        warmup_steps=100,
        save_total_limit=2,
        seed=42,
        logging_dir="./logs",
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        print(f"True labels distribution: {np.bincount(labels)}")
        print(f"Predicted labels distribution: {np.bincount(preds)}")

        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
            "f1_weighted": f1_score(labels, preds, average="weighted")
        }

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()

    print("ФИНАЛЬНАЯ ОЦЕНКА НА ТЕСТЕ")

    test_preds = trainer.predict(test_dataset)
    test_labels = test_df["label"].values
    test_predictions = test_preds.predictions.argmax(-1)

    print(f"\nTrue labels distribution: {np.bincount(test_labels)}")
    print(f"Predicted labels distribution: {np.bincount(test_predictions)}")

    print("\nДетальный отчёт:")
    print(classification_report(
        test_labels,
        test_predictions,
        target_names=[str(cls) for cls in label_encoder.classes_]
    ))

    trainer.save_model("./models/rubert_4class_enhanced")
    tokenizer.save_pretrained("./models/rubert_4class_enhanced")

    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")

    accuracy = accuracy_score(test_labels, test_predictions)
    f1_macro = f1_score(test_labels, test_predictions, average='macro')
    f1_weighted = f1_score(test_labels, test_predictions, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")

    print("\nАНАЛИЗ ПО КЛАССАМ:")
    for i, class_name in enumerate(label_encoder.classes_):
        class_mask = test_labels == i
        if sum(class_mask) > 0:
            class_accuracy = accuracy_score(
                test_labels[class_mask],
                test_predictions[class_mask]
            )
            print(f"Класс {class_name}: Accuracy = {class_accuracy:.3f}")

if __name__ == "__main__":
    main()