# -*- coding: utf-8 -*-
import torch
import random
from torch.utils.data import Dataset
class SemanticSimilarityDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        question = row['Вопрос']
        reference_answer = row['Эталонный ответ преподавателя']
        student_answer = row['Ответ студента']

        text_a = f"Задание: {question} Студент ответил: {student_answer}"
        text_b = f"Правильный ответ: {reference_answer}"

        encoded = self.tokenizer(
            text_a,
            text_b,
            padding="max_length",
            truncation="only_second",
            max_length=self.max_len,
            return_tensors="pt"
        )
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        encoded["labels"] = torch.tensor(row["label"], dtype=torch.long)

        return encoded
class AugmentedSemanticSimilarityDataset(SemanticSimilarityDataset):

    def __init__(self, df, tokenizer, max_len=512, is_training=False):
        super().__init__(df, tokenizer, max_len)
        self.is_training = is_training
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        question = row['Вопрос']
        reference_answer = row['Эталонный ответ преподавателя']
        student_answer = row['Ответ студента']

        if self.is_training and random.random() < 0.7:
            text_a, text_b = self._augment_template(question, reference_answer, student_answer)
        else:
            text_a = f"Задание: {question} Студент ответил: {student_answer}"
            text_b = f"Правильный ответ: {reference_answer}"
        encoded = self.tokenizer(
            text_a,
            text_b,
            padding="max_length",
            truncation="only_second",
            max_length=self.max_len,
            return_tensors="pt"
        )
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        encoded["labels"] = torch.tensor(row["label"], dtype=torch.long)
        return encoded
    def _augment_template(self, question, reference_answer, student_answer):
        templates = [
            (f"Вопрос: {question} Ответ студента: {student_answer}",
             f"Эталонный ответ: {reference_answer}"),
            (f"Задание: {question} Учащийся написал: {student_answer}",
             f"Правильный вариант: {reference_answer}"),
            (f"Дано задание: {question} Студент ответил: {student_answer}",
             f"Образцовый ответ: {reference_answer}"),
            (f"Учебный вопрос: {question} Работа студента: {student_answer}",
             f"Эталон: {reference_answer}"),
            (f"Question: {question} Student: {student_answer}",
             f"Reference: {reference_answer}"),
            (f"Сравни ответ студента с эталоном. Задание: {question}",
             f"Студент: {student_answer} | Эталон: {reference_answer}"),
            (f"Оцени соответствие ответа. Вопрос: {question}",
             f"Ответ студента: {student_answer} | Правильный ответ: {reference_answer}"),
        ]
        return random.choice(templates)