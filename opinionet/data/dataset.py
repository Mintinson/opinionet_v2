"""
Dataset implementation for OpinioNet.

This module provides data loading and processing functionality
using modern PyTorch and Hugging Face transformers.
"""

from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split

# Category mappings
ID2COMMON = ["物流", "服务", "包装", "价格", "真伪", "整体", "其他"]
ID2LAPTOP = ID2COMMON + ["硬件&性能", "软件&性能", "外观", "使用场景"]
ID2MAKEUP = ID2COMMON + ["成分", "尺寸", "功效", "气味", "使用体验", "新鲜度"]
ID2P = ["正面", "中性", "负面"]

# Reverse mappings
LAPTOP2ID = {cat: idx for idx, cat in enumerate(ID2LAPTOP)}
MAKEUP2ID = {cat: idx for idx, cat in enumerate(ID2MAKEUP)}
CAT2IDDICT = {
    "laptop": LAPTOP2ID,
    "makeup": MAKEUP2ID,
}

P2ID = {polar: idx for idx, polar in enumerate(ID2P)}


def pad_batch_sequences(
    sequences: List[List[int]], pad_value: int = 0, max_len: Optional[int] = None
) -> List[List[int]]:
    """
    Pad sequences to the same length.

    Args:
        sequences: List of sequences (lists of integers)
        pad_value: Value to use for padding
        max_len: Maximum length to pad to (if None, use the length of the longest sequence)

    Returns:
        Padded sequences
    """
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    padded_seqs = [
        seq + [pad_value] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len]
        for seq in sequences
    ]
    return padded_seqs


class ReviewDataset(Dataset):
    """
    Dataset for Aspect-Based Sentiment Analysis (ABSA) and Opinion Mining.

    This dataset handles the loading, tokenization, and label alignment for review texts.
    It supports both 'laptop' and 'makeup' domains, mapping domain-specific categories
    to consistent internal IDs. It processes raw CSV data into a format suitable for
    training OpinioNet, including boundary pointers for aspects and opinions.

    The dataset produces samples containing:
    - Raw review text and ground truth labels (for evaluation).
    - Tokenized input IDs.
    - Sequence-level labels for 7 tasks: Aspect Start (AS), Aspect End (AE),
      Opinion Start (OS), Opinion End (OE), Objectiveness (OBJ), Category (C),
      and Polarity (P).
    """

    def __init__(
        self,
        reviews_path: Union[str, Path, pd.DataFrame],
        labels_path: Optional[Union[str, Path, pd.DataFrame]] = None,
        tokenizer: Any = None,
        data_type: Literal["laptop", "makeup"] = "makeup",
        max_length: int = 122,  # 120 + [CLS] + [SEP]
    ) -> None:
        """
        Initialize the ReviewDataset.

        Args:
            reviews_path: Path to the CSV file containing reviews or a pre-loaded DataFrame.
                Must contain 'id' and 'Reviews' columns.
            labels_path: Optional path to the CSV file containing labels or a DataFrame.
                Must contain 'id', 'A_start', 'A_end', 'O_start', 'O_end', 'Categories',
                and 'Polarities' columns.
            tokenizer: A Hugging Face-compatible tokenizer instance.
            data_type: Domain indicator, either 'laptop' or 'makeup', used for category mapping.
            max_length: Maximum sequence length including special tokens.
        """
        super().__init__()

        self.tokenizer = tokenizer
        self.data_type = data_type
        self.max_length = max_length
        self.pad_id: int = tokenizer.convert_tokens_to_ids(["[PAD]"])[0]

        # Load reviews
        reviews_df = (
            pd.read_csv(reviews_path)
            if not isinstance(reviews_path, pd.DataFrame)
            else reviews_path
        )

        # Load labels if provided
        labels_df = None
        if labels_path is not None:
            labels_df = (
                pd.read_csv(labels_path)
                if not isinstance(labels_path, pd.DataFrame)
                else labels_path
            )

        # Set category mapping
        if data_type not in CAT2IDDICT:
            raise ValueError(f"Unsupported data_type: {data_type}")
        self.c2id = CAT2IDDICT[data_type]

        # Preprocess data
        self.samples = self._preprocess_data(reviews_df, labels_df)

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple:
        """
        Retrieves a single sample by index.

        Args:
            index: The index of the sample to retrieve.

        Returns:
            A tuple containing (raw_data, token_ids, labels).
        """
        return self.samples[index]

    def _preprocess_data(
        self, reviews_df: pd.DataFrame, labels_df: Optional[pd.DataFrame]
    ) -> List[Tuple[Tuple[str, Optional[List[Tuple]]], List[int], Optional[Tuple[List[int], ...]]]]:
        """
        Iterates through the reviews and aligns them with labels if provided.

        This method performs character-level truncation, tokenization, and label
        encoding for each review in the input DataFrame.

        Args:
            reviews_df: DataFrame containing review texts.
            labels_df: Optional DataFrame containing ground truth annotations.

        Returns:
            A list of tuples, where each tuple is (raw_data, token_ids, labels).
            raw_data is (text, raw_label_list).
        """

        def process_single_review(
            hash_v: Any, row: Any
        ) -> Tuple[
            Tuple[str, Optional[List[Tuple]]], List[int], Optional[Tuple[List[int], ...]]
        ]:
            review_id = row["id"]
            review_text: str = row["Reviews"][:120]  # Truncate to max length

            # tokenize review
            tokens = self._tokenize_review(review_text)
            token_ids: List[int] = self.tokenizer.convert_tokens_to_ids(tokens)

            # Process labels if available
            if labels_df is not None:
                labels_data = labels_df[labels_df["id"] == review_id]
                labels = self._process_labels(labels_data, len(token_ids))
                raw_data = (review_text, self._extract_raw_labels(labels_data))
            else:
                labels = None
                raw_data = (review_text, None)

            return (raw_data, token_ids, labels)

        return [
            process_single_review(hash_v, row) for hash_v, row in reviews_df.iterrows()
        ]

    def _tokenize_review(self, review: str) -> List[str]:
        """
        Tokenizes a review text using a character-based approach.

        Maps spaces to '[unused1]', known characters to themselves, and unknown
        characters to '[UNK]'. Wraps the sequence with '[CLS]' and '[SEP]'.

        Args:
            review: The raw string text of the review.

        Returns:
            A list of token strings.
        """

        def char_to_token(char: str) -> str:
            if char == " ":
                return "[unused1]"
            elif char in self.tokenizer.vocab:
                return char
            else:
                return "[UNK]"

        return ["[CLS]"] + list(map(char_to_token, review)) + ["[SEP]"]

    def _process_labels(
        self, labels_data: pd.DataFrame, seq_len: int
    ) -> Tuple[List[int], ...]:
        """
        Converts raw span annotations into sequence-level target labels.

        For each annotated aspect-opinion pair, this method fills the corresponding
        indices in the target lists with boundary pointers, category IDs, and
        polarity IDs. It handles offset shifts caused by the '[CLS]' token.

        Args:
            labels_data: Subset of the labels DataFrame for a specific review.
            seq_len: The length of the tokenized sequence.

        Returns:
            A tuple of 7 lists representing the targets for:
            (AS, AE, OS, OE, OBJ, C, P).
        """
        # Initialize label lists
        lb_as = [-1] * seq_len
        lb_ae = [-1] * seq_len
        lb_os = [-1] * seq_len
        lb_oe = [-1] * seq_len
        lb_obj = [0] * seq_len
        lb_c = [-1] * seq_len
        lb_p = [-1] * seq_len

        # Process each opinion
        for _, label_row in labels_data.iterrows():
            a_s = label_row["A_start"].strip()
            a_e = label_row["A_end"].strip()
            o_s = label_row["O_start"].strip()
            o_e = label_row["O_end"].strip()
            category = label_row["Categories"].strip()
            polarity = label_row["Polarities"].strip()

            # Convert category and polarity to IDs
            c_id = self.c2id.get(category, -1)
            p_id = P2ID.get(polarity, -1)

            # Convert positions (add 1 for [CLS] token)
            if a_s != "" and a_e != "":
                a_s, a_e = int(a_s) + 1, int(a_e)
            else:
                a_s, a_e = 0, 0

            if o_s != "" and o_e != "":
                o_s, o_e = int(o_s) + 1, int(o_e)
            else:
                o_s, o_e = 0, 0

            # Validate positions
            if a_s >= seq_len - 1:
                a_s, a_e = 0, 0
            if o_s >= seq_len - 1:
                o_s, o_e = 0, 0

            # Clamp positions
            a_s = min(a_s, seq_len - 2)
            a_e = min(a_e, seq_len - 2)
            o_s = min(o_s, seq_len - 2)
            o_e = min(o_e, seq_len - 2)

            # Set labels for aspect span
            if a_s > 0:
                lb_as[a_s : a_e + 1] = [a_s] * (a_e - a_s + 1)
                lb_ae[a_s : a_e + 1] = [a_e] * (a_e - a_s + 1)
                lb_os[a_s : a_e + 1] = [o_s] * (a_e - a_s + 1)
                lb_oe[a_s : a_e + 1] = [o_e] * (a_e - a_s + 1)
                lb_obj[a_s : a_e + 1] = [1] * (a_e - a_s + 1)
                lb_c[a_s : a_e + 1] = [c_id] * (a_e - a_s + 1)
                lb_p[a_s : a_e + 1] = [p_id] * (a_e - a_s + 1)

            # Set labels for opinion span
            if o_s > 0:
                lb_as[o_s : o_e + 1] = [a_s] * (o_e - o_s + 1)
                lb_ae[o_s : o_e + 1] = [a_e] * (o_e - o_s + 1)
                lb_os[o_s : o_e + 1] = [o_s] * (o_e - o_s + 1)
                lb_oe[o_s : o_e + 1] = [o_e] * (o_e - o_s + 1)
                lb_obj[o_s : o_e + 1] = [1] * (o_e - o_s + 1)
                lb_c[o_s : o_e + 1] = [c_id] * (o_e - o_s + 1)
                lb_p[o_s : o_e + 1] = [p_id] * (o_e - o_s + 1)

        return lb_as, lb_ae, lb_os, lb_oe, lb_obj, lb_c, lb_p

    def _extract_raw_labels(self, labels_data: pd.DataFrame) -> List[Tuple]:
        """
        Extracts raw annotation tuples for evaluation purposes.

        Args:
            labels_data: Subset of the labels DataFrame for a specific review.

        Returns:
            A list of tuples: (a_start, a_end, o_start, o_end, category_id, polarity_id).
        """
        raw_labels = []

        for _, row in labels_data.iterrows():
            a_s = row["A_start"].strip()
            a_e = row["A_end"].strip()
            o_s = row["O_start"].strip()
            o_e = row["O_end"].strip()
            category = row["Categories"].strip()
            polarity = row["Polarities"].strip()

            c_id = self.c2id.get(category, -1)
            p_id = P2ID.get(polarity, -1)

            if a_s != "" and a_e != "":
                a_s, a_e = int(a_s) + 1, int(a_e)
            else:
                a_s, a_e = 0, 0

            if o_s != "" and o_e != "":
                o_s, o_e = int(o_s) + 1, int(o_e)
            else:
                o_s, o_e = 0, 0

            raw_labels.append((a_s, a_e, o_s, o_e, c_id, p_id))

        return raw_labels

    def collate_fn(
        self, batch: List[Tuple]
    ) -> Tuple[Tuple[List, List], List[torch.LongTensor], Optional[List[torch.Tensor]]]:
        """
        Batches and pads samples for the DataLoader.

        Handles the padding of input IDs, attention masks, and token masks.
        If labels are present, it also pads the 7 target sequences using
        appropriate padding values (0 for objectiveness, -1 for others).

        Args:
            batch: A list of tuples from __getitem__.

        Returns:
            A tuple containing:
            - ((raw_texts, raw_labels)): Lists of raw data for evaluation.
            - [input_ids, attention_mask, token_mask]: Tensors for model input.
            - targets: A list of 7 tensors for loss calculation, or None.
        """
        rv_raw = []
        lb_raw = []
        input_ids_list = []
        attention_mask_list = []
        token_mask_list = []

        for raw_data, token_ids, _ in batch:
            rv_raw.append(raw_data[0])
            lb_raw.append(raw_data[1])  # Append raw labels

            input_ids_list.append(token_ids)
            attention_mask_list.append([1] * len(token_ids))
            token_mask_list.append([0] + [1] * (len(token_ids) - 2) + [0])

        # Pad sequences
        input_ids = torch.LongTensor(pad_batch_sequences(input_ids_list, self.pad_id))
        attention_mask = torch.LongTensor(pad_batch_sequences(attention_mask_list, 0))
        token_mask = torch.LongTensor(pad_batch_sequences(token_mask_list, 0))

        inputs = [input_ids, attention_mask, token_mask]

        # Process targets if available
        targets: Optional[List[torch.Tensor]] = None
        first_labels = batch[0][2]

        if first_labels is not None:
            # Initialize temporary storage for collecting batch data
            # raw_targets: [label_type_index][batch_index][sequence]
            raw_targets: List[List[List[int]]] = [[] for _ in range(len(first_labels))]

            for _, _, labels in batch:
                if labels is not None:
                    for i, label_list in enumerate(labels):
                        raw_targets[i].append(label_list)

            # Convert to tensors
            targets = []
            for i, label_batch in enumerate(raw_targets):
                if i == 4:  # Objectiveness (float)
                    targets.append(torch.FloatTensor(pad_batch_sequences(label_batch, 0)))
                else:  # Others (long)
                    targets.append(torch.LongTensor(pad_batch_sequences(label_batch, -1)))

        return (rv_raw, lb_raw), inputs, targets


def get_data_loaders(
    reviews_path: Union[str, Path],
    labels_path: Union[str, Path],
    tokenizer: Any,
    batch_size: int,
    data_type: Literal["makeup", "laptop"] = "makeup",
    val_split: float = 0.15,
    num_workers: int = 4,
    seed: int = 502,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.

    Args:
        reviews_path: Path to reviews CSV
        labels_path: Path to labels CSV
        tokenizer: Tokenizer instance
        batch_size: Batch size
        data_type: Type of data ('makeup' or 'laptop')
        val_split: Validation split ratio
        num_workers: Number of data loading workers
        seed: Random seed for split

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create full dataset
    full_dataset = ReviewDataset(
        reviews_path, labels_path, tokenizer, data_type=data_type
    )

    # Split into train/val
    train_size = int(len(full_dataset) * (1 - val_split))
    val_size = len(full_dataset) - train_size

    torch.manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=full_dataset.collate_fn,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=full_dataset.collate_fn,
        drop_last=False,
    )

    return train_loader, val_loader


class CorpusDataset(Dataset):
    """
    Dataset for unlabeled corpus used in MLM pretraining.

    This dataset loads review text and prepares it for masked language modeling.
    """

    def __init__(self, corpus_path: Union[str, Path], tokenizer: Any) -> None:
        """
        Initialize corpus dataset.

        Args:
            corpus_path: Path to corpus CSV file
            tokenizer: Tokenizer to use
        """
        super().__init__()

        corpus_df = pd.read_csv(corpus_path, encoding="utf-8")

        self.tokenizer = tokenizer
        self.samples = self._preprocess_data(corpus_df, tokenizer)

    def _preprocess_data(
        self, corpus_df: pd.DataFrame, tokenizer: Any
    ) -> List[Tuple[str, List[int], List[Tuple[int, int]]]]:
        """Preprocess corpus data.

        Args:
            corpus_df (pd.DataFrame): DataFrame with 'Reviews' column
            tokenizer (Any): Tokenizer

        Returns:
            List[Tuple[str, List[int], List[Tuple[int, int]]]]:
                List of (text, token_ids, word_intervals)
        """
        import jieba

        def process_single_review(
            row: Any,
        ) -> Tuple[str, List[int], List[Tuple[int, int]]]:
            review_text = row["Reviews"]

            RV = ["[CLS]"]
            RV_INTERVALS = []
            words = jieba.cut(review_text)
            for word in words:
                s = len(RV)
                for c in word:
                    if c == " ":
                        RV.append("[unused1]")
                    elif c in tokenizer.vocab:
                        RV.append(c)
                    else:
                        RV.append("[UNK]")
                e = len(RV)
                RV_INTERVALS.append((s, e))

            RV.append("[SEP]")
            RV = tokenizer.convert_tokens_to_ids(RV)

            return (review_text, RV, RV_INTERVALS)

        return [
            process_single_review(row)
            for _, row in corpus_df.iterrows()
            if len(row["Reviews"]) >= 120
        ]

    def collate_fn(
        self, batch_samples: List[Tuple[str, List[int], List[Tuple[int, int]]]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate function for batching with MLM masking.

        Args:
            batch_samples: List of (text, token_ids, word_intervals)

        Returns:
            Tuple of (input_ids, attention_mask, lm_labels)
        """
        import numpy as np

        def _apply_mlm_masking(
            rv: List[int], rv_intervals: List[Tuple[int, int]]
        ) -> Tuple[List[int], List[int]]:
            """Apply masked language modeling to a single review."""
            masked_rv = rv.copy()
            lm_label = [-1] * len(masked_rv)

            # Sample 15% of words to mask
            mask_word_num = int(len(rv_intervals) * 0.15)
            masked_word_idxs = np.random.choice(
                len(rv_intervals), mask_word_num, replace=False
            )

            for word_idx in masked_word_idxs:
                s, e = rv_intervals[word_idx]
                for token_idx in range(s, e):
                    lm_label[token_idx] = masked_rv[token_idx]
                    rand = np.random.rand()

                    # Determine replacement: 80% mask, 10% random, 10% keep
                    if rand < 0.8:  # Mask
                        replace_id = self.tokenizer.vocab["[MASK]"]
                    elif rand < 0.9:  # Random replacement
                        replace_id = np.random.choice(len(self.tokenizer.vocab))
                    else:  # Keep original
                        replace_id = masked_rv[token_idx]

                    masked_rv[token_idx] = replace_id

            return masked_rv, lm_label

        # Process batch samples using functional approach
        processed = [
            _apply_mlm_masking(rv, intervals) for _, rv, intervals in batch_samples
        ]
        input_ids_list, lm_labels_list = zip(*processed)
        attn_mask_list = [[1] * len(ids) for ids in input_ids_list]

        # Pad and convert to tensors
        return (
            torch.LongTensor(
                pad_batch_sequences(list(input_ids_list), self.tokenizer.vocab["[PAD]"])
            ),
            torch.LongTensor(pad_batch_sequences(list(attn_mask_list), 0)),
            torch.LongTensor(pad_batch_sequences(list(lm_labels_list), -1)),
        )

    def __getitem__(self, index: int) -> Tuple[str, List[int], List[Tuple[int, int]]]:
        """Get item by index."""
        return self.samples[index]

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.samples)


def get_pretrain_loaders(
    tokenizer: Any, batch_size: int = 12, val_split: float = 0.15
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for pretraining.

    Args:
        tokenizer: Tokenizer to use
        batch_size: Batch size
        val_split: Validation split ratio

    Returns:
        Tuple of (makeup_train_loader, makeup_val_loader, corpus_loader)
    """

    from torch.utils.data import ConcatDataset

    # Load makeup review datasets (labeled)
    makeup_rv1 = ReviewDataset(
        reviews_path="data/TRAIN/Train_reviews.csv",
        labels_path="data/TRAIN/Train_labels.csv",
        tokenizer=tokenizer,
        data_type="makeup",
    )
    makeup_rv2 = ReviewDataset(
        reviews_path="data/TRAIN/Train_makeup_reviews.csv",
        labels_path="data/TRAIN/Train_makeup_labels.csv",
        tokenizer=tokenizer,
        data_type="makeup",
    )
    makeup_rv = ConcatDataset([makeup_rv1, makeup_rv2])

    # Load corpus datasets (unlabeled, for MLM)
    laptop_corpus1 = CorpusDataset("data/TEST/Test_reviews.csv", tokenizer)
    laptop_corpus2 = CorpusDataset("data/TRAIN/Train_laptop_corpus.csv", tokenizer)
    laptop_corpus3 = CorpusDataset("data/TRAIN/Train_laptop_reviews.csv", tokenizer)
    makeup_corpus1 = CorpusDataset("data/TEST/Test_reviews1.csv", tokenizer)
    makeup_corpus2 = CorpusDataset("data/TRAIN/Train_reviews.csv", tokenizer)
    makeup_corpus3 = CorpusDataset("data/TRAIN/Train_makeup_reviews.csv", tokenizer)

    corpus_rv = ConcatDataset(
        [
            laptop_corpus1,
            laptop_corpus2,
            laptop_corpus3,
            makeup_corpus1,
            makeup_corpus2,
            makeup_corpus3,
        ]
    )

    # Create corpus loader
    corpus_loader = DataLoader(
        corpus_rv,
        batch_size=batch_size,
        collate_fn=laptop_corpus1.collate_fn,
        shuffle=True,
        num_workers=5,
        drop_last=False,
    )

    # Split makeup data into train/val
    makeup_train_size = int(len(makeup_rv) * (1 - val_split))
    makeup_val_size = len(makeup_rv) - makeup_train_size
    torch.manual_seed(502)
    makeup_train, makeup_val = random_split(
        makeup_rv, [makeup_train_size, makeup_val_size]
    )

    # Create makeup loaders
    makeup_train_loader = DataLoader(
        makeup_train,
        batch_size=batch_size,
        collate_fn=makeup_rv1.collate_fn,
        shuffle=True,
        num_workers=5,
        drop_last=False,
    )
    makeup_val_loader = DataLoader(
        makeup_val,
        batch_size=batch_size,
        collate_fn=makeup_rv1.collate_fn,
        shuffle=False,
        num_workers=5,
        drop_last=False,
    )

    return makeup_train_loader, makeup_val_loader, corpus_loader


__all__ = [
    "ReviewDataset",
    "get_data_loaders",
    "ID2COMMON",
    "ID2LAPTOP",
    "ID2MAKEUP",
    "ID2P",
    "LAPTOP2ID",
    "MAKEUP2ID",
    "P2ID",
    "pad_batch_sequences",
    "CorpusDataset",
    "get_pretrain_loaders",
]
