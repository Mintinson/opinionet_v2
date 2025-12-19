"""Data augmentation utilities for aspect-based sentiment analysis.

This module provides functionality to augment training data by replacing
aspect-opinion pairs while maintaining semantic consistency.
"""

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Literal, Set, Tuple, cast

import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass
class SpanElement:
    """
    Represents a text span (aspect or opinion) in a review for reconstruction.

    Attributes:
        idx: The original index in the labels DataFrame.
        text: The text content of the span.
        span: A tuple of (start_index, end_index) as strings.
        offset: The length difference after replacement (used for position updates).
        type: The type of span, either 'a' (aspect) or 'o' (opinion).
    """

    idx: int
    text: str
    span: Tuple[str, str]
    offset: int
    type: Literal["a", "o"]  # 'a' for aspect, 'o' for opinion


def _build_category_dictionary(
    labels_df: pd.DataFrame,
) -> Dict[str, Dict[str, Set[Tuple[str, str]]]]:
    """
    Builds a lookup dictionary for aspect-opinion pairs grouped by category and polarity.

    This dictionary is used to sample semantically similar replacements during augmentation.
    It ensures that an aspect-opinion pair used for replacement has been seen in the
    same category and with the same sentiment polarity in the training set.

    Args:
        labels_df: DataFrame containing labeled data. Expected columns:
                   'Categories', 'AspectTerms', 'OpinionTerms', 'Polarities'.

    Returns:
        A nested dictionary structure: {category: {polarity: {(aspect, opinion), ...}}}.
    """
    cate_dict: Dict[str, Dict[str, Set[Tuple[str, str]]]] = defaultdict(lambda: defaultdict(set))

    for _, row in labels_df.iterrows():
        cate = str(row["Categories"])
        aspect = str(row["AspectTerms"])
        opinion = str(row["OpinionTerms"])
        polarity = str(row["Polarities"])

        cate_dict[cate][polarity].add((aspect, opinion))

    return {k: dict(v) for k, v in cate_dict.items()}


def _sample_replacement_pair(
    pair_set: Set[Tuple[str, str]],
    current_aspect: str,
    current_opinion: str,
    max_attempts: int = 1000,
) -> Tuple[str, str]:
    """
    Samples a new aspect-opinion pair from a set, ensuring it differs from the current one.

    Args:
        pair_set: Set of available (aspect, opinion) tuples for the target category/polarity.
        current_aspect: The aspect term currently in the review.
        current_opinion: The opinion term currently in the review.
        max_attempts: Safety limit for random sampling to prevent infinite loops.

    Returns:
        A tuple of (new_aspect, new_opinion). Returns original if no alternatives exist.
    """
    if len(pair_set) <= 1:
        return current_aspect, current_opinion

    pair_list = list(pair_set)
    new_aspect, new_opinion = current_aspect, current_opinion

    for _ in range(max_attempts):
        new_aspect, new_opinion = random.choice(pair_list)
        # Accept if: (1) current is "_" and new is "_", OR (2) pair is different
        if (current_aspect == "_" and new_aspect == "_") or (
            new_aspect != current_aspect or new_opinion != current_opinion
        ):
            break

    return new_aspect, new_opinion


def _extract_spans(group: pd.DataFrame) -> List[SpanElement]:
    """
    Extracts and sorts all unique aspect and opinion spans from a group of labels.

    This is a critical step for review reconstruction, as spans must be processed
    in order of their appearance in the text to correctly calculate new offsets.

    Args:
        group: A DataFrame containing all annotations for a single review.

    Returns:
        A list of SpanElement objects, sorted by their starting character position.
    """
    spans: List[SpanElement] = []
    span_set = set()

    for idx, row in group.iterrows():
        # Extract aspect span
        aspect_span = (str(row["A_start"]).strip(), str(row["A_end"]).strip())
        if (
            row["AspectTerms"] != "_"
            and aspect_span not in span_set
            and aspect_span[0].isdigit()
        ):
            span_set.add(aspect_span)
            spans.append(
                SpanElement(
                    idx=idx, # pyright: ignore[reportArgumentType]
                    text=row["AspectTerms"],
                    span=aspect_span,
                    offset=row["AspectOffset"],
                    type="a",
                )
            )

        # Extract opinion span
        opinion_span = (str(row["O_start"]).strip(), str(row["O_end"]).strip())
        if (
            row["OpinionTerms"] != "_"
            and opinion_span not in span_set
            and opinion_span[0].isdigit()
        ):
            span_set.add(opinion_span)
            spans.append(
                SpanElement(
                    idx=idx, # pyright: ignore[reportArgumentType]
                    text=row["OpinionTerms"],
                    span=opinion_span,
                    offset=row["OpinionOffset"],
                    type="o",
                )
            )

    return sorted(spans, key=lambda s: int(s.span[0]))


def _reconstruct_review(
    original_review: str, sorted_spans: List[SpanElement], group: pd.DataFrame
) -> Tuple[str, pd.DataFrame]:
    """
    Reconstructs the review text by replacing spans and updating their coordinates.

    As terms are replaced with strings of different lengths, the start and end
    positions of all subsequent spans in the review must be shifted.

    Args:
        original_review: The raw text of the review before augmentation.
        sorted_spans: List of SpanElements sorted by position.
        group: The label DataFrame for this review, which will be updated in-place.

    Returns:
        A tuple containing the new review string and the updated label DataFrame.
    """
    new_review = ""
    last_end = 0

    for span in sorted_spans:
        start_pos = int(span.span[0])
        end_pos = int(span.span[1])

        # Append text before current span and the replacement text
        new_review += original_review[last_end:start_pos]
        if span.text != "_":
            new_review += span.text
            new_start = len(new_review) - len(span.text)
            new_end = len(new_review)
        else:
            new_start = new_end = len(new_review)

        # Update span positions in group
        if span.type == "a":
            group.loc[span.idx, "A_start"] = str(new_start)
            group.loc[span.idx, "A_end"] = str(new_end)
        else:
            group.loc[span.idx, "O_start"] = str(new_start)
            group.loc[span.idx, "O_end"] = str(new_end)

        last_end = end_pos

    new_review += original_review[last_end:]
    return new_review, group


def _augment_group(
    group: pd.DataFrame,
    indices: List[int],
    sample_size: int,
    cate_dict: Dict[str, Dict[str, Set[Tuple[str, str]]]],
) -> pd.DataFrame:
    """
    Selects a subset of annotations in a review and replaces their terms.

    Args:
        group: DataFrame group containing annotations for one review.
        indices: List of row indices within the group that share the same polarity.
        sample_size: Number of annotations to actually replace.
        cate_dict: The category-polarity lookup dictionary.

    Returns:
        The modified DataFrame group with updated terms and calculated length offsets.
    """
    # Initialize offset tracking
    group["AspectOffset"] = 0
    group["OpinionOffset"] = 0

    # Randomly select rows to augment
    chosen_indices = np.random.choice(indices, sample_size, replace=False)

    for idx in chosen_indices:
        row = group.loc[idx]
        cate = str(row["Categories"])
        aspect = str(row["AspectTerms"])
        opinion = str(row["OpinionTerms"])
        polarity = str(row["Polarities"])

        # Get replacement pair
        pair_set = cate_dict.get(cate, {}).get(polarity, set())
        new_aspect, new_opinion = _sample_replacement_pair(
            pair_set, aspect, opinion
        )

        # Update group with replacements
        group.loc[idx, "AspectTerms"] = new_aspect
        group.loc[idx, "AspectOffset"] = (
            len(new_aspect) if new_aspect != "_" else 0
        ) - (len(aspect) if aspect != "_" else 0)

        group.loc[idx, "OpinionTerms"] = new_opinion
        group.loc[idx, "OpinionOffset"] = (
            len(new_opinion) if new_opinion != "_" else 0
        ) - (len(opinion) if opinion != "_" else 0)

    return group


def data_augment(
    reviews_df: pd.DataFrame, labels_df: pd.DataFrame, epochs: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs data augmentation on the entire dataset over multiple epochs.

    The augmentation strategy involves:
    1. Building a dictionary of valid aspect-opinion pairs per category/polarity.
    2. For each review, grouping its annotations by polarity.
    3. Generating multiple augmented versions of the review by replacing different
       combinations of aspect-opinion pairs.
    4. Re-calculating all character offsets for the new review text.

    Args:
        reviews_df: DataFrame with 'id' and 'Reviews'.
        labels_df: DataFrame with annotation details.
        epochs: Number of times to iterate over the dataset for augmentation.

    Returns:
        A tuple of (augmented_reviews_df, augmented_labels_df).
    """
    # Build category-polarity dictionary
    cate_dict = _build_category_dictionary(labels_df)

    # Initialize output DataFrames
    new_reviews_df = pd.DataFrame(columns=reviews_df.columns)
    new_labels_df = pd.DataFrame(columns=labels_df.columns)

    review_id = 1
    label_idx = 1
    label_groups = labels_df.groupby("id")

    for epoch in range(epochs):
        for group_id, group in tqdm(
            label_groups, desc=f"Epoch {epoch + 1}/{epochs}"
        ):
            # Ensure group_id is treated as an index key for reviews_df
            gid = cast(int, group_id)
            # Find the review text associated with this group ID
            review_row = reviews_df[reviews_df["id"] == gid]
            if review_row.empty:
                continue
            original_review = str(review_row.iloc[0]["Reviews"])

            # Group labels by polarity
            polarity_groups: Dict[str, List[int]] = defaultdict(list)
            for idx, row in group.iterrows():
                polarity_groups[str(row["Polarities"])].append(cast(int, idx))

            # Generate augmented samples for each polarity
            for polarity, indices in polarity_groups.items():
                for sample_size in range(1, len(indices) + 1):
                    augmented_group = _augment_group(
                        group.copy(),
                        indices,
                        sample_size,
                        cate_dict,
                    )

                    # Reconstruct review with replacements
                    sorted_spans = _extract_spans(augmented_group)
                    new_review, updated_group = _reconstruct_review(
                        original_review, sorted_spans, augmented_group
                    )

                    # Save results
                    updated_group = updated_group.drop(
                        columns=["AspectOffset", "OpinionOffset"], errors="ignore"
                    )
                    for _, row in updated_group.iterrows():
                        row_data = row.tolist()
                        row_data[0] = review_id
                        new_labels_df.loc[label_idx] = row_data
                        label_idx += 1

                    new_reviews_df.loc[review_id] = [review_id, new_review]
                    review_id += 1

    return new_reviews_df, new_labels_df


if __name__ == "__main__":
    data_type = "makeup"

    epochs = 3

    reviews_df = pd.read_csv("data/TRAIN/Train_reviews.csv", encoding="utf-8")
    labels_df = pd.read_csv("data/TRAIN/Train_labels.csv", encoding="utf-8")

    new_reviews_df, new_labels_df = data_augment(reviews_df, labels_df, epochs=epochs)

    new_reviews_df.to_csv(
        f"data/TRAIN/Train_reviews_augmented_{data_type}_{epochs}x.csv",
        index=False,
        encoding="utf-8",
    )
    new_labels_df.to_csv(
        f"data/TRAIN/Train_labels_augmented_{data_type}_{epochs}x.csv",
        index=False,
        encoding="utf-8",
    )
