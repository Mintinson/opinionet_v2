# import re

# import pandas as pd
# import torch
# from tqdm import tqdm
# from transformers import MarianMTModel, MarianTokenizer


# class BackTranslator:
#     def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
#         print(f"Loading models on {device}...")
#         self.device = device

#         # zh -> en
#         self.zh_en_name = "Helsinki-NLP/opus-mt-zh-en"
#         self.zh_en_tokenizer = MarianTokenizer.from_pretrained(self.zh_en_name)
#         self.zh_en_model = MarianMTModel.from_pretrained(self.zh_en_name).to(device) # pyright: ignore[reportArgumentType]

#         # en -> zh
#         self.en_zh_name = "Helsinki-NLP/opus-mt-en-zh"
#         self.en_zh_tokenizer = MarianTokenizer.from_pretrained(self.en_zh_name)
#         self.en_zh_model = MarianMTModel.from_pretrained(self.en_zh_name).to(device) # pyright: ignore[reportArgumentType]
#         print("Models loaded.")

#     def translate_batch(self, texts, target_lang="en"):
#         """Perform batch translation."""
#         if not texts:
#             return []

#         if target_lang == "en":
#             tokenizer, model = self.zh_en_tokenizer, self.zh_en_model
#         else:
#             tokenizer, model = self.en_zh_tokenizer, self.en_zh_model

#         # Preprocessing: remove any special whitespace that may exist
#         texts = [t.strip() for t in texts]

#         # Tokenize
#         inputs = tokenizer(
#             texts, return_tensors="pt", padding=True, truncation=True, max_length=512
#         ).to(self.device)

#         # generate (using beam search to reduce gibberish risk, or sampling to increase diversity)
#         # Here, for robustness, we use num_beams=5; if more diversity is desired, do_sample=True can be added
#         with torch.no_grad():
#             translated = model.generate(
#                 **inputs,
#                 num_beams=3,  # Use beam search to ensure quality
#                 do_sample=True,  # Enable sampling for diversity
#                 temperature=0.8,  # Temperature controls randomness
#                 max_new_tokens=50,  # Limit generation length to prevent infinite repetition
#             )

#         decoded = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

#         if target_lang == "zh":
#             decoded = [t.replace(" ", "") for t in decoded]

#         return decoded

#     def back_translate(self, texts):
#         if not texts:
#             return []

#         # 1. Chinese -> English
#         en_texts = self.translate_batch(texts, target_lang="en")

#         # 2. English -> Chinese
#         zh_texts = self.translate_batch(en_texts, target_lang="zh")

#         return zh_texts


# # ==========================================
# # 2. Validation function (Core fix)
# # ==========================================
# def is_valid_chinese_term(original, augmented):
#     """
#     Check if the augmented term is valid.
#     1. Does not contain English letters.
#     2. Length does not explode excessively.
#     3. Not an empty string.
#     """
#     if not augmented or augmented.strip() == "":
#         return False

#     # Check for English letters (a-z, A-Z)
#     if re.search(r"[a-zA-Z]", augmented):
#         return False

#     # Check for length explosion (e.g., "good" -> "very very very good...")
#     if len(augmented) > 3 * len(original) + 2:
#         return False

#     return True


# # ==========================================
# # 3. Main augmentation process
# # ==========================================
# def augment_data(reviews_df, labels_df, translator, multiplier=1):
#     new_reviews_list = []
#     new_labels_list = []

#     # Get new starting ID
#     start_new_id = reviews_df["id"].max() + 1

#     # Group labels by ID for faster lookup
#     grouped_labels = labels_df.groupby("id")

#     print(f"Starting augmentation for {len(reviews_df)} reviews...")

#     for _ in range(multiplier):
#         for _, review_row in tqdm(reviews_df.iterrows(), total=len(reviews_df)):
#             original_id = review_row["id"]
#             original_text = review_row["Reviews"]

#             # Skip augmentation if the review has no corresponding labels
#             if original_id not in grouped_labels.groups:
#                 continue

#             current_labels = grouped_labels.get_group(original_id)

#             # --- Step A: Extract terms to translate (Aspect and Opinion) ---
#             spans = []
#             for idx, row in current_labels.iterrows():
#                 # Aspect
#                 if row["AspectTerms"] != "_" and pd.notna(row["A_start"]):
#                     spans.append(
#                         {
#                             "type": "Aspect",
#                             "text": row["AspectTerms"],
#                             "start": int(row["A_start"]),
#                             "end": int(row["A_end"]),
#                             "orig_row_idx": idx,
#                         }
#                     )
#                 # Opinion
#                 if row["OpinionTerms"] != "_" and pd.notna(row["O_start"]):
#                     spans.append(
#                         {
#                             "type": "Opinion",
#                             "text": row["OpinionTerms"],
#                             "start": int(row["O_start"]),
#                             "end": int(row["O_end"]),
#                             "orig_row_idx": idx,
#                         }
#                     )

#             # Sort by position
#             # Extract text for batch translation; positions will be recalculated later
#             spans.sort(key=lambda x: x["start"])

#             if not spans:
#                 continue

#             # --- Step B: Batch back-translation ---
#             texts_to_aug = [s["text"] for s in spans]
#             augmented_texts = translator.back_translate(texts_to_aug)

#             # --- Step C: Assemble new sentence and validate ---
#             new_text = ""
#             last_pos = 0
#             offset = 0

#             # Temporary storage for updated label data
#             # key: orig_row_idx, value: dict of updates
#             row_updates = {idx: {} for idx in current_labels.index}

#             valid_augmentation = True  # Flag to track if augmentation was successful

#             for span, aug_text in zip(spans, augmented_texts):
#                 # !!! Core Validation !!!
#                 # Fallback to original term if translation contains English or gibberish
#                 final_term = aug_text
#                 if not is_valid_chinese_term(span["text"], aug_text):
#                     final_term = span["text"]

#                 # Append unchanged parts
#                 new_text += original_text[last_pos : span["start"]]

#                 # Append new term
#                 new_text += final_term

#                 # Calculate new positions
#                 current_len = len(final_term)
#                 orig_len = span["end"] - span["start"]

#                 new_start = span["start"] + offset
#                 new_end = new_start + current_len

#                 # Record label updates
#                 rid = span["orig_row_idx"]
#                 if span["type"] == "Aspect":
#                     row_updates[rid]["AspectTerms"] = final_term
#                     row_updates[rid]["A_start"] = new_start
#                     row_updates[rid]["A_end"] = new_end
#                 else:
#                     row_updates[rid]["OpinionTerms"] = final_term
#                     row_updates[rid]["O_start"] = new_start
#                     row_updates[rid]["O_end"] = new_end

#                 # Update offset and pointer
#                 offset += current_len - orig_len
#                 last_pos = span["end"]

#             # Append remaining part of the sentence
#             new_text += original_text[last_pos:]

#             # --- Step D: Save results ---
#             # Save new review
#             new_reviews_list.append({"id": start_new_id, "Reviews": new_text})

#             # Save corresponding labels
#             for idx, row in current_labels.iterrows():
#                 new_row = row.copy()
#                 new_row["id"] = start_new_id

#                 # Apply updates
#                 if idx in row_updates:
#                     for k, v in row_updates[idx].items():
#                         new_row[k] = v

#                 # If a term is '_', its position is usually NaN or empty; keep as is
#                 # Note: If Aspect is '_' but Opinion changes, Aspect's position remains unaffected
#                 # Only non-'_' items have updates, so '_' items correctly retain original values.

#                 new_labels_list.append(new_row)

#             start_new_id += 1

#     return pd.DataFrame(new_reviews_list), pd.DataFrame(new_labels_list)


# # ==========================================
# # 4. Script execution
# # ==========================================
# if __name__ == "__main__":
#     # Read data
#     try:
#         reviews_df = pd.read_csv("data/TRAIN/Train_reviews.csv")
#         labels_df = pd.read_csv("data/TRAIN/Train_labels.csv")

#         # Instantiate translator
#         translator = BackTranslator()

#         # Perform augmentation (3x)
#         aug_reviews, aug_labels = augment_data(
#             reviews_df, labels_df, translator, multiplier=3
#         )

#         # Ensure consistent column order
#         aug_reviews = aug_reviews[reviews_df.columns]
#         aug_labels = aug_labels[labels_df.columns]

#         # Save
#         aug_reviews.to_csv("data/TRAIN/Train_reviews_augmented_makeup_3x_fixed.csv", index=False)
#         aug_labels.to_csv("data/TRAIN/Train_labels_augmented_makeup_3x_fixed.csv", index=False)

#         print("Done! Results saved.")

#     except Exception as e:
#         print(f"Error: {e}")


import os
import re

import pandas as pd
import torch
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer


# ==========================================
# 1. BackTranslator Class
# ==========================================
class BackTranslator:
    def __init__(
        self,
        device="cpu",
        zh_en_name="Helsinki-NLP/opus-mt-zh-en",
        en_zh_name="Helsinki-NLP/opus-mt-en-zh",
    ):
        """
        Initialize the BackTranslator with translation models.
        """
        print(f"Loading models on {device}...")
        self.device = device

        # Load Chinese -> English model
        print(f"Loading ZH->EN model: {zh_en_name}")
        self.zh_en_tokenizer = MarianTokenizer.from_pretrained(zh_en_name)
        self.zh_en_model = MarianMTModel.from_pretrained(zh_en_name).to(device)

        # Load English -> Chinese model
        print(f"Loading EN->ZH model: {en_zh_name}")
        self.en_zh_tokenizer = MarianTokenizer.from_pretrained(en_zh_name)
        self.en_zh_model = MarianMTModel.from_pretrained(en_zh_name).to(device)
        print("Models loaded successfully.")

    def translate_batch(self, texts, target_lang="en"):
        """
        Translate a batch of texts.
        """
        if not texts:
            return []

        # Select model and tokenizer based on target language
        if target_lang == "en":
            tokenizer, model = self.zh_en_tokenizer, self.zh_en_model
        else:
            tokenizer, model = self.en_zh_tokenizer, self.en_zh_model

        # Preprocessing
        texts = [str(t).strip() for t in texts]

        # Tokenize
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)

        # Generate translation
        # adjust parameters (temperature, top_k) to control diversity vs quality
        with torch.no_grad():
            translated = model.generate(
                **inputs,
                num_beams=3,  # Beam search for quality
                do_sample=True,  # Enable sampling for diversity
                temperature=0.8,  # Higher temperature = more diversity
                max_new_tokens=60,
            )

        decoded = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

        # Post-processing: remove spaces for Chinese output
        if target_lang == "zh":
            decoded = [t.replace(" ", "") for t in decoded]

        return decoded

    def back_translate(self, texts):
        """
        Perform back-translation: ZH -> EN -> ZH
        """
        if not texts:
            return []

        # Step 1: Chinese to English
        en_texts = self.translate_batch(texts, target_lang="en")

        # Step 2: English to Chinese
        zh_texts = self.translate_batch(en_texts, target_lang="zh")

        return zh_texts


# ==========================================
# 2. Validation Logic
# ==========================================
def is_valid_chinese_term(original, augmented):
    """
    Validate the augmented term to avoid low-quality data.
    1. Should not contain English letters (indicates translation failure).
    2. Length should not be excessively long (indicates repetition/hallucination).
    """
    if not augmented or augmented.strip() == "":
        return False

    # Check for English characters (a-z, A-Z)
    if re.search(r"[a-zA-Z]", augmented):
        return False

    # Check for length explosion (e.g., "Good" -> "Good Good Good...")
    # Allow 3x length + 2 chars buffer
    if len(augmented) > 3 * len(original) + 2:
        return False

    return True


# ==========================================
# 3. Augmentation Logic
# ==========================================
def augment_data(reviews_df, labels_df, translator, multiplier=1):
    """
    Augment data by back-translating AspectTerms and OpinionTerms.
    Re-calculates start/end indices based on length changes.
    """
    new_reviews_list = []
    new_labels_list = []

    # Start ID for new data (continue from the max existing ID)
    start_new_id = reviews_df["id"].max() + 1

    # Group labels by ID for faster access
    grouped_labels = labels_df.groupby("id")

    print(f"Starting augmentation (Multiplier: {multiplier}x)...")

    for _ in range(multiplier):
        # Iterate over each original review
        for _, review_row in tqdm(
            reviews_df.iterrows(), total=len(reviews_df), desc="Processing"
        ):
            original_id = review_row["id"]
            original_text = review_row["Reviews"]

            # Skip if no labels exist for this review
            if original_id not in grouped_labels.groups:
                continue

            current_labels = grouped_labels.get_group(original_id)

            # --- Step A: Extract terms to translate ---
            spans = []
            for idx, row in current_labels.iterrows():
                # Extract Aspect
                if row["AspectTerms"] != "_" and pd.notna(row["A_start"]):
                    spans.append(
                        {
                            "type": "Aspect",
                            "text": row["AspectTerms"],
                            "start": int(row["A_start"]),
                            "end": int(row["A_end"]),
                            "orig_row_idx": idx,
                        }
                    )
                # Extract Opinion
                if row["OpinionTerms"] != "_" and pd.notna(row["O_start"]):
                    spans.append(
                        {
                            "type": "Opinion",
                            "text": row["OpinionTerms"],
                            "start": int(row["O_start"]),
                            "end": int(row["O_end"]),
                            "orig_row_idx": idx,
                        }
                    )

            # Sort spans by start position to handle replacements sequentially
            spans.sort(key=lambda x: x["start"])

            if not spans:
                continue

            # --- Step B: Batch Back-Translation ---
            texts_to_aug = [s["text"] for s in spans]
            augmented_texts = translator.back_translate(texts_to_aug)

            # --- Step C: Reconstruct Sentence & Update Indices ---
            new_text = ""
            last_pos = 0
            offset = 0  # Tracks index shift

            # Store updates for labels: key=row_idx, value=dict of changes
            row_updates = {idx: {} for idx in current_labels.index}

            for span, aug_text in zip(spans, augmented_texts):
                # Check validity of translation
                final_term = aug_text
                if not is_valid_chinese_term(span["text"], aug_text):
                    final_term = span["text"]  # Fallback to original

                # Append text before the term
                new_text += original_text[last_pos : span["start"]]

                # Append the (possibly augmented) term
                new_text += final_term

                # Calculate new positions
                current_len = len(final_term)
                orig_len = span["end"] - span["start"]

                new_start = span["start"] + offset
                new_end = new_start + current_len

                # Record updates
                rid = span["orig_row_idx"]
                if span["type"] == "Aspect":
                    row_updates[rid]["AspectTerms"] = final_term
                    row_updates[rid]["A_start"] = new_start
                    row_updates[rid]["A_end"] = new_end
                else:
                    row_updates[rid]["OpinionTerms"] = final_term
                    row_updates[rid]["O_start"] = new_start
                    row_updates[rid]["O_end"] = new_end

                # Update offset and pointer
                offset += current_len - orig_len
                last_pos = span["end"]

            # Append remaining text
            new_text += original_text[last_pos:]

            # --- Step D: Save New Data ---
            new_reviews_list.append({"id": start_new_id, "Reviews": new_text})

            for idx, row in current_labels.iterrows():
                new_row = row.copy()
                new_row["id"] = start_new_id

                # Apply updates if any
                if idx in row_updates:
                    for k, v in row_updates[idx].items():
                        new_row[k] = v

                new_labels_list.append(new_row)

            start_new_id += 1

    return pd.DataFrame(new_reviews_list), pd.DataFrame(new_labels_list)


# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    # --- Configuration Parameters ---
    INPUT_REVIEWS_FILE = "data/TRAIN/Train_reviews.csv"
    INPUT_LABELS_FILE = "data/TRAIN/Train_labels.csv"

    OUTPUT_REVIEWS_FILE = "data/TRAIN/Train_reviews_augmented_backtranslate.csv"
    OUTPUT_LABELS_FILE = "data/TRAIN/Train_labels_augmented_backtranslate.csv"

    AUGMENT_MULTIPLIER = 1  # How many augmented copies to generate

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Models (HuggingFace paths)
    ZH_EN_MODEL = "Helsinki-NLP/opus-mt-zh-en"
    EN_ZH_MODEL = "Helsinki-NLP/opus-mt-en-zh"

    # --- 1. Load Data ---
    print("Reading input files...")
    if not os.path.exists(INPUT_REVIEWS_FILE) or not os.path.exists(INPUT_LABELS_FILE):
        print("Error: Input files not found in current directory.")
        exit(1)

    reviews_df = pd.read_csv(INPUT_REVIEWS_FILE)
    labels_df = pd.read_csv(INPUT_LABELS_FILE)

    # --- 2. Initialize Translator ---
    translator = BackTranslator(
        device=DEVICE, zh_en_name=ZH_EN_MODEL, en_zh_name=EN_ZH_MODEL
    )

    # --- 3. Perform Augmentation ---
    aug_reviews_df, aug_labels_df = augment_data(
        reviews_df, labels_df, translator, multiplier=AUGMENT_MULTIPLIER
    )

    # --- 4. Combine with Original Data ---
    print("Combining original and augmented data...")
    final_reviews_df = pd.concat([reviews_df, aug_reviews_df], ignore_index=True)
    final_labels_df = pd.concat([labels_df, aug_labels_df], ignore_index=True)

    # Ensure column order
    final_reviews_df = final_reviews_df[reviews_df.columns]
    final_labels_df = final_labels_df[labels_df.columns]

    # --- 5. Save Results ---
    final_reviews_df.to_csv(OUTPUT_REVIEWS_FILE, index=False, encoding="utf-8")
    final_labels_df.to_csv(OUTPUT_LABELS_FILE, index=False, encoding="utf-8")

    print("=" * 40)
    print("Augmentation Complete!")
    print(f"Original Reviews: {len(reviews_df)}")
    print(f"Augmented Reviews: {len(aug_reviews_df)}")
    print(f"Total Reviews: {len(final_reviews_df)}")
    print(f"Files saved to:\n  - {OUTPUT_REVIEWS_FILE}\n  - {OUTPUT_LABELS_FILE}")
    print("=" * 40)
