from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from transformers import BertConfig, BertModel


def margin_negsub_bce_with_logits(
    logits: torch.Tensor, target: torch.Tensor, margin: float = 0.1, neg_sub: float = 0.5
) -> torch.Tensor:
    """
    Binary cross entropy with margin-based negative subsampling.

    Args:
        logits: Model predictions (logits)
        target: Target labels
        margin: Margin threshold for keeping samples
        neg_sub: Negative subsample weight

    Returns:
        Loss value
    """
    y = torch.sigmoid(logits)
    keep_mask = (torch.abs(target - y) > margin).float()
    pos_keep = keep_mask * target

    neg_keep = keep_mask - pos_keep

    loss_pos = -pos_keep * torch.log(torch.clamp(y, min=1e-8))
    loss_neg = -neg_keep * neg_sub * torch.log(torch.clamp(1 - y, min=1e-10, max=1.0))
    loss = (loss_pos + loss_neg).mean()

    return loss


def focal_bce_with_logits(
    logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    grad = torch.abs(target - probs) ** gamma
    grad /= grad.mean()
    loss = grad * F.binary_cross_entropy(probs, target, reduction="none")
    return loss.mean()


def focal_ce_with_logits(
    logit: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -1,
    alpha: Optional[float] = None,
    gamma: float = 2.0,
    smooth: float = 0.05,
) -> torch.Tensor:
    """
    Focal cross entropy loss with label smoothing.

    Args:
        logit: Model predictions (logits)
        target: Target labels
        ignore_index: Index to ignore in loss calculation
        alpha: Weighting factor
        gamma: Focusing parameter
        smooth: Label smoothing factor

    Returns:
        Loss value
    """
    num_classes = logit.size(1)
    logit = F.softmax(logit, dim=1)

    if alpha is None:
        alpha = 1.0

    if logit.dim() > 2:
        logit = logit.view(logit.size(0), logit.size(1), -1)
        logit = logit.transpose(1, 2).contiguous()
        logit = logit.view(-1, logit.size(-1))

    target = target.view(-1)
    logit = logit[target != ignore_index]
    target = target[target != ignore_index]

    target_onehot = F.one_hot(target, num_classes=num_classes).float()
    if smooth:
        target_onehot = torch.clamp(target_onehot, smooth, 1.0 - smooth)

    pt = (target_onehot * logit).sum(1) + 1e-10
    logpt = pt.log()
    loss = -alpha * torch.pow((1 - pt), gamma) * logpt
    return loss.mean()


class OpinionNet(nn.Module):
    def __init__(
        self,
        config: BertConfig,
        hidden_size: int = 100,
        dropout_prob: float = 0.3,
        version: str = "large",
        focal: bool = False,
        num_common_categories: int = 7,
        num_laptop_categories: int = 11,
        num_makeup_categories: int = 13,
        num_polarities: int = 3,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.version = version
        self.focal = focal
        self.hidden_size = hidden_size
        self.bert_hidden_size = config.hidden_size

        # BERT encoder
        self.bert = BertModel(config)

        # Aspect start/end layers
        self.w_as11 = nn.Linear(self.bert_hidden_size, hidden_size)
        self.w_as12 = nn.Linear(self.bert_hidden_size, hidden_size)
        self.w_ae21 = nn.Linear(self.bert_hidden_size, hidden_size)
        self.w_ae22 = nn.Linear(self.bert_hidden_size, hidden_size)

        # Opinion start/end layers
        self.w_os11 = nn.Linear(self.bert_hidden_size, hidden_size)
        self.w_os12 = nn.Linear(self.bert_hidden_size, hidden_size)
        self.w_oe21 = nn.Linear(self.bert_hidden_size, hidden_size)
        self.w_oe22 = nn.Linear(self.bert_hidden_size, hidden_size)

        # Output layers for pointer network
        self.w_as2 = nn.Linear(hidden_size, 1)
        self.w_ae2 = nn.Linear(hidden_size, 1)
        self.w_os2 = nn.Linear(hidden_size, 1)
        self.w_oe2 = nn.Linear(hidden_size, 1)

        # Objectiveness layer
        self.w_obj = nn.Linear(self.bert_hidden_size, 1)

        # Category and polarity layers
        self.w_common = nn.Linear(self.bert_hidden_size, num_common_categories)
        self.w_makeup = nn.Linear(
            self.bert_hidden_size, num_makeup_categories - num_common_categories
        )
        self.w_laptop = nn.Linear(
            self.bert_hidden_size, num_laptop_categories - num_common_categories
        )
        self.w_p = nn.Linear(self.bert_hidden_size, num_polarities)

        # MLM head for pretraining (using BERT's vocab size)
        self.mlm_head = nn.Linear(self.bert_hidden_size, config.vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout_prob)

        # Move to device if specified
        if device is not None:
            self.to(device)

    def forward_mlm(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        masked_ml_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for Masked Language Modeling (pretraining).

        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_len]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]
            masked_ml_labels (Optional[torch.Tensor], optional): Labels for masked tokens [batch_size, seq_len], -1 for non-masked.
              Defaults to None.

        Returns:
            torch.Tensor: MLM loss if labels provided, otherwise prediction scores
        """
        # Get BERT encodings
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        sequence_output: torch.Tensor = (
            outputs.last_hidden_state
        )  # (batch_size, seq_len, bert_hidden_size)

        # MLM prediction scores
        prediction_scores: torch.Tensor = self.mlm_head(
            sequence_output
        )  # [batch_size, seq_len, vocab_size]

        if masked_ml_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, prediction_scores.size(-1)),
                masked_ml_labels.view(-1),
            )
            return masked_lm_loss
        else:
            return prediction_scores

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_mask: torch.Tensor,
        data_type: Literal["laptop", "makeup"] = "laptop",
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Performs the forward pass of OpinioNet for joint aspect-opinion extraction.

        The model processes input through a BERT encoder and multiple task-specific heads:
        - Pointer Networks: Identify aspect/opinion boundaries using 2D logit matrices.
        - Objectiveness: Binary classification for each token.
        - Category: Domain-specific multi-class classification (Common + Laptop/Makeup).
        - Polarity: Sentiment classification.

        Args:
            input_ids: Token IDs [batch_size, seq_len].
            attention_mask: BERT attention mask [batch_size, seq_len].
            token_mask: Mask for valid tokens (excluding padding/special tokens) [batch_size, seq_len].
            data_type: Domain selector for the category head ("laptop" or "makeup").

        Returns:
            A tuple of (probabilities, logits), where each is a list of 7 tensors:
            0. as: Aspect Start [batch_size, seq_len, seq_len]
            1. ae: Aspect End [batch_size, seq_len, seq_len]
            2. os: Opinion Start [batch_size, seq_len, seq_len]
            3. oe: Opinion End [batch_size, seq_len, seq_len]
            4. obj: Objectiveness [batch_size, seq_len]
            5. c: Category [batch_size, seq_len, num_categories]
            6. p: Polarity [batch_size, seq_len, num_polarities]
        """

        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        sequence_output = (
            outputs.last_hidden_state
        )  # (batch_size, seq_len, bert_hidden_size)

        # Aspect start/end logits using pointer network
        as_logits = self._compute_pointer_logits(
            sequence_output, self.w_as11, self.w_as12, self.w_as2
        )

        ae_logits = self._compute_pointer_logits(
            sequence_output, self.w_ae21, self.w_ae22, self.w_ae2
        )

        # Opinion start/end logits using pointer network
        os_logits = self._compute_pointer_logits(
            sequence_output, self.w_os11, self.w_os12, self.w_os2
        )
        oe_logits = self._compute_pointer_logits(
            sequence_output, self.w_oe21, self.w_oe22, self.w_oe2
        )

        # Objectiveness logits
        obj_logits: torch.Tensor = self.w_obj(self.dropout(sequence_output)).squeeze(-1)
        # Category logits
        common_logits = self.w_common(self.dropout(sequence_output))

        if data_type == "laptop":
            special_logits = self.w_laptop(self.dropout(sequence_output))
        else:
            special_logits = self.w_makeup(self.dropout(sequence_output))
        c_logits = torch.cat([common_logits, special_logits], dim=-1)

        # Polarity logits
        p_logits = self.w_p(self.dropout(sequence_output))

        # Apply masks
        token_mask_with_cls = token_mask.clone()
        token_mask_with_cls[:, 0] = 1  # Include [CLS] token

        # Pointer mask: [batch_size, seq_len, seq_len]
        pointer_mask = token_mask_with_cls.unsqueeze(2) * token_mask_with_cls.unsqueeze(1)
        pointer_mask[:, 0, :] = 0  # Exclude [CLS] as start position
        pointer_mask = (1 - pointer_mask).bool()

        # Token mask: [batch_size, seq_len]
        token_mask_bool = (1 - token_mask).bool()

        # Mask logits
        as_logits = as_logits.masked_fill(pointer_mask, -1e5)
        ae_logits = ae_logits.masked_fill(pointer_mask, -1e5)
        os_logits = os_logits.masked_fill(pointer_mask, -1e5)
        oe_logits = oe_logits.masked_fill(pointer_mask, -1e5)
        obj_logits = obj_logits.masked_fill(token_mask_bool, -1e5)

        # Compute probabilities
        probs = [
            F.softmax(as_logits, dim=-1),
            F.softmax(ae_logits, dim=-1),
            F.softmax(os_logits, dim=-1),
            F.softmax(oe_logits, dim=-1),
            torch.sigmoid(obj_logits),
            F.softmax(c_logits, dim=-1),
            F.softmax(p_logits, dim=-1),
        ]

        logits = [
            as_logits,
            ae_logits,
            os_logits,
            oe_logits,
            obj_logits,
            c_logits,
            p_logits,
        ]

        return probs, logits

    def compute_loss(
        self,
        logits: List[torch.Tensor],
        targets: List[torch.Tensor],
        neg_sub: bool = False,
    ) -> torch.Tensor:
        """Compute training loss."""
        ce_fn = focal_ce_with_logits if self.focal else F.cross_entropy

        # Sum CE losses for all outputs except objectiveness (index 4)
        loss = sum(
            ce_fn(logit.permute(0, 2, 1), tgt, ignore_index=-1)
            for i, (logit, tgt) in enumerate(zip(logits, targets))
            if i != 4
        )

        # Add objectiveness loss
        loss += margin_negsub_bce_with_logits(
            logits[4], targets[4], neg_sub=0.5 if neg_sub else 1.0
        )

        return loss

    def generate_candidates(
        self, probs: List[torch.Tensor], threshold: float = 0.01
    ) -> List[List[Tuple[Tuple[int, ...], float]]]:
        """Generate opinion candidates from model predictions.

        Args:
            probs (List[torch.Tensor]): Model probability outputs
            threshold (float, optional): Confidence threshold. Defaults to 0.01.

        Returns:
            List[List[Tuple[Tuple[int, ...], float]]]: List of candidates for each sample in batch
        """
        as_probs, ae_probs, os_probs, oe_probs, obj_probs, c_probs, p_probs = probs

        # Vectorized prediction and score computation
        prob_tensors = [as_probs, ae_probs, os_probs, oe_probs, c_probs, p_probs]
        preds_scores = [p.max(dim=-1) for p in prob_tensors]

        scores = [x[0] for x in preds_scores]
        preds = [x[1] for x in preds_scores]
        as_p, ae_p, os_p, oe_p, c_p, p_p = preds

        # Vectorized confidence computation
        confidence = obj_probs
        for s in scores:
            confidence = confidence * s

        # Vectorized validity checks
        # 1. Start <= End
        valid_span = (as_p <= ae_p) & (os_p <= oe_p)
        # 2. No overlap: min(ae, oe) < max(as, os)
        no_overlap = torch.min(ae_p, oe_p) < torch.max(as_p, os_p)

        is_valid = valid_span & no_overlap

        # Prepare data on CPU
        confidence = confidence.cpu()
        is_valid = is_valid.cpu()
        # Stack predictions: [batch, seq, 6] -> (as, ae, os, oe, c, p)
        preds_stack = torch.stack(preds, dim=-1).cpu()

        results = []
        for b in range(confidence.size(0)):
            # Filter valid candidates
            mask = is_valid[b]
            if not mask.any():
                results.append([])
                continue

            b_conf = confidence[b][mask]
            b_preds = preds_stack[b][mask]

            # Sort by confidence descending
            sorted_conf, sorted_idx = torch.sort(b_conf, descending=True)
            sorted_preds = b_preds[sorted_idx]

            # Convert to python objects
            b_conf_list = sorted_conf.tolist()
            b_preds_list = sorted_preds.tolist()

            batch_res = []
            for pred, conf in zip(b_preds_list, b_conf_list):
                if batch_res and conf < threshold:
                    break
                batch_res.append((tuple(pred), conf))

            results.append(batch_res)

        return results

    @staticmethod
    def nms_filter(
        results: List[List[Tuple[Tuple[int, ...], float]]], threshold: float = 0.1
    ) -> List[List[Tuple[Tuple[int, ...], float]]]:
        """Apply non-maximum suppression to filter overlapping results."""

        def _filter_sample(
            candidates: List[Tuple[Tuple[int, ...], float]],
        ) -> List[Tuple[Tuple[int, ...], float]]:
            kept = []
            for cand in sorted(candidates, key=lambda x: -x[1]):
                if kept and cand[1] < threshold:
                    break

                # Check overlap: aspect overlap AND opinion overlap
                # cand[0] indices: 0=as, 1=ae, 2=os, 3=oe
                c_s = cand[0]
                if not any(
                    min(k[0][1], c_s[1]) >= max(k[0][0], c_s[0])
                    and min(k[0][3], c_s[3]) >= max(k[0][2], c_s[2])
                    for k in kept
                ):
                    kept.append(cand)
            return kept

        return [_filter_sample(res) for res in results]

    def _compute_pointer_logits(
        self, sequence_output: torch.Tensor, w1a: nn.Linear, w1b: nn.Linear, w2: nn.Linear
    ) -> torch.Tensor:
        """Compute pointer network logits.

        (leaking_relu(x * w1a + y * w1b)) * W3

        Args:
            sequence_output (torch.Tensor): BERT output [batch_size, seq_len, hidden]
            w1a (nn.Linear): First layer weights for pointer network
            w1b (nn.Linear): First layer weights for pointer network
            w2 (nn.Linear): Output layer

        Returns:
            torch.Tensor: output tensor with shape [batch_size, seq_len, seq_len]
        """
        # [batch_size, seq_len, 1, hidden] + [batch_size, 1, seq_len, hidden]
        # -> [batch_size, seq_len, seq_len, hidden]
        hidden = F.leaky_relu(
            w1a(self.dropout(sequence_output)).unsqueeze(2)
            + w1b(self.dropout(sequence_output)).unsqueeze(1)
        )
        # -> [batch_size, seq_len, seq_len]
        logits = w2(hidden).squeeze(-1)
        return logits


__all__ = [
    "OpinionNet",
    "margin_negsub_bce_with_logits",
    "focal_bce_with_logits",
    "focal_ce_with_logits",
]
