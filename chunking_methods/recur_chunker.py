import sys
import os
import time
import copy
import math
import numpy as np
import nltk
from nltk.tokenize import PunktSentenceTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from .cover_constrained import max_sum_boundaries

CONFIG_DEFAULT = {  # The default may somewhat change after evaluations and ablation studies
    'model_name': 'intfloat/multilingual-e5-small',
    'window_make_embs': 512,
    'normalize_cos_boundaries': False,
    'window_size_cos': 1,
    'weight_balance_spaces_embs': -1,  # Good value, maybe: 1.0
    'weight_space_margin': 3.0,
    'space_weight_multiplier': 1.0,
    'spaces_weights': {
        " ": 1.0,
        "\t": 2.0,  # Tab	Indentation, column alignment	Medium, less absolute than newline
        "\n": 4.0,  # Line Feed (LF, Unix newline)	End of line, separates logical units	High
        "\r": 4.0,
        # Carriage Return (CR)	Legacy newline (classic Mac), paired with LF in Windows (\r\n)	High (same role as LF when used)
        "\v": 3.0,
        # Vertical Tab	Rare, used for spacing in some systems	Medium–High (line separation, but less standardized)
        "\f": 8.0,  # Form Feed	Page breaks in printers, some markup	Very High (document-level separation)
        # "\u00A0": 2.0, # Non-breaking space (NBSP)	Prevents line break between words	Medium (important in layout but not structural in data)
        "\u2028": 4.0,  # Explicit Unicode newline High
        "\u2029": 6.0,  # Paragraph Separator	Explicit Unicode paragraph break	Very High (stronger than newline)
    },
    # 'space_weight_other': 1.0,
    'sentence_weight': 3.0,
    'chunk_size_max': 600,  # not restricted from above
    'recurs_size_max': -1,  # Set to a size < 'chunk_size_max' if want to apply cover after the recursive splits
    'ratio_cover': 1.8,
    'relax_splits_percentile': -1,  # percentile of boundaries strength, to remove weaker splits
    'out_config_chunker': True,
    'out_chunks_of_text': False,
    'out_embs_chunks': True,  # Set to False if want only chunk boundaries (for review, evaluations)
    'out_embs_text': False,  # Set to True if want also the embedding of the whole text
    'out_boundaries_weights': False,
    'check_sents_toks_spans': False,
}


class ChunkerTok(nn.Module):
    def __init__(self, config=CONFIG_DEFAULT):
        """The given dictionary config must have 'model_name' and
            either 'path_full_model' or 'path_state_dict'
        """
        super(ChunkerTok, self).__init__()
        if not config:
            config = {}
        self.config = config
        self.device = config.get('device', 'cpu')
        self.model_name = config.get('model_name', 'intfloat/multilingual-e5-small')
        self.model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.window_make_embs = config.get(
            'window_make_embs', self.tokenizer.model_max_length) - 2  # wrap tokens
        self.normalize_cos_boundaries = config.get('normalize_cos_boundaries')
        self.window_size_cos = config.get('window_size_cos', 1)
        self.weight_balance_spaces_embs = config.get('weight_balance_spaces_embs', -1)
        self.weight_space_margin = config.get('weight_space_margin', -1)
        self.space_weight_multiplier = config.get('space_weight_multiplier', 1.0)
        self.spaces_weights = config.get('spaces_weights')  # map space type -> weight
        self.space_weight_other = config.get('space_weight_other', 1.0)  # for unidentified spaces
        self.sentence_weight = config.get('sentence_weight')  # weight of a sentence boundary
        self.chunk_size_max = config.get('chunk_size_max', 600)
        self.recurs_size_max = config.get('recurs_size_max', 299)
        self.recurs_size_min = config.get('recurs_size_min', -1)
        self.threshold_dev = config.get('threshold_dev', -1)
        if self.recurs_size_max > 0 and self.recurs_size_max < self.chunk_size_max:
            self.do_cover = True
            ratio_cover_max = self.chunk_size_max // self.recurs_size_max
            assert ratio_cover_max >= 2
            self.ratio_cover = config.get('ratio_cover', 1)
            assert self.ratio_cover < ratio_cover_max
        else:
            self.do_cover = False
            self.recurs_size_max = self.chunk_size_max
        self.recurs_apply_margin_min = config.get('recurs_apply_margin_min', -1)
        if self.recurs_apply_margin_min == -1:
            self.recurs_apply_margin_min = round(0.13 * self.chunk_size_max)
        self.recurs_margin = config.get('recurs_margin', -1)
        if self.recurs_margin == -1:
            self.recurs_margin = self.chunk_size_max // 20
        self.relax_splits_percentile = config.get('relax_splits_percentile', -1)
        self.check_sents_toks_spans = config.get('check_sents_toks_spans', False)
        self.out_chunks_and_seps_of_text = config.get('out_chunks_and_seps_of_text', False)
        self.out_chunks_of_text = config.get('out_chunks_of_text', True)
        self.out_chunkbounds_tokens = config.get('out_chunkbounds_tokens', False)
        self.out_embs_chunks = config.get('out_embs_chunks', False)
        self.out_embs_text = config.get('out_embs_text', False)
        self.out_boundaries_weights = config.get('out_boundaries_weights', False)
        self.out_config_chunker = config.get('out_config_chunker', False)
        self.sent_tokenizer = PunktSentenceTokenizer()
        if self.space_weight_multiplier != 1.0:
            self.spaces_weights = {
                k: v * self.space_weight_multiplier for k, v in self.spaces_weights.items()}
            self.space_weight_other *= self.space_weight_multiplier
            if self.sentence_weight is not None:
                self.sentence_weight *= self.space_weight_multiplier
        self.id_tok_beg = torch.tensor(self.tokenizer.cls_token_id)  # Convenient
        self.id_tok_end = torch.tensor(self.tokenizer.sep_token_id)
        self.tok0 = self.id_tok_beg.unsqueeze(0)
        self.tok1 = self.id_tok_end.unsqueeze(0)

    def get_chunks_embs(self, text, id_text=None, algo_name='chunk_tok'):
        text = text.strip()
        text_info = self.split_text_to_sentences(text)
        encoding = self.tokenizer(
            text, return_offsets_mapping=True, return_tensors="pt", truncation=False)
        tokens = encoding["input_ids"][0][1:-1]  # without the wrapping tokens
        tokens_spans = encoding["offset_mapping"][0][1:-1]
        sentences_boundaries = self.assign_tokens_to_sentences(
            tokens_spans=tokens_spans, sentences_spans=text_info['spans'])
        embs = self.get_embeddings_of_tokens(tokens, sentences_boundaries)
        boundaries_embs_info = self.grade_tokenboundaries_by_embs(embs)
        boundaries = boundaries_embs_info['boundaries']
        avg_boundaries_embs = boundaries_embs_info['avg']
        self.grade_tokenboundaries_by_spaces(
            boundaries_embs=boundaries,
            avg_boundaries_embs=avg_boundaries_embs,
            text=text,
            tokens_spans=tokens_spans,
            sentences_boundaries=sentences_boundaries)
        boundaries = boundaries.numpy()
        if self.recurs_size_min > 0:
            if self.threshold_dev > 0:
                ixs_boundaries = self.split_to_chunks_recursively_with_threshold_stdev(
                    boundaries)
            else:
                ixs_boundaries = self.split_to_chunks_recursively_with_stop(boundaries)
        else:
            ixs_boundaries = self.split_to_chunks_recursively(boundaries)
        if ixs_boundaries:
            assert ixs_boundaries[0] <= self.recurs_size_max
            assert len(boundaries) - ixs_boundaries[-1] <= self.recurs_size_max + 1
            if self.relax_splits_percentile > 0:
                chunk_boundaries = self.relax_chunksplits(boundaries, ixs_boundaries)
            else:
                chunk_boundaries = ixs_boundaries
            boundaries_weights = boundaries[chunk_boundaries]
        else:
            chunk_boundaries, boundaries_weights = [], []
        if self.do_cover:
            w_small = -1.0e10
            chunk_boundaries_use = np.insert(chunk_boundaries, 0, 0)
            chunk_boundaries_use = np.append(chunk_boundaries_use, len(boundaries))
            boundaries_weights_use = np.insert(boundaries_weights, 0, w_small)
            boundaries_weights_use = np.append(boundaries_weights_use, w_small)
            out_cover = self.select_boundaries_by_cover(
                boundaries_weights_use, chunk_boundaries_use)
            if out_cover[0] is None and out_cover[1] is None:
                chunk_boundaries_final = chunk_boundaries
            else:
                chunk_boundaries_final = [chunk_boundaries_use[i] for i in out_cover[1]]
                if self.out_boundaries_weights:
                    boundaries_weights = [boundaries[i].item() for i in chunk_boundaries_final]
        else:
            chunk_boundaries_final = chunk_boundaries
        chunks_info = self.split_text_by_tokenboundaries(
            text, tokens_spans, chunk_boundaries_final)
        result = {'chunk_ends': chunks_info['chunk_ends']}
        if algo_name is not None:
            result['algo_name'] = algo_name
        if id_text is not None:
            result['id_text'] = id_text
        if self.out_config_chunker:
            result['config'] = self.get_config_chunker()
        if self.out_chunkbounds_tokens:  # for convenience, ready for json:
            result['chunkbounds_tokens'] = [int(i) for i in chunk_boundaries_final]
            result['chunkbounds_tokens'].append(int(len(tokens_spans)))
        if self.out_chunks_and_seps_of_text:
            chunks = chunks_info['chunks']
            result['separators'] = []
            for chunk1, chunk2 in zip(chunks[:-1], chunks[1:]):
                sep = get_separator_of_chunks(chunk1, chunk2)
                result['separators'].append(sep)
            result['chunks'] = [s.strip() for s in chunks]
        elif self.out_chunks_of_text:
            result['chunks'] = chunks_info['chunks']
        if self.out_boundaries_weights:
            result['boundaries_weights'] = boundaries_weights
        if self.out_embs_chunks:
            result['embs_chunks'] = self.get_chunk_embeddings(embs, chunk_boundaries)
        if self.out_embs_text:
            result['embs_text'] = torch.stack(embs).mean(dim=0).tolist()
        return result

    def select_boundaries_by_cover(self, boundaries_weights, boundaries_ixs):
        K = int(len(boundaries_ixs) / self.ratio_cover)
        if K < 1 or K >= int(len(boundaries_ixs)):
            return None, None
        sum_best, boundaries_selected = max_sum_boundaries(
            boundaries_weights, boundaries_ixs, K=K, L=self.chunk_size_max)
        return sum_best, boundaries_selected

    def split_to_chunks_recursively(self, arr, ix_beg=0):
        n = len(arr)
        if n == 0 or n <= self.recurs_size_max:
            return []
        if n >= self.recurs_apply_margin_min:
            max_idx = np.argmax(arr[self.recurs_margin:n - self.recurs_margin])
            max_idx += self.recurs_margin
        else:
            max_idx = np.argmax(arr)
        global_idx = ix_beg + max_idx
        left_splits = self.split_to_chunks_recursively(arr[:max_idx], ix_beg)
        right_splits = self.split_to_chunks_recursively(arr[max_idx + 1:], global_idx + 1)
        return left_splits + [global_idx] + right_splits

    def split_to_chunks_recursively_with_threshold_stdev(
            self, arr, ix_beg=0, threshold=-1):
        n = len(arr)
        if n == 0 or n <= self.recurs_size_min:
            return []
        if n >= self.recurs_apply_margin_min:
            max_idx = np.argmax(arr[self.recurs_margin:n - self.recurs_margin])
            max_idx += self.recurs_margin
        else:
            max_idx = np.argmax(arr)
        max_val = arr[max_idx]
        global_idx = ix_beg + max_idx
        if threshold < 0:
            avg = np.mean(arr)
            dev = np.std(arr)
            threshold = avg + self.threshold_dev * dev
        if n <= self.recurs_size_max:  # Already can do stopping
            if max_val < threshold:
                return []
        left_splits = self.split_to_chunks_recursively_with_threshold_stdev(
            arr[:max_idx], ix_beg, threshold=threshold)
        right_splits = self.split_to_chunks_recursively_with_threshold_stdev(
            arr[max_idx + 1:], global_idx + 1, threshold=threshold)
        return left_splits + [global_idx] + right_splits

    def split_to_chunks_recursively_with_stop(
            self, arr, ix_beg=0, prev_max_val=None, prev_ratio=None):
        n = len(arr)
        if n == 0 or n <= self.recurs_size_min:
            return []
        if n >= self.recurs_apply_margin_min:
            max_idx = np.argmax(arr[self.recurs_margin:n - self.recurs_margin])
            max_idx += self.recurs_margin
        else:
            max_idx = np.argmax(arr)
        max_val = arr[max_idx]
        global_idx = ix_beg + max_idx
        if prev_max_val is not None:  # checking the ratio
            ratio = max_val / prev_max_val
            if n <= self.recurs_size_max:  # Already can do stopping
                if prev_ratio is not None and ratio > prev_ratio:
                    return []  # Ratio increased → stop splitting here
        else:
            ratio = None
        left_splits = self.split_to_chunks_recursively_with_stop(
            arr[:max_idx], ix_beg, prev_max_val=max_val, prev_ratio=ratio)
        right_splits = self.split_to_chunks_recursively_with_stop(
            arr[max_idx + 1:], global_idx + 1, prev_max_val=max_val, prev_ratio=ratio)
        return left_splits + [global_idx] + right_splits

    def relax_chunksplits(self, arr, ixs):
        values = arr[ixs]
        threshold = np.percentile(values, self.relax_splits_percentile)
        i_use = np.where(values >= threshold)[0]
        ixs_use = [ixs[i] for i in i_use]
        return ixs_use

    def split_text_by_tokenboundaries(self, text, tokens_spans, chunk_boundaries):
        chunks, chunk_ends, ix_chunk_beg = [], [], 0
        for ix_token_last in chunk_boundaries:
            ix_chunk_end = tokens_spans[ix_token_last][1]
            chunk_ends.append(ix_chunk_end.item())
            chunk = text[ix_chunk_beg:ix_chunk_end]
            chunks.append(chunk)
            ix_chunk_beg = tokens_spans[ix_token_last + 1][0]
        chunk = text[ix_chunk_beg:]  # The last chunk
        chunks.append(chunk)
        chunk_ends.append(tokens_spans[-1][1].item())  # for convenience, must be ~len(text)
        return {'chunk_ends': chunk_ends, 'chunks': chunks}

    def get_chunk_embeddings(self, embs, chunk_boundaries):
        chunks_embs, ix_chunk_beg = [], 0
        for ix_token_last in chunk_boundaries:
            embs_chunk = embs[ix_chunk_beg:ix_token_last + 1]
            avg_embs = torch.stack(embs_chunk).mean(dim=0).tolist()
            chunks_embs.append(avg_embs)
            ix_chunk_beg = ix_token_last + 1
        embs_chunk = embs[ix_chunk_beg:]
        avg_embs = torch.stack(embs_chunk).mean(dim=0).tolist()
        chunks_embs.append(avg_embs)
        return chunks_embs

    def split_text_to_sentences(self, text):
        spans = list(self.sent_tokenizer.span_tokenize(text))
        sents, borders = [], []
        for i_span, span in enumerate(spans):
            sent = text[span[0]: span[1]]
            sents.append(sent)
            if i_span > 0:
                border = text[spans[i_span - 1][1]: span[0]]
                borders.append(border)
        borders.append('')
        assert len(sents) == len(spans)
        assert len(borders) == len(spans)
        return {'sents': sents, 'borders': borders, 'spans': spans}

    def check_compatibility_spans_tokens_vs_sentences(self, tokens_spans, sentences_spans):
        for sent1, sent2 in zip(sentences_spans[:-1], sentences_spans[1:]):
            for tspan in tokens_spans:
                assert not (tspan[0] < sent1[1] and tspan[1] >= sent2[0])  # overlaps 2 sents
                assert not (tspan[0] >= sent1[1] and tspan[1] < sent2[0])  # outside both

    def assign_tokens_to_sentences(self, tokens_spans, sentences_spans):
        """"
        Arguments:
          tokens_spans List[(int,int)]: List of tuples corresponding to tokens of the text
            (The wrapping tokens SLS and SEP are excluded)
            Each tuple is two elements: text character indexes for token's beging & end
          sentences_spans List[(int,int)]: List of tuples corresponding to all sentences
            Each tuple is two elements: text character indexes for sentence's beging & end
        Returns:
          sentences_boundaries List[int]: Each is an index of the sentence's first token
            The very first index=0 is not included
        """
        if self.check_sents_toks_spans:
            self.check_compatibility_spans_tokens_vs_sentences(tokens_spans, sentences_spans)
        A = np.array([s[0] for s in sentences_spans])  # ix of the first character
        B = np.array([s[1] - 1 for s in tokens_spans])  # ix of the last character
        bounds = np.searchsorted(B, A, side='left')  # seek right closest B[j]>=A[i]
        bounds = np.where(bounds < len(B), bounds, -1)  # If pos==len(B), => no element >= A[i]
        assert bounds[0] == 0
        bounds = bounds[1:]  # not including the very first token, index=0
        assert len(bounds) == len(sentences_spans) - 1
        return bounds

    def get_embeddings_of_tokens(self, tokens, sentences_boundaries):
        embs = []
        sents_ends = np.append(sentences_boundaries, len(tokens))
        beg = 0
        window_beg, window_end = 0, 0
        for end in sents_ends:
            sent_beg, sent_end = beg, end
            beg = end
            if sent_end - sent_beg <= self.window_make_embs:  # try add sentence if fits:
                if window_end - window_beg + sent_end - sent_beg <= self.window_make_embs:
                    window_end += sent_end - sent_beg  # added the sentence
                else:  # make embeddings and start new window:
                    if window_end > window_beg:
                        self.get_embs_simple(embs, tokens, window_beg, window_end)
                    window_beg = window_end
                    window_end = sent_end
            else:  # the sentence is too long, split it:
                if window_end > window_beg:
                    self.get_embs_simple(embs, tokens, window_beg, window_end)
                window_beg = window_end
                for i in range(0, sent_end - sent_beg, self.window_make_embs):
                    window_end += self.window_make_embs
                    window_end = min(sent_end, window_end)
                    self.get_embs_simple(embs, tokens, window_beg, window_end)
                    window_beg = window_end
        if window_end > window_beg:
            self.get_embs_simple(embs, tokens, window_beg, window_end)
        assert len(embs) == len(tokens)
        return embs

    def get_embs_simple(self, embs, tokens, window_beg, window_end):
        ids_toks = torch.cat((self.tok0, tokens[window_beg:window_end], self.tok1))
        with torch.no_grad():
            outputs = self.model(input_ids=ids_toks.unsqueeze(0))
        embs_add = outputs.last_hidden_state[0][1:-1]  # no CLS and SEP
        embs.extend(embs_add)

    def grade_tokenboundaries_by_embs(self, embs):
        embs_normed = torch.stack(embs, dim=0)
        embs_normed = F.normalize(embs_normed, dim=1)
        K = self.window_size_cos
        if K == 1:
            boundaries = (embs_normed[:-1] * embs_normed[1:]).sum(dim=1)
        else:
            N, D = embs_normed.shape
            cumsum = torch.cat([torch.zeros(1, D), embs_normed.cumsum(dim=0)], dim=0)  # (N+1,D)
            idx = torch.arange(1, N)  # N-1 boundaries: 1..N-1
            l0 = (idx - K).clamp(min=0)  # Left windows:
            l1 = idx
            left_sums = cumsum[l1] - cumsum[l0]
            left_counts = (l1 - l0).unsqueeze(1)
            left_means = left_sums / left_counts
            r0 = idx  # Right windows:
            r1 = (idx + K).clamp(max=N)
            right_sums = cumsum[r1] - cumsum[r0]
            right_counts = (r1 - r0).unsqueeze(1)
            right_means = right_sums / right_counts
            left_means = F.normalize(left_means, dim=1)  # Cosine similarities:
            right_means = F.normalize(right_means, dim=1)
            boundaries = (left_means * right_means).sum(dim=1)  # (N-1,)
        assert len(boundaries) == len(embs) - 1
        boundaries = 1 - boundaries
        avg_boundaries_embs = boundaries.mean()
        if self.normalize_cos_boundaries:
            boundaries = boundaries / avg_boundaries_embs
            avg_boundaries_embs = 1
        return {'avg': avg_boundaries_embs, 'boundaries': boundaries}

    def grade_tokenboundaries_by_spaces(
            self, boundaries_embs, avg_boundaries_embs, text,
            tokens_spans, sentences_boundaries):
        boundaries = torch.zeros(len(boundaries_embs))
        assert len(boundaries) == len(tokens_spans) - 1
        span_prev = tokens_spans[0]
        for i, span in enumerate(tokens_spans[1:]):
            beg, end = span_prev[1], span[0]
            if beg != end:
                piece = text[beg:end]
                boundaries[i] += self.weight_by_spaces(piece)
            span_prev = span
        self.grade_tokenboundaries_by_sentences(boundaries, sentences_boundaries)
        nonzero_mask = (boundaries != 0)
        boundaries_nonzero = boundaries[nonzero_mask]
        avg = torch.mean(boundaries_nonzero)
        if avg == 0:
            return
        if self.weight_balance_spaces_embs > 0:
            factor = self.weight_balance_spaces_embs * avg_boundaries_embs / avg
            boundaries *= factor
        if self.weight_space_margin > 0:
            boundaries[nonzero_mask] += self.weight_space_margin
        boundaries_embs.add_(boundaries)

    def grade_tokenboundaries_by_sentences(self, boundaries, sentences_boundaries):
        """
        Arguments:
          boundaries List[float]: Weights of boundaries between tokens,
            starting with the first pair - a boundary between token 0 and token 1.
          sentences_boundaries List[int]: List of indexes of tokens (all the text tokens)
            from which a sentence begins, starting from the second sentence.
        """
        if self.sentence_weight is None:
            return
        for ix_tok in sentences_boundaries:
            boundaries[ix_tok - 1] += self.sentence_weight

    def weight_by_spaces(self, text):
        w, N = 0, 0
        for space, weight in self.spaces_weights.items():
            n = text.count(space)
            N += n
            w += n * weight
        w += max(0, len(text) - N) * self.space_weight_other
        return w

    def get_config_chunker(self):
        config_chunker = {
            k: v for k, v in self.config.items()
            if not k.startswith('out_') and not k.startswith('check_')}
        return config_chunker


def get_separator_of_chunks(chunk1, chunk2):
    chunk1_r = chunk1.rstrip()
    separator = chunk1[len(chunk1_r):]
    chunk2_l = chunk2.lstrip()
    diff = len(chunk2) - len(chunk2_l)
    separator += chunk2[:diff]
    assert len(chunk1_r) + len(chunk2_l) + len(separator) == len(chunk1) + len(chunk2)
    return separator