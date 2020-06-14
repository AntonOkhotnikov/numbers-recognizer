from itertools import groupby

import editdistance


class CERmetrics(object):

    def __init__(self, idx_blank, text_transformer, enrich_target=True):
        self.idx_blank = idx_blank
        self.text_transformer = text_transformer
        self.enrich_target = enrich_target

    def calculate_cer_ctc(self, ys_hat, ys_pad):
            """Calculate sentence-level CER score for CTC.
            :param torch.Tensor ys_hat: prediction (batch, seqlen)
            :param torch.Tensor ys_pad: reference (batch, seqlen)
            :return: average sentence-level CER score
            :rtype float
            """
            cers, char_ref_lens = [], []
            for i, y in enumerate(ys_hat):
                y_hat = [x[0] for x in groupby(y)]
                y_true = ys_pad[i]
                seq_hat, seq_true = [], []
                for idx in y_hat:
                    idx = int(idx)
                    if idx != -1 and idx != self.idx_blank:
                        seq_hat.append(self.text_transformer.index_map[int(idx)])

                for idx in y_true:
                    idx = int(idx)
                    if idx != -1 and idx != self.idx_blank:
                        seq_true.append(self.text_transformer.index_map[int(idx)])

                hyp_chars = "".join(seq_hat)
                ref_chars = "".join(seq_true)

                if self.enrich_target:
                    # remove '*' from CER calculation
                    hyp_chars = hyp_chars.replace('*', '')
                    ref_chars = ref_chars.replace('*', '')

                if len(ref_chars) > 0:
                    cers.append(editdistance.eval(hyp_chars, ref_chars))
                    char_ref_lens.append(len(ref_chars))

            cer_ctc = float(sum(cers)) / sum(char_ref_lens) if cers else None
            return cer_ctc

    def calculate_cer(self, hyps, refs):
        cers = []
        char_ref_lens = []
        for hyp, ref in zip(hyps, refs):
            if self.enrich_target:
                # remove '*' from CER calculation
                hyp = hyp.replace('*', '')
                ref = ref.replace('*', '')

            if len(ref) > 0:
                cers.append(editdistance.eval(hyp, ref))
                char_ref_lens.append(len(ref))
                # TODO: debug only
                if len(hyp) > 0:
                    print(hyp, ref)

        cer_ctc = float(sum(cers)) / sum(char_ref_lens) if cers else None
        return cer_ctc