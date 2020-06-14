import torch


def GreedyDecoder(output, labels, label_lengths, blank_label, text_transformer, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes.T):
        decode = []
        targets.append(text_transformer.int2txt(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transformer.int2txt(decode))
    return decodes, targets
