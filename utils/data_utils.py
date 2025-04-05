import torch


def generate_batch_text_tokens(text_lens, max_len, vocab_size, pad_index=0):
    return torch.stack(
        list(
            torch.concat(
                (
                    torch.randint(1, vocab_size, (L,)),
                    torch.ones(max_len - L) * pad_index,
                )
            )
            for L in text_lens
        ),
        dim=0,
    ).to(torch.long)
