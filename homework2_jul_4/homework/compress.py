from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image

from .autoregressive import Autoregressive
from .bsq import Tokenizer


class Compressor:
    PRECISION = 16
    TOP = (1 << 32) - 1
    HALF = 1 << 31
    QUARTER = 1 << 30

    def __init__(self, tokenizer: Tokenizer, autoregressive: Autoregressive):
        super().__init__()
        self.tokenizer = tokenizer
        self.autoregressive = autoregressive

    def _freq_table(self, tokens: torch.Tensor, i: int):
        # cumulative frequency table for token i given everything before it
        logits, _ = self.autoregressive(tokens)
        probs = torch.softmax(logits.view(-1, logits.shape[-1])[i].double().cpu(), dim=-1)
        freq = (probs * (1 << self.PRECISION)).long() + 1
        cum = torch.cumsum(freq, 0).tolist()
        return cum, cum[-1]

    def compress(self, x: torch.Tensor) -> bytes:
        """
        Compress the image into a torch.uint8 bytes stream (1D tensor).

        Use arithmetic coding.
        """
        device = next(self.autoregressive.parameters()).device
        tokens = self.tokenizer.encode_index(x.to(device))
        h, w = tokens.shape
        symbols = tokens.flatten().tolist()

        low, high, pending = 0, self.TOP, 0
        bits = []

        def emit(bit):
            nonlocal pending
            bits.append(bit)
            bits.extend([1 - bit] * pending)
            pending = 0

        known = torch.zeros(1, h, w, dtype=torch.long, device=device)
        known_flat = known.view(-1)
        with torch.no_grad():
            for i, s in enumerate(symbols):
                cum, total = self._freq_table(known, i)
                lo = cum[s - 1] if s > 0 else 0
                hi = cum[s]
                span = high - low + 1
                high = low + span * hi // total - 1
                low = low + span * lo // total
                while True:
                    if high < self.HALF:
                        emit(0)
                    elif low >= self.HALF:
                        emit(1)
                        low -= self.HALF
                        high -= self.HALF
                    elif low >= self.QUARTER and high < 3 * self.QUARTER:
                        pending += 1
                        low -= self.QUARTER
                        high -= self.QUARTER
                    else:
                        break
                    low = low * 2
                    high = high * 2 + 1
                known_flat[i] = s

        pending += 1
        emit(0 if low < self.QUARTER else 1)
        bits.extend([0] * 16)

        while len(bits) % 8:
            bits.append(0)
        data = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for b in bits[i : i + 8]:
                byte = byte * 2 + b
            data.append(byte)
        return bytes(data)

    def decompress(self, x: bytes) -> torch.Tensor:
        """
        Decompress a tensor into a PIL image.
        You may assume the output image is 150 x 100 pixels.
        """
        import bisect

        device = next(self.autoregressive.parameters()).device
        dummy = self.tokenizer.encode_index(torch.zeros(100, 150, 3, device=device))
        h, w = dummy.shape

        bits = []
        for byte in x:
            for k in range(7, -1, -1):
                bits.append((byte >> k) & 1)
        pos = 0

        def read_bit():
            nonlocal pos
            b = bits[pos] if pos < len(bits) else 0
            pos += 1
            return b

        low, high = 0, self.TOP
        value = 0
        for _ in range(32):
            value = value * 2 + read_bit()

        known = torch.zeros(1, h, w, dtype=torch.long, device=device)
        known_flat = known.view(-1)
        with torch.no_grad():
            for i in range(h * w):
                cum, total = self._freq_table(known, i)
                span = high - low + 1
                target = ((value - low + 1) * total - 1) // span
                s = bisect.bisect_right(cum, target)
                lo = cum[s - 1] if s > 0 else 0
                hi = cum[s]
                high = low + span * hi // total - 1
                low = low + span * lo // total
                while True:
                    if high < self.HALF:
                        pass
                    elif low >= self.HALF:
                        low -= self.HALF
                        high -= self.HALF
                        value -= self.HALF
                    elif low >= self.QUARTER and high < 3 * self.QUARTER:
                        low -= self.QUARTER
                        high -= self.QUARTER
                        value -= self.QUARTER
                    else:
                        break
                    low = low * 2
                    high = high * 2 + 1
                    value = value * 2 + read_bit()
                known_flat[i] = s

        return self.tokenizer.decode_index(known[0])


def compress(tokenizer: Path, autoregressive: Path, image: Path, compressed_image: Path):
    """
    Compress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    images: Path to the image to compress.
    compressed_image: Path to save the compressed image tensor.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    x = torch.tensor(np.array(Image.open(image)), dtype=torch.uint8, device=device)
    cmp_img = cmp.compress(x.float() / 255.0 - 0.5)
    with open(compressed_image, "wb") as f:
        f.write(cmp_img)


def decompress(tokenizer: Path, autoregressive: Path, compressed_image: Path, image: Path):
    """
    Decompress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    compressed_image: Path to the compressed image tensor.
    images: Path to save the image to compress.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    with open(compressed_image, "rb") as f:
        cmp_img = f.read()

    x = cmp.decompress(cmp_img)
    img = Image.fromarray(((x + 0.5) * 255.0).clamp(min=0, max=255).byte().cpu().numpy())
    img.save(image)


if __name__ == "__main__":
    from fire import Fire

    Fire({"compress": compress, "decompress": decompress})