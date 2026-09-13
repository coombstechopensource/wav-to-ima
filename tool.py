"""
WAV -> RemindMi raw IMA ADPCM converter

Creates files compatible with the ESP32 RemindMi sketch:
    4-byte header:
      uint16 little-endian initial PCM predictor
      uint8  initial IMA step index
      uint8  reserved
    followed by raw IMA ADPCM 4-bit nibbles, LOW nibble first.

Input:
    PCM WAV, preferably 16-bit.
    The script also handles 8-bit PCM and resamples to 22050 Hz.
    Stereo WAV files are downmixed to mono.

Usage:
    python wav_to_ima.py input.wav output.ima
"""

import sys
import wave
import struct
import math
import os

TARGET_RATE = 22050

STEP_TABLE = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31,
    34, 37, 41, 45, 50, 55, 60, 66, 73, 80, 88, 97, 107, 118, 130,
    143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449,
    494, 544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411,
    1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327, 3660, 4026,
    4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487,
    12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794,
    32767
]

INDEX_TABLE = [
    -1, -1, -1, -1, 2, 4, 6, 8,
    -1, -1, -1, -1, 2, 4, 6, 8
]


def read_wav(filename):
    with wave.open(filename, "rb") as w:
        channels = w.getnchannels()
        rate = w.getframerate()
        width = w.getsampwidth()
        frames = w.getnframes()
        comptype = w.getcomptype()

        if comptype != "NONE":
            raise ValueError("Compressed WAV files are not supported.")

        raw = w.readframes(frames)

    if width == 2:
        values = list(struct.unpack("<" + "h" * (len(raw) // 2), raw))
    elif width == 1:
        # Standard WAV 8-bit PCM is unsigned.
        values = [(b - 128) * 256 for b in raw]
    else:
        raise ValueError("Only 8-bit or 16-bit PCM WAV files are supported.")

    if channels > 1:
        values = [
            int(sum(values[i:i + channels]) / channels)
            for i in range(0, len(values), channels)
        ]

    return values, rate


def resample(samples, source_rate, target_rate):
    if source_rate == target_rate:
        return samples

    if not samples:
        return []

    out_length = int(round(len(samples) * target_rate / source_rate))
    result = []

    for n in range(out_length):
        pos = n * (source_rate / target_rate)
        i = int(pos)
        frac = pos - i

        if i >= len(samples) - 1:
            result.append(samples[-1])
        else:
            a = samples[i]
            b = samples[i + 1]
            result.append(int(round(a + (b - a) * frac)))

    return result


def choose_initial_index(samples):
    # Estimate a useful starting step size from the first few differences.
    if len(samples) < 2:
        return 0

    diffs = [
        abs(samples[i] - samples[i - 1])
        for i in range(1, min(len(samples), 1000))
    ]
    average = sum(diffs) / len(diffs)

    # Find a step close to the typical initial change.
    best = min(
        range(len(STEP_TABLE)),
        key=lambda i: abs(STEP_TABLE[i] - max(1, average))
    )
    return max(0, min(88, best))


def encode_ima(samples):
    if not samples:
        raise ValueError("WAV contains no audio samples.")

    predictor = max(-32768, min(32767, samples[0]))
    index = choose_initial_index(samples)

    initial_predictor = predictor
    initial_index = index

    packed = bytearray()
    current_byte = 0
    low_nibble = True

    for target in samples[1:]:
        step = STEP_TABLE[index]
        diff = target - predictor

        code = 0
        if diff < 0:
            code |= 8
            diff = -diff

        delta = step
        if diff >= delta:
            code |= 4
            diff -= delta
        if diff >= delta // 2:
            code |= 2
            diff -= delta // 2
        if diff >= delta // 4:
            code |= 1

        # Reconstruct the encoder's predictor exactly as the decoder will.
        diffq = step >> 3
        if code & 1:
            diffq += step >> 2
        if code & 2:
            diffq += step >> 1
        if code & 4:
            diffq += step

        if code & 8:
            predictor -= diffq
        else:
            predictor += diffq

        predictor = max(-32768, min(32767, predictor))

        index += INDEX_TABLE[code]
        index = max(0, min(88, index))

        if low_nibble:
            current_byte = code & 0x0F
            low_nibble = False
        else:
            current_byte |= (code & 0x0F) << 4
            packed.append(current_byte)
            low_nibble = True

    if not low_nibble:
        # One padding nibble. It is decoded as one extra sample; the
        # difference is only about 45 microseconds at 22050 Hz.
        packed.append(current_byte)

    header = struct.pack("<hBB", initial_predictor, initial_index, 0)
    return header + packed


def convert(input_file, output_file):
    samples, source_rate = read_wav(input_file)

    print(f"Input sample rate : {source_rate} Hz")
    print(f"Input samples     : {len(samples):,}")

    samples = resample(samples, source_rate, TARGET_RATE)

    duration = len(samples) / TARGET_RATE

    encoded = encode_ima(samples)

    with open(output_file, "wb") as f:
        f.write(encoded)

    os.unlink(input_file)

    print(f"Output sample rate: {TARGET_RATE} Hz")
    print(f"Duration          : {duration:.2f} seconds")
    print(f"Output size       : {len(encoded):,} bytes")
    print(f"Created           : {output_file}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python wav_to_ima.py input.wav output.ima")
        sys.exit(1)

    try:
        convert(sys.argv[1], sys.argv[2])
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
