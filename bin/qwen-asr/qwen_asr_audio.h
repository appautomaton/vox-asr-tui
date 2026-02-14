/*
 * qwen_asr_audio.h - WAV loading and mel spectrogram computation
 */

#ifndef QWEN_ASR_AUDIO_H
#define QWEN_ASR_AUDIO_H

#include <stddef.h>
#include <stdint.h>

/* Load a WAV file, returns mono float32 samples in [-1,1] at 16kHz.
 * Handles: 16-bit PCM, mono or stereo (mixed to mono).
 * Resamples to 16kHz if needed.
 * Returns NULL on error. Caller must free returned buffer. */
float *qwen_load_wav(const char *path, int *out_n_samples);

/* Parse a WAV file from a memory buffer. Caller must free returned buffer. */
float *qwen_parse_wav_buffer(const uint8_t *data, size_t size, int *out_n_samples);

/* Read audio from stdin (auto-detect WAV or raw s16le 16kHz mono).
 * Returns NULL on error. Caller must free returned buffer. */
float *qwen_read_pcm_stdin(int *out_n_samples);

/* Compute log-mel spectrogram from audio samples.
 * Uses dynamic maximum for clamping (unlike Voxtral's fixed 1.5).
 * samples: mono float32 at 16kHz
 * n_samples: number of samples
 * out_frames: set to number of mel frames produced
 * Returns: [128, n_frames] mel spectrogram (caller must free)
 * Note: Returns in [mel_bins, frames] layout for Conv2D compatibility. */
float *qwen_mel_spectrogram(const float *samples, int n_samples, int *out_frames);

#endif /* QWEN_ASR_AUDIO_H */
