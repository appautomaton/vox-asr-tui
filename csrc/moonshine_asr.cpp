#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "moonshine-c-api.h"

namespace {

struct ParsedWav {
  int32_t sample_rate = 0;
  std::vector<float> samples;
};

uint16_t ReadU16LE(const uint8_t *p) {
  return static_cast<uint16_t>(p[0]) |
         (static_cast<uint16_t>(p[1]) << 8);
}

uint32_t ReadU32LE(const uint8_t *p) {
  return static_cast<uint32_t>(p[0]) |
         (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

bool ReadAllStdin(std::vector<uint8_t> *out) {
  if (out == nullptr) return false;
  std::vector<uint8_t> data;
  constexpr size_t kChunkSize = 65536;
  std::vector<char> chunk(kChunkSize);

  while (std::cin.good()) {
    std::cin.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
    const std::streamsize n = std::cin.gcount();
    if (n <= 0) break;
    data.insert(data.end(), chunk.begin(), chunk.begin() + n);
  }

  if (data.empty()) return false;
  *out = std::move(data);
  return true;
}

bool ParsePcm16MonoWav(const std::vector<uint8_t> &bytes, ParsedWav *out,
                       std::string *error) {
  if (out == nullptr) return false;
  if (bytes.size() < 44) {
    if (error) *error = "WAV too small";
    return false;
  }
  if (std::memcmp(bytes.data(), "RIFF", 4) != 0 ||
      std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
    if (error) *error = "Input is not a RIFF/WAVE file";
    return false;
  }

  bool found_fmt = false;
  bool found_data = false;
  uint16_t audio_format = 0;
  uint16_t channels = 0;
  uint32_t sample_rate = 0;
  uint16_t bits_per_sample = 0;
  const uint8_t *data_ptr = nullptr;
  uint32_t data_size = 0;

  size_t offset = 12;
  while (offset + 8 <= bytes.size()) {
    const uint8_t *chunk = bytes.data() + offset;
    const uint32_t chunk_size = ReadU32LE(chunk + 4);
    const size_t chunk_data_offset = offset + 8;
    const size_t chunk_data_end = chunk_data_offset + chunk_size;
    if (chunk_data_end > bytes.size()) break;

    if (std::memcmp(chunk, "fmt ", 4) == 0) {
      if (chunk_size < 16) {
        if (error) *error = "Invalid fmt chunk";
        return false;
      }
      audio_format = ReadU16LE(bytes.data() + chunk_data_offset + 0);
      channels = ReadU16LE(bytes.data() + chunk_data_offset + 2);
      sample_rate = ReadU32LE(bytes.data() + chunk_data_offset + 4);
      bits_per_sample = ReadU16LE(bytes.data() + chunk_data_offset + 14);
      found_fmt = true;
    } else if (std::memcmp(chunk, "data", 4) == 0) {
      data_ptr = bytes.data() + chunk_data_offset;
      data_size = chunk_size;
      found_data = true;
    }

    const size_t padded_size = chunk_size + (chunk_size % 2);
    offset = chunk_data_offset + padded_size;
  }

  if (!found_fmt || !found_data) {
    if (error) *error = "WAV missing fmt or data chunk";
    return false;
  }
  if (audio_format != 1 || channels != 1 || bits_per_sample != 16) {
    if (error) {
      *error = "WAV must be PCM 16-bit mono";
    }
    return false;
  }
  if (sample_rate <= 0) {
    if (error) *error = "Invalid WAV sample rate";
    return false;
  }
  if (data_size < 2) {
    if (error) *error = "WAV data is empty";
    return false;
  }

  const size_t sample_count = data_size / 2;
  out->sample_rate = static_cast<int32_t>(sample_rate);
  out->samples.resize(sample_count);

  for (size_t i = 0; i < sample_count; i++) {
    const uint8_t *p = data_ptr + (i * 2);
    const int16_t s = static_cast<int16_t>(ReadU16LE(p));
    out->samples[i] = static_cast<float>(s) / 32768.0f;
  }

  return true;
}

int32_t ParseModelArch(const std::string &value) {
  if (value == "tiny") return MOONSHINE_MODEL_ARCH_TINY;
  if (value == "base") return MOONSHINE_MODEL_ARCH_BASE;
  if (value == "tiny-streaming") return MOONSHINE_MODEL_ARCH_TINY_STREAMING;
  if (value == "base-streaming") return MOONSHINE_MODEL_ARCH_BASE_STREAMING;
  if (value == "small-streaming") return MOONSHINE_MODEL_ARCH_SMALL_STREAMING;
  if (value == "medium-streaming") return MOONSHINE_MODEL_ARCH_MEDIUM_STREAMING;
  return -1;
}

std::string JoinTranscriptLines(const transcript_t *transcript) {
  if (transcript == nullptr || transcript->line_count == 0 || transcript->lines == nullptr) {
    return "";
  }

  std::string text;
  for (uint64_t i = 0; i < transcript->line_count; i++) {
    const transcript_line_t &line = transcript->lines[i];
    if (line.text == nullptr || line.text[0] == '\0') {
      continue;
    }
    if (!text.empty()) {
      text.push_back('\n');
    }
    text += line.text;
  }
  return text;
}

}  // namespace

int main(int argc, char *argv[]) {
  std::string model_dir = "bin/moonshine-models/medium-streaming-en";
  int32_t model_arch = MOONSHINE_MODEL_ARCH_MEDIUM_STREAMING;

  for (int i = 1; i < argc; i++) {
    const std::string arg = argv[i];
    if (arg == "--model-dir" && i + 1 < argc) {
      model_dir = argv[++i];
      continue;
    }
    if (arg == "--model-arch" && i + 1 < argc) {
      const int32_t parsed = ParseModelArch(argv[++i]);
      if (parsed < 0) {
        std::cerr << "Unsupported --model-arch value\n";
        return 2;
      }
      model_arch = parsed;
      continue;
    }
    if (arg == "-h" || arg == "--help") {
      std::cout << "Usage: moonshine_asr [--model-dir PATH] "
                   "[--model-arch medium-streaming]\n";
      return 0;
    }
    std::cerr << "Unknown arg: " << arg << "\n";
    return 2;
  }

  std::vector<uint8_t> wav_bytes;
  if (!ReadAllStdin(&wav_bytes)) {
    std::cerr << "Failed to read WAV bytes from stdin\n";
    return 2;
  }

  ParsedWav wav;
  std::string wav_error;
  if (!ParsePcm16MonoWav(wav_bytes, &wav, &wav_error)) {
    std::cerr << "Invalid WAV input: " << wav_error << "\n";
    return 2;
  }

  const int32_t transcriber_handle = moonshine_load_transcriber_from_files(
      model_dir.c_str(), static_cast<uint32_t>(model_arch), nullptr, 0,
      MOONSHINE_HEADER_VERSION);
  if (transcriber_handle < 0) {
    std::cerr << "Failed to load Moonshine model: "
              << moonshine_error_to_string(transcriber_handle) << "\n";
    return 3;
  }

  transcript_t *transcript = nullptr;
  const int32_t transcribe_error = moonshine_transcribe_without_streaming(
      transcriber_handle, wav.samples.data(), wav.samples.size(), wav.sample_rate,
      0, &transcript);
  if (transcribe_error != MOONSHINE_ERROR_NONE) {
    std::cerr << "Moonshine transcription failed: "
              << moonshine_error_to_string(transcribe_error) << "\n";
    moonshine_free_transcriber(transcriber_handle);
    return 4;
  }

  const std::string text = JoinTranscriptLines(transcript);
  if (!text.empty()) {
    std::cout << text;
  }

  moonshine_free_transcriber(transcriber_handle);
  return 0;
}
