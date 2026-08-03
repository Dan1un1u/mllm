#include <iostream>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include <sstream>
#include <string>
#include <vector>
#include "mllm/backends/qnn/aot_rt/QnnAOTRuntime.hpp"
#include "mllm/models/qwen3/configuration_qwen3.hpp"
#include "mllm/models/qwen3/tokenization_qwen3.hpp"

using mllm::Argparse;
using namespace mllm::qnn::aot;  // NOLINT

namespace {

struct AccuracyCase {
  std::string id;
  std::string match_mode;
  std::string expected;
  std::string prompt;
};

std::vector<AccuracyCase> loadAccuracyCases(const std::string& path) {
  std::ifstream stream(path);
  if (!stream.is_open()) { throw std::runtime_error("Cannot open accuracy eval file: " + path); }
  std::vector<AccuracyCase> cases;
  std::string line;
  while (std::getline(stream, line)) {
    if (line.empty() || line[0] == '#') { continue; }
    std::vector<std::string> fields;
    size_t begin = 0;
    while (fields.size() < 3) {
      const auto separator = line.find('\t', begin);
      if (separator == std::string::npos) { break; }
      fields.push_back(line.substr(begin, separator - begin));
      begin = separator + 1;
    }
    fields.push_back(line.substr(begin));
    if (fields.size() != 4) {
      throw std::runtime_error(
          "Accuracy eval line must contain id<TAB>match_mode<TAB>expected<TAB>prompt");
    }
    if (fields[1] != "number" && fields[1] != "text" && fields[1] != "choice" &&
        fields[1] != "exact") {
      throw std::runtime_error("Unsupported accuracy match mode: " + fields[1]);
    }
    cases.push_back({fields[0], fields[1], fields[2], fields[3]});
  }
  return cases;
}

std::string normalizeAnswer(const std::string& value) {
  std::string normalized;
  for (size_t i = 0; i < value.size();) {
    if (value[i] == '<') {
      const auto end = value.find('>', i + 1);
      if (end != std::string::npos) {
        i = end + 1;
        continue;
      }
    }
    // Ignore common UTF-8 CJK punctuation while retaining CJK answer text.
    bool skippedPunctuation = false;
    for (const std::string_view punctuation : {"。", "，", "！", "？", "：", "；"}) {
      if (value.compare(i, punctuation.size(), punctuation) == 0) {
        i += punctuation.size();
        skippedPunctuation = true;
        break;
      }
    }
    if (skippedPunctuation) { continue; }
    const auto ch = static_cast<unsigned char>(value[i++]);
    if (ch >= 0x80) {
      normalized.push_back(static_cast<char>(ch));
    } else if (std::isalnum(ch)) {
      normalized.push_back(static_cast<char>(std::tolower(ch)));
    }
  }
  return normalized;
}

bool isAscii(const std::string& value) {
  return std::all_of(value.begin(), value.end(),
                     [](const unsigned char ch) { return ch < 0x80; });
}

std::string lowercaseAscii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char ch) {
    return ch < 0x80 ? static_cast<char>(std::tolower(ch)) : static_cast<char>(ch);
  });
  return value;
}

bool containsStandaloneAscii(const std::string& output, const std::string& expected) {
  const auto haystack = lowercaseAscii(output);
  const auto needle = lowercaseAscii(expected);
  size_t position = 0;
  while ((position = haystack.find(needle, position)) != std::string::npos) {
    const auto isWord = [](const unsigned char ch) { return std::isalnum(ch) || ch == '_'; };
    const bool leftBoundary =
        position == 0 || !isWord(static_cast<unsigned char>(haystack[position - 1]));
    const size_t end = position + needle.size();
    const bool rightBoundary =
        end == haystack.size() || !isWord(static_cast<unsigned char>(haystack[end]));
    if (leftBoundary && rightBoundary) { return true; }
    position = end;
  }
  return false;
}

std::string lastNumericToken(const std::string& value) {
  std::string last;
  for (size_t i = 0; i < value.size();) {
    const auto ch = static_cast<unsigned char>(value[i]);
    const bool signedNumber =
        (ch == '+' || ch == '-') && i + 1 < value.size() &&
        std::isdigit(static_cast<unsigned char>(value[i + 1]));
    if (!std::isdigit(ch) && !signedNumber) {
      ++i;
      continue;
    }
    std::string token;
    if (signedNumber) { token.push_back(value[i++]); }
    while (i < value.size()) {
      const auto current = static_cast<unsigned char>(value[i]);
      if (std::isdigit(current) || current == '.' || current == ',' || current == '/' ||
          current == '%') {
        if (current != ',') { token.push_back(static_cast<char>(current)); }
        ++i;
      } else {
        break;
      }
    }
    while (!token.empty() && token.back() == '.') { token.pop_back(); }
    if (!token.empty()) { last = token; }
  }
  return last;
}

std::string firstNumericToken(const std::string& value) {
  for (size_t i = 0; i < value.size();) {
    const auto ch = static_cast<unsigned char>(value[i]);
    const bool signedNumber =
        (ch == '+' || ch == '-') && i + 1 < value.size() &&
        std::isdigit(static_cast<unsigned char>(value[i + 1]));
    if (!std::isdigit(ch) && !signedNumber) {
      ++i;
      continue;
    }
    std::string token;
    if (signedNumber) { token.push_back(value[i++]); }
    while (i < value.size()) {
      const auto current = static_cast<unsigned char>(value[i]);
      if (std::isdigit(current) || current == '.' || current == ',' || current == '/' ||
          current == '%') {
        if (current != ',') { token.push_back(static_cast<char>(current)); }
        ++i;
      } else {
        break;
      }
    }
    while (!token.empty() && token.back() == '.') { token.pop_back(); }
    if (!token.empty()) { return token; }
  }
  return {};
}

std::string normalizeNumeric(const std::string& value) {
  std::string normalized;
  for (const unsigned char ch : value) {
    if (!std::isspace(ch) && ch != ',') {
      normalized.push_back(static_cast<char>(std::tolower(ch)));
    }
  }
  while (!normalized.empty() && normalized.back() == '.') { normalized.pop_back(); }
  return normalized;
}

std::string answerRegion(const std::string& output, bool& explicitAnswer) {
  explicitAnswer = false;
  const auto lower = lowercaseAscii(output);
  std::vector<std::pair<size_t, size_t>> markers;
  for (size_t marker = lower.find("answer"); marker != std::string::npos;
       marker = lower.find("answer", marker + 6)) {
    markers.emplace_back(marker, 6);
  }
  const std::string chineseAnswer = "答案";
  for (size_t marker = output.find(chineseAnswer); marker != std::string::npos;
       marker = output.find(chineseAnswer, marker + chineseAnswer.size())) {
    markers.emplace_back(marker, chineseAnswer.size());
  }
  std::sort(markers.begin(), markers.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });
  for (const auto& [marker, markerLength] : markers) {
    auto candidate = output.substr(marker + markerLength);
    if (!normalizeAnswer(candidate).empty()) {
      explicitAnswer = true;
      return candidate;
    }
  }
  const bool completed = output.find("<|im_end|>") != std::string::npos ||
                         output.find("<|endoftext|>") != std::string::npos;
  return completed ? output : std::string{};
}

bool answerMatches(const AccuracyCase& test, const std::string& output) {
  const auto normalizedOutput = normalizeAnswer(output);
  bool explicitAnswer = false;
  const auto region = answerRegion(output, explicitAnswer);
  const auto normalizedRegion = normalizeAnswer(region);
  const auto numericOutput =
      explicitAnswer ? firstNumericToken(region) : lastNumericToken(region);
  std::stringstream stream(test.expected);
  std::string expected;
  while (std::getline(stream, expected, '|')) {
    const auto normalizedExpected = normalizeAnswer(expected);
    if (normalizedExpected.empty()) { continue; }
    if (test.match_mode == "number") {
      if (!numericOutput.empty() &&
          normalizeNumeric(numericOutput) == normalizeNumeric(expected)) {
        return true;
      }
      continue;
    }
    if (normalizedOutput == normalizedExpected) { return true; }
    if (test.match_mode == "exact") { continue; }
    if (region.empty()) { continue; }
    const bool standaloneMatch =
        isAscii(expected) ? containsStandaloneAscii(region, expected)
                          : normalizedRegion.find(normalizedExpected) != std::string::npos;
    if ((test.match_mode == "text" || test.match_mode == "choice") &&
        normalizedExpected.size() >= 2 && standaloneMatch) {
      const std::string negatedEnglish = "not" + normalizedExpected;
      const std::string negatedChinese = "不是" + normalizedExpected;
      if (normalizedRegion.find(negatedEnglish) != std::string::npos ||
          normalizedRegion.find(negatedChinese) != std::string::npos) {
        continue;
      }
      return true;
    }
  }
  return false;
}

std::string csvEscape(const std::string& value) {
  std::string escaped = "\"";
  for (const char ch : value) {
    if (ch == '"') { escaped.push_back('"'); }
    escaped.push_back(ch);
  }
  escaped.push_back('"');
  return escaped;
}

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model").help("Model path").def("qwen3_qnn.mllm");
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer").help("Tokenizer path").def("tokenizer.json");
  auto& config_path = Argparse::add<std::string>("-c|--config").help("Config path").required(true);
  auto& ar_len = Argparse::add<int>("--ar_len").help("Autoregressive length (chunk size)").def(128);
  auto& max_new_tokens =
      Argparse::add<int>("--max_new_tokens").help("Maximum generated tokens, including the first prefill token").def(1024);
  auto& perf = Argparse::add<bool>("--perf").help("Print and persist end-to-end prefill/decode throughput");
  auto& eval_file =
      Argparse::add<std::string>("--eval_file").help("Run tab-separated short-answer accuracy sanity suite");

  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }

  mllm::initQnnBackend(model_path.get());

  auto qwen3_cfg = mllm::models::qwen3::Qwen3Config(config_path.get());

  RunnerConfig config;
  config.num_layers = qwen3_cfg.num_hidden_layers;
  config.num_heads = qwen3_cfg.num_key_value_heads;
  config.head_dim = qwen3_cfg.head_dim;
  config.vocab_size = qwen3_cfg.vocab_size;
  config.context_len = 1024;
  config.ar_len = ar_len.get();

  auto tokenizer = mllm::models::qwen3::Qwen3Tokenizer(tokenizer_path.get());

  if (eval_file.isSet()) {
    const auto cases = loadAccuracyCases(eval_file.get());
    const char* profileDir = std::getenv("MLLM_QNN_PROFILE_DIR");
    const std::string outputPath =
        std::string(profileDir ? profileDir : "/data/local/tmp") + "/qnn_accuracy_eval.csv";
    std::ofstream outputFile(outputPath, std::ios::trunc);
    if (!outputFile.is_open()) {
      std::cerr << "Cannot write accuracy eval output: " << outputPath << "\n";
      return 1;
    }
    outputFile << "id,match_mode,expected,output,normalized_output,pass\n";
    int passed = 0;
    const int decode_steps = std::max(0, max_new_tokens.get() - 1);
    // QNN graph tensor wrappers pin the first Runner's buffer addresses.
    // Keep one Runner alive for the whole suite and reset its cache in place
    // between independent prompts.
    Runner runner(config, &tokenizer);
    if (!runner.load()) {
      std::cerr << "Failed to initialize runner for accuracy suite\n";
      return 1;
    }
    for (const auto& test : cases) {
      auto input_tensor = tokenizer.convertMessage({.prompt = test.prompt});
      std::string generated;
      runner.reset();
      runner.generate(input_tensor["sequence"], decode_steps,
                      [&](const std::string& token) { generated += token; }, false);
      const bool ok = answerMatches(test, generated);
      passed += ok ? 1 : 0;
      outputFile << csvEscape(test.id) << ',' << csvEscape(test.match_mode) << ','
                 << csvEscape(test.expected) << ',' << csvEscape(generated) << ','
                 << csvEscape(normalizeAnswer(generated)) << ',' << (ok ? 1 : 0) << '\n';
      fmt::print("[ACC] {:<24} {}  output={}\n", test.id, ok ? "PASS" : "FAIL", generated);
    }
    fmt::print("Accuracy sanity summary: {}/{} ({:.2f}%)\n", passed, cases.size(),
               cases.empty() ? 0.0 : 100.0 * passed / cases.size());
    return cases.empty() ? 1 : 0;
  }

  std::string prompt_text;
  fmt::print("💬 Prompt text (or 'exit/quit'): ");
  std::getline(std::cin, prompt_text);

  auto input_tensor = tokenizer.convertMessage({.prompt = prompt_text});

  Runner runner(config, &tokenizer);
  if (!runner.load()) {
    std::cerr << "Failed to load model\n";
    return 1;
  }

  // PromptProcessor produces the first generated token. TokenGenerator should
  // therefore run at most max_new_tokens - 1 additional decode steps.
  const int decode_steps = std::max(0, max_new_tokens.get() - 1);
  runner.generate(input_tensor["sequence"], decode_steps,
                  [](const std::string& token) { std::cout << token << std::flush; }, perf.isSet());
  std::cout << "\n";

  return 0;
});
