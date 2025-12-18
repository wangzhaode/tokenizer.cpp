#include "tokenizer.hpp"
#include <set>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <unordered_map>
#include <cmath>
#include <oniguruma.h>
#include <utf8proc/utf8proc.h>
#include "jinja.hpp"

namespace tokenizer {

// ==========================================
// C++11 Polyfills
// ==========================================
template<typename T, typename... Args>
std::unique_ptr<T> std_make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// ==========================================
// Internal Data Structures
// ==========================================

struct Encoding {
    std::vector<int> input_ids;
    std::vector<int> attention_mask;
};

struct PreTokenizedString {
    std::vector<std::string> splits;
};

// ==========================================
// Component Interfaces
// ==========================================

class Normalizer {
public:
    virtual ~Normalizer() = default;
    virtual std::string normalize(const std::string& text) const = 0;
};

class PreTokenizer {
public:
    virtual ~PreTokenizer() = default;
    virtual void pre_tokenize(PreTokenizedString& pts) const = 0;
};

class Model {
public:
    virtual ~Model() = default;
    virtual std::vector<int> tokenize(const std::string& text) const = 0;
    virtual int token_to_id(const std::string& token) const = 0;
    virtual std::string id_to_token(int id) const = 0;
    virtual size_t vocab_size() const = 0;
};

class PostProcessor {
public:
    virtual ~PostProcessor() = default;
    virtual void process(Encoding& encoding) const = 0;
};

class Decoder {
public:
    virtual ~Decoder() = default;
    virtual void decode(std::vector<std::string>& tokens) const = 0;
};

// ==========================================
// Utils
// ==========================================

static std::string OnigurumaRegexEscape(const std::string& pattern) {
    std::string escaped;
    for (char c : pattern) {
        if (c == '\\' || c == '.' || c == '+' || c == '*' || c == '?' || c == '(' || c == ')' || c == '[' || c == ']' || c == '{' || c == '}' || c == '|' || c == '^' || c == '$') {
            escaped += '\\';
        }
        escaped += c;
    }
    return escaped;
}

static std::string get_token_content(const nlohmann::json& j) {
    if (j.is_string()) return j.get<std::string>();
    if (j.is_object() && j.contains("content")) return j["content"].get<std::string>();
    return "";
}

static std::unordered_map<unsigned char, std::string> create_bytes_char_map() {
    auto u2u = [](int cp) -> std::string {
        std::string out;
        if (cp <= 0x7F) out += (char)cp;
        else if (cp <= 0x7FF) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
        else if (cp <= 0xFFFF) { out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
        return out;
    };
    std::unordered_map<unsigned char, std::string> bs;
    for (int b = 33; b <= 126; ++b) bs[(unsigned char)b] = u2u(b);
    for (int b = 161; b <= 172; ++b) bs[(unsigned char)b] = u2u(b);
    for (int b = 174; b <= 255; ++b) bs[(unsigned char)b] = u2u(b);
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (bs.find((unsigned char)b) == bs.end()) bs[(unsigned char)b] = u2u(256 + n++);
    }
    return bs;
}

// ==========================================
// Component Implementations
// ==========================================

class OnigRegex {
public:
    OnigRegex(const std::string& pattern) : regex_(nullptr), valid_(false) {
        regex_t* reg;
        OnigErrorInfo einfo;
        onig_init();
        int r = onig_new(&reg, (uint8_t*)pattern.c_str(), (uint8_t*)(pattern.c_str() + pattern.length()),
                         ONIG_OPTION_DEFAULT, ONIG_ENCODING_UTF8, ONIG_SYNTAX_DEFAULT, &einfo);
        if (r == ONIG_NORMAL) {
            regex_ = (void*)reg;
            valid_ = true;
        } else {
            valid_ = false;
        }
    }
    ~OnigRegex() {
        if (regex_) onig_free((regex_t*)regex_);
    }
    void* get() const { return regex_; }
    bool is_valid() const { return valid_; }

    bool search(const std::string& text, int start_offset, int end_offset, int& match_start, int& match_end) const {
        if (!valid_ || text.empty()) return false;
        const uint8_t* str = (const uint8_t*)text.c_str();
        const uint8_t* start = str + start_offset;
        const uint8_t* end = str + end_offset;
        OnigRegion* region = onig_region_new();
        int r = onig_search((regex_t*)regex_, str, str + text.length(), start, end, region, ONIG_OPTION_NONE);
        if (r >= 0) {
            match_start = region->beg[0];
            match_end = region->end[0];
            onig_region_free(region, 1);
            return true;
        }
        onig_region_free(region, 1);
        return false;
    }

private:
    void* regex_;
    bool valid_;
};

class NFKCNormalizer : public Normalizer {
public:
    std::string normalize(const std::string& text) const override {
        uint8_t* result = nullptr;
        ssize_t ret = utf8proc_map((const uint8_t*)text.c_str(), 0, &result, (utf8proc_option_t)(UTF8PROC_NULLTERM | UTF8PROC_STABLE | UTF8PROC_COMPOSE | UTF8PROC_COMPAT));
        if (ret >= 0 && result) {
            std::string normalized_text = (const char*)result;
            free(result);
            return normalized_text;
        }
        return text; // Return original if normalization fails
    }
};

class PrependNormalizer : public Normalizer {
    std::string prepend_;
public:
    PrependNormalizer(const std::string& p) : prepend_(p) {}
    std::string normalize(const std::string& text) const override { return prepend_ + text; }
};

class ReplaceNormalizer : public Normalizer {
    std::string pattern_, content_;
public:
    ReplaceNormalizer(const std::string& p, const std::string& c) : pattern_(p), content_(c) {}
    std::string normalize(const std::string& text) const override {
        if (pattern_.empty()) return text;
        std::string out = text;
        size_t pos = 0;
        while ((pos = out.find(pattern_, pos)) != std::string::npos) {
            out.replace(pos, pattern_.length(), content_);
            pos += content_.length();
        }
        return out;
    }
};

class SequenceNormalizer : public Normalizer {
    std::vector<std::shared_ptr<Normalizer>> normalizers_;
public:
    SequenceNormalizer(const std::vector<std::shared_ptr<Normalizer>>& n) : normalizers_(n) {}
    std::string normalize(const std::string& text) const override {
        std::string out = text;
        for (const auto& n : normalizers_) out = n->normalize(out);
        return out;
    }
};

class SequencePreTokenizer : public PreTokenizer {
public:
    std::vector<std::shared_ptr<PreTokenizer>> pts_;
    SequencePreTokenizer(const std::vector<std::shared_ptr<PreTokenizer>>& pts) : pts_(pts) {}
    void pre_tokenize(PreTokenizedString& pts) const override {
        for (const auto& pt : pts_) pt->pre_tokenize(pts);
    }
};

class ByteLevelPreTokenizer : public PreTokenizer {
    bool use_regex_ = false;
    mutable std::shared_ptr<OnigRegex> regex_;
public:
    ByteLevelPreTokenizer(bool use_regex = false) : use_regex_(use_regex) {
        if (use_regex_) {
            regex_ = std::make_shared<OnigRegex>("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");
        }
    }
    void pre_tokenize(PreTokenizedString& pts) const override {
        if (use_regex_ && regex_ && regex_->is_valid()) {
            std::vector<std::string> next_splits;
            for (const auto& s : pts.splits) {
                if (s.empty()) continue;
                int last_pos = 0;
                while (last_pos < (int)s.size()) {
                    int match_start = -1, match_end = -1;
                    if (regex_->search(s, last_pos, (int)s.size(), match_start, match_end)) {
                        if (match_start > last_pos) {
                            next_splits.push_back(s.substr(last_pos, match_start - last_pos));
                        }
                        if (match_end > match_start) {
                            next_splits.push_back(s.substr(match_start, match_end - match_start));
                        }
                        last_pos = match_end;
                        if (match_start == match_end) last_pos++;
                    } else {
                        next_splits.push_back(s.substr(last_pos));
                        break;
                    }
                }
            }
            pts.splits = next_splits;
        }
        static auto byte_map = create_bytes_char_map();
        for (auto& s : pts.splits) {
            std::string out;
            for (unsigned char b : s) out += byte_map[b];
            s = out;
        }
    }
};

class DigitsPreTokenizer : public PreTokenizer {
    bool individual_digits_;
public:
    DigitsPreTokenizer(bool id) : individual_digits_(id) {}
    void pre_tokenize(PreTokenizedString& pts) const override {
        std::vector<std::string> next_splits;
        for (const auto& s : pts.splits) {
            std::string current;
            for (size_t i = 0; i < s.length(); ) {
                int32_t cp;
                int len = utf8proc_iterate((const uint8_t*)s.data() + i, s.size() - i, &cp);
                if (len <= 0) break;
                std::string c = s.substr(i, len);
                bool is_digit = (c.length() == 1 && c[0] >= '0' && c[0] <= '9');
                if (is_digit && individual_digits_) {
                    if (!current.empty()) { next_splits.push_back(current); current.clear(); }
                    next_splits.push_back(c);
                } else {
                    current += c;
                }
                i += len;
            }
            if (!current.empty()) next_splits.push_back(current);
        }
        pts.splits = next_splits;
    }
};

class MetaspacePreTokenizer : public PreTokenizer {
public:
    std::string replacement_;
    bool add_prefix_space_;
    MetaspacePreTokenizer(const std::string& rep, bool aps) : replacement_(rep), add_prefix_space_(aps) {}
    void pre_tokenize(PreTokenizedString& pts) const override {
        if (add_prefix_space_ && !pts.splits.empty() && !pts.splits[0].empty() && pts.splits[0][0] != ' ') {
            pts.splits[0] = " " + pts.splits[0];
        }
        for (auto& s : pts.splits) {
            std::string out;
            for (size_t i = 0; i < (int)s.size();) {
                int32_t cp;
                int len = utf8proc_iterate((const uint8_t*)s.data() + i, s.size() - i, &cp);
                if (len <= 0) break;
                std::string c = s.substr(i, len);
                out += (c == " ") ? replacement_ : c;
                i += len;
            }
            s = out;
        }
    }
};

class SplitPreTokenizer : public PreTokenizer {
public:
    std::unique_ptr<OnigRegex> regex_;
    bool invert_;
    std::string behavior_;
    SplitPreTokenizer(const std::string& pattern, bool invert, const std::string& behavior = "Isolated")
        : regex_(std_make_unique<OnigRegex>(pattern)), invert_(invert), behavior_(behavior) {}
    void pre_tokenize(PreTokenizedString& pts) const override {
        if (!regex_ || !regex_->is_valid()) return;
        std::vector<std::string> new_splits;
        for (const auto& s : pts.splits) {
            int current_pos = 0;
            while (current_pos < (int)s.size()) {
                int match_start = -1, match_end = -1;
                if (regex_->search(s, current_pos, (int)s.size(), match_start, match_end)) {
                    // Found a match
                    if (invert_) {
                        // Invert means we keep the matched parts
                        if (match_end > match_start) {
                            new_splits.push_back(s.substr(match_start, match_end - match_start));
                        }
                    } else {
                        // Not inverted means we split by the matched parts
                        if (match_start > current_pos) {
                            new_splits.push_back(s.substr(current_pos, match_start - current_pos));
                        }
                        if (behavior_ == "Isolated" && match_end > match_start) {
                            new_splits.push_back(s.substr(match_start, match_end - match_start));
                        }
                        // If behavior_ == "Removed", we just don't add the matched part
                    }
                    current_pos = match_end;
                    if (match_start == match_end) { // Handle zero-width matches to avoid infinite loops
                        current_pos++;
                    }
                } else {
                    // No more matches, add the rest of the string
                    if (current_pos < (int)s.size()) {
                        new_splits.push_back(s.substr(current_pos));
                    }
                    break;
                }
            }
        }
        pts.splits = new_splits;
    }
};

// Moved create_bytes_char_map up

class BPEModel : public Model {
public:
    bool use_byte_level_;
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> id_to_token_;
    std::map<std::pair<int, int>, int> merges_;
    mutable std::unordered_map<std::string, std::vector<int>> cache_;

    BPEModel(const std::map<std::string, int>& vocab,
             const std::map<std::pair<int, int>, int>& merges,
             const std::map<std::string, int>& added_tokens,
             bool use_byte_level,
             bool byte_fallback)
        : use_byte_level_(use_byte_level) {
        for (auto const& x : vocab) { vocab_[x.first] = x.second; id_to_token_[x.second] = x.first; }
        for (auto const& x : merges) merges_[x.first] = x.second;
    }

    int token_to_id(const std::string& token) const override {
        auto it = vocab_.find(token);
        return (it != vocab_.end()) ? it->second : -1;
    }
    std::string id_to_token(int id) const override {
        auto it = id_to_token_.find(id);
        return (it != id_to_token_.end()) ? it->second : "";
    }
    size_t vocab_size() const override { return vocab_.size(); }

    std::vector<int> tokenize(const std::string& text) const override {
        if (text.empty()) return {};
        auto cit = cache_.find(text);
        if (cit != cache_.end()) return cit->second;
        std::vector<int> out;
        if (use_byte_level_) {
            static auto byte_map = create_bytes_char_map();
            for (unsigned char b : text) {
                int id = token_to_id(byte_map[b]);
                if (id != -1) out.push_back(id);
            }
        } else {
            const uint8_t* ptr = (const uint8_t*)text.c_str();
            size_t len = text.length(), off = 0;
            int32_t cp;
            while (off < len) {
                ssize_t ret = utf8proc_iterate(ptr + off, len - off, &cp);
                if (ret <= 0) {
                    char buf[16]; snprintf(buf, sizeof(buf), "<0x%02X>", (unsigned char)ptr[off]);
                    int id = token_to_id(buf); if (id != -1) out.push_back(id);
                    off++; continue;
                }
                std::string s((const char*)ptr + off, ret);
                int id = token_to_id(s);
                if (id != -1) out.push_back(id);
                else {
                    for (size_t i = 0; i < (size_t)ret; ++i) {
                        char buf[16]; snprintf(buf, sizeof(buf), "<0x%02X>", (unsigned char)ptr[off+i]);
                        int bid = token_to_id(buf); if (bid != -1) out.push_back(bid);
                    }
                }
                off += ret;
            }
        }
        while (out.size() > 1) {
            int best = -1, min_r = 1e9;
            for (size_t i = 0; i < out.size() - 1; ++i) {
                auto it = merges_.find({out[i], out[i+1]});
                if (it != merges_.end() && it->second < min_r) { min_r = it->second; best = i; }
            }
            if (best == -1) break;
            std::string m = id_to_token(out[best]) + id_to_token(out[best+1]);
            int nid = token_to_id(m); if (nid == -1) break;
            out[best] = nid; out.erase(out.begin() + best + 1);
        }
        cache_[text] = out;
        return out;
    }

    void load(const nlohmann::json& v, const nlohmann::json& m) {
        for (auto it = v.begin(); it != v.end(); ++it) { vocab_[it.key()] = it.value(); id_to_token_[it.value()] = it.key(); }
        int rank = 0;
        for (const auto& item : m) {
            std::string s1, s2;
            if (item.is_string()) {
                std::string line = item.get<std::string>(); size_t p = line.find(' ');
                if (p != std::string::npos) { s1 = line.substr(0, p); s2 = line.substr(p + 1); }
            } else if (item.is_array() && item.size() >= 2) { s1 = item[0].get<std::string>(); s2 = item[1].get<std::string>(); }
            if (!s1.empty() && !s2.empty()) merges_[{token_to_id(s1), token_to_id(s2)}] = rank++;
        }
    }
};

class TemplateProcessing : public PostProcessor {
public:
    struct Step { bool is_token; int id; };
    std::vector<Step> steps_;
    TemplateProcessing(const std::vector<Step>& s) : steps_(s) {}
    void process(Encoding& enc) const override {
        std::vector<int> out;
        for (const auto& s : steps_) {
            if (s.is_token) { if (s.id != -1) out.push_back(s.id); }
            else out.insert(out.end(), enc.input_ids.begin(), enc.input_ids.end());
        }
        enc.input_ids = out;
        enc.attention_mask.assign(out.size(), 1);
    }
};

class ReplaceDecoder : public Decoder {
    std::string pattern_, content_;
public:
    ReplaceDecoder(const std::string& p, const std::string& c) : pattern_(p), content_(c) {}
    void decode(std::vector<std::string>& tokens) const override {
        for (auto& t : tokens) {
            size_t pos = 0;
            while ((pos = t.find(pattern_, pos)) != std::string::npos) {
                t.replace(pos, pattern_.length(), content_);
                pos += content_.length();
            }
        }
    }
};

class StripDecoder : public Decoder {
    std::string content_;
    int start_, stop_;
public:
    StripDecoder(const std::string& c, int start, int stop) : content_(c), start_(start), stop_(stop) {}
    void decode(std::vector<std::string>& tokens) const override {
        if (tokens.empty()) return;
        if (start_ > 0 && !tokens[0].empty() && tokens[0].find(content_) == 0) {
            tokens[0] = tokens[0].substr(content_.length());
        }
        if (stop_ > 0 && !tokens.back().empty()) {
            size_t pos = tokens.back().rfind(content_);
            if (pos != std::string::npos && pos + content_.length() == tokens.back().length()) {
                tokens.back() = tokens.back().substr(0, pos);
            }
        }
    }
};

class FuseDecoder : public Decoder {
public:
    void decode(std::vector<std::string>& tokens) const override {
        if (tokens.size() <= 1) return;
        std::string fused;
        for (const auto& t : tokens) fused += t;
        tokens = {fused};
    }
};

class ByteFallbackDecoder : public Decoder {
public:
    void decode(std::vector<std::string>& tokens) const override {
        for (auto& t : tokens) {
            if (t.length() >= 3 && t.substr(0, 3) == "<0x") {
                int b; if (sscanf(t.c_str(), "<0x%02X>", &b) == 1) t = std::string(1, (char)b);
            }
        }
    }
};

class ByteLevelDecoder : public Decoder {
public:
    void decode(std::vector<std::string>& tokens) const override {
        static auto bm = []() {
            std::unordered_map<std::string, unsigned char> m;
            for (const auto& p : create_bytes_char_map()) m[p.second] = p.first;
            return m;
        }();
        for (auto& t : tokens) {
            std::string out;
            for (size_t i = 0; i < t.length(); ) {
                const uint8_t* tp = (const uint8_t*)t.c_str();
                int32_t cp; ssize_t r = utf8proc_iterate(tp + i, t.length() - i, &cp);
                if (r > 0) {
                    std::string ch(t.substr(i, r)); auto it = bm.find(ch);
                    if (it != bm.end()) out += (char)it->second; else out += ch;
                    i += r;
                } else out += t[i++];
            }
            t = out;
        }
    }
};

class SequenceDecoder : public Decoder {
    std::vector<std::shared_ptr<Decoder>> decoders_;
public:
    SequenceDecoder(const std::vector<std::shared_ptr<Decoder>>& d) : decoders_(d) {}
    void decode(std::vector<std::string>& tokens) const override {
        for (const auto& d : decoders_) d->decode(tokens);
    }
};

class CoreDecoder : public Decoder {
public:
    std::shared_ptr<Model> model_;
    CoreDecoder(std::shared_ptr<Model> m) : model_(m) {}
    void decode(std::vector<std::string>& tokens) const override { /* Not used in this design */ }
};

// ==========================================
// PreTrainedTokenizer::Impl
// ==========================================

struct PreTrainedTokenizer::Impl {
    struct AddedToken { int id; std::string content; bool special; bool lstrip; bool rstrip; bool normalized; };
    std::shared_ptr<Normalizer> normalizer_;
    std::shared_ptr<PreTokenizer> pre_tokenizer_;
    std::shared_ptr<Model> model_;
    std::shared_ptr<PostProcessor> post_processor_;
    std::shared_ptr<Decoder> decoder_;
    struct { int pad=-1, bos=-1, eos=-1, unk=-1; } special_tokens_;
    std::shared_ptr<OnigRegex> added_tokens_regex_;
    std::vector<AddedToken> added_tokens_;
    std::string chat_template_;
    std::shared_ptr<jinja::Template> jinja_template_;

    std::vector<int> encode(const PreTrainedTokenizer* public_api, const std::string& text, bool add_special_tokens) const {
        if (text.empty()) return {};
        std::vector<int> input_ids;

        // 1. Identify added tokens in original text (assuming normalized: false for most)
        std::vector<std::pair<std::string, bool>> units;
        size_t last = 0;
        while (last < text.length()) {
            int match_start = -1, match_end = -1;
            if (added_tokens_regex_ && added_tokens_regex_->search(text, (int)last, (int)text.length(), match_start, match_end)) {
                std::string match_token = text.substr(match_start, match_end - match_start);
                const AddedToken* at = nullptr;
                for (const auto& t : added_tokens_) { if (t.content == match_token) { at = &t; break; } }

                size_t prefix_start = last;
                size_t prefix_end = match_start;
                size_t next_start = match_end;

                if (at) {
                    if (at->lstrip) {
                        while (prefix_end > prefix_start && isspace((unsigned char)text[prefix_end - 1])) prefix_end--;
                    }
                    if (at->rstrip) {
                        while (next_start < text.length() && isspace((unsigned char)text[next_start])) next_start++;
                    }
                }

                if (prefix_end > prefix_start) units.push_back({text.substr(prefix_start, prefix_end - prefix_start), false});
                units.push_back({match_token, true});
                last = next_start;
            } else {
                units.push_back({text.substr(last), false});
                break;
            }
        }

        if (add_special_tokens && special_tokens_.bos != -1) input_ids.push_back(special_tokens_.bos);

        for (const auto& unit : units) {
            if (unit.second) {
                int id = public_api->token_to_id(unit.first);
                if (id != -1) input_ids.push_back(id);
            } else {
                // 2. Normalize only non-special units
                std::string normalized = normalizer_ ? normalizer_->normalize(unit.first) : unit.first;
                if (normalized.empty()) continue;

                // 3. Pre-tokenize and model tokenize
                PreTokenizedString pts; pts.splits.push_back(normalized);
                if (pre_tokenizer_) pre_tokenizer_->pre_tokenize(pts);
                for (const auto& s : pts.splits) {
                    auto ids = model_->tokenize(s);
                    input_ids.insert(input_ids.end(), ids.begin(), ids.end());
                }
            }
        }

        if (add_special_tokens && special_tokens_.eos != -1) input_ids.push_back(special_tokens_.eos);
        return input_ids;
    }

    bool load_from_json(PreTrainedTokenizer* public_api, const nlohmann::json& j) {
        if (j.contains("model")) {
            auto vocab = j["model"]["vocab"].get<std::map<std::string, int>>();
            std::map<std::pair<int, int>, int> merges;
            if (j["model"].contains("merges")) {
                int rank = 0;
                for (const auto& item : j["model"]["merges"]) {
                    std::string s1, s2;
                    if (item.is_string()) {
                        std::string line = item.get<std::string>();
                        size_t pos = line.find(' ');
                        if (pos != std::string::npos) { s1 = line.substr(0, pos); s2 = line.substr(pos + 1); }
                    } else if (item.is_array() && item.size() >= 2) {
                        s1 = item[0].get<std::string>();
                        s2 = item[1].get<std::string>();
                    }
                    if (!s1.empty() && !s2.empty() && vocab.count(s1) && vocab.count(s2))
                        merges[{vocab[s1], vocab[s2]}] = rank++;
                }
            }
            std::map<std::string, int> added_tokens;
            bool byte_fallback = false;
            if (j["model"].contains("byte_fallback")) byte_fallback = j["model"]["byte_fallback"].get<bool>();

            bool use_byte_level = false;
            auto check_bl = [](const nlohmann::json& c) -> bool {
                if (!c.is_object()) return false;
                if (c.value("type", "") == "ByteLevel") return true;
                if (c.contains("pretokenizers")) {
                    for (const auto& s : c["pretokenizers"]) if (s.is_object() && s.value("type", "") == "ByteLevel") return true;
                }
                if (c.contains("processors")) {
                    for (const auto& s : c["processors"]) if (s.is_object() && s.value("type", "") == "ByteLevel") return true;
                }
                if (c.contains("decoders")) {
                    for (const auto& s : c["decoders"]) if (s.is_object() && s.value("type", "") == "ByteLevel") return true;
                }
                return false;
            };
            if (check_bl(j.value("pre_tokenizer", nlohmann::json()))) use_byte_level = true;
            if (check_bl(j.value("post_processor", nlohmann::json()))) use_byte_level = true;
            if (check_bl(j.value("decoder", nlohmann::json()))) use_byte_level = true;

            // If we have a ByteLevelPreTokenizer in the sequence, BPEModel should not do the mapping itself
            // to avoid double mapping.
            bool pt_has_byte_level = false;
            if (j.contains("pre_tokenizer") && j["pre_tokenizer"].is_object()) {
                auto pt = j["pre_tokenizer"];
                if (pt.value("type", "") == "ByteLevel") pt_has_byte_level = true;
                else if (pt.value("type", "") == "Sequence" && pt.contains("pretokenizers")) {
                    for (const auto& s : pt["pretokenizers"]) if (s.is_object() && s.value("type", "") == "ByteLevel") pt_has_byte_level = true;
                }
            }

            auto bpe = std::make_shared<BPEModel>(vocab, merges, added_tokens, use_byte_level && !pt_has_byte_level, byte_fallback);
            this->model_ = bpe;
        }
        if (j.contains("normalizer") && !j["normalizer"].is_null()) {
            auto create_norm = [&](const nlohmann::json& s) -> std::shared_ptr<Normalizer> {
                std::string type = s.value("type", "");
                if (type == "NFKC") return std::make_shared<NFKCNormalizer>();
                if (type == "Prepend") return std::make_shared<PrependNormalizer>(s.value("prepend", ""));
                if (type == "Replace") {
                    std::string p;
                    if (s["pattern"].is_object()) p = s["pattern"].value("String", "");
                    else if (s["pattern"].is_string()) p = s["pattern"].get<std::string>();
                    return std::make_shared<ReplaceNormalizer>(p, s.value("content", ""));
                }
                return nullptr;
            };
            if (j["normalizer"].value("type", "") == "Sequence") {
                std::vector<std::shared_ptr<Normalizer>> norms;
                for (const auto& s : j["normalizer"]["normalizers"]) {
                    auto n = create_norm(s);
                    if (n) norms.push_back(n);
                }
                this->normalizer_ = std::make_shared<SequenceNormalizer>(norms);
            } else {
                this->normalizer_ = create_norm(j["normalizer"]);
            }
        }
        if (j.contains("decoder") && !j["decoder"].is_null()) {
            auto create_dec = [&](const nlohmann::json& s) -> std::shared_ptr<Decoder> {
                std::string type = s.value("type", "");
                if (type == "Replace") {
                    std::string p;
                    if (s["pattern"].is_object()) p = s["pattern"].value("String", "");
                    else if (s["pattern"].is_string()) p = s["pattern"].get<std::string>();
                    return std::make_shared<ReplaceDecoder>(p, s.value("content", ""));
                }
                if (type == "ByteFallback") return std::make_shared<ByteFallbackDecoder>();
                if (type == "ByteLevel") return std::make_shared<ByteLevelDecoder>();
                if (type == "Fuse") return std::make_shared<FuseDecoder>();
                if (type == "Strip") return std::make_shared<StripDecoder>(s.value("content", ""), s.value("start", 0), s.value("stop", 0));
                return nullptr;
            };
            if (j["decoder"].value("type", "") == "Sequence") {
                std::vector<std::shared_ptr<Decoder>> decs;
                for (const auto& s : j["decoder"]["decoders"]) {
                    auto d = create_dec(s);
                    if (d) decs.push_back(d);
                }
                this->decoder_ = std::make_shared<SequenceDecoder>(decs);
            } else {
                this->decoder_ = create_dec(j["decoder"]);
            }
        }
        if (!this->decoder_) {
            // Default decoder if none specified
            this->decoder_ = std::make_shared<ByteLevelDecoder>();
        }
        if (j.contains("pre_tokenizer") && !j["pre_tokenizer"].is_null()) {
            auto pt = j["pre_tokenizer"];
            auto create_pt = [&](const nlohmann::json& s) -> std::shared_ptr<PreTokenizer> {
                std::string type = s.value("type", "");
                if (type == "Split") {
                    std::string p;
                    if (s.contains("pattern")) {
                        if (s["pattern"].is_object()) p = s["pattern"].value("Regex", "");
                        else if (s["pattern"].is_string()) p = s["pattern"].get<std::string>();
                    }
                    if (!p.empty()) return std::make_shared<SplitPreTokenizer>(p, s.value("invert", false), s.value("behavior", "Isolated"));
                } else if (type == "Metaspace") {
                    return std::make_shared<MetaspacePreTokenizer>(s.value("str_rep", "▁"), s.value("add_prefix_space", true));
                } else if (type == "ByteLevel") {
                    return std::make_shared<ByteLevelPreTokenizer>(s.value("use_regex", true));
                } else if (type == "Digits") {
                    return std::make_shared<DigitsPreTokenizer>(s.value("individual_digits", false));
                }
                return nullptr;
            };
            if (pt.value("type", "") == "Sequence" && pt.contains("pretokenizers")) {
                std::vector<std::shared_ptr<PreTokenizer>> pts;
                for (const auto& s : pt["pretokenizers"]) {
                    auto p = create_pt(s); if (p) pts.push_back(p);
                }
                this->pre_tokenizer_ = std::make_shared<SequencePreTokenizer>(pts);
            } else {
                this->pre_tokenizer_ = create_pt(pt);
            }
        }
        if (j.contains("post_processor") && !j["post_processor"].is_null()) {
            auto pp = j["post_processor"];
            auto ptl = [&](const nlohmann::json& s) {
                std::vector<TemplateProcessing::Step> steps;
                if (s.contains("single")) {
                    for (auto& i : s["single"]) {
                        if (i.contains("SpecialToken")) steps.push_back({true, public_api->token_to_id(i["SpecialToken"]["id"].get<std::string>())});
                        else if (i.contains("Sequence")) steps.push_back({false, 0});
                    }
                    this->post_processor_ = std::make_shared<TemplateProcessing>(steps);
                }
            };
            if (pp.value("type", "") == "TemplateProcessing") ptl(pp);
            else if (pp.value("type", "") == "Sequence" && pp.contains("processors")) { for (auto& s : pp["processors"]) if (s.value("type", "") == "TemplateProcessing") { ptl(s); break; } }
        }
        if (j.contains("added_tokens") && j["added_tokens"].is_array()) {
            std::vector<std::string> cs;
            for (auto& item : j["added_tokens"]) {
                std::string c = item.value("content", ""); int id = item.value("id", -1);
                bool special = item.value("special", false);
                bool lstrip = item.value("lstrip", false);
                bool rstrip = item.value("rstrip", false);
                bool normalized = item.value("normalized", false);
                if (c.empty() || id == -1) continue;
                cs.push_back(c);
                this->added_tokens_.push_back({id, c, special, lstrip, rstrip, normalized}); // Store added token info
                if (c == "[PAD]" || c == "<pad>") this->special_tokens_.pad = id;
                if (c == "[BOS]" || c == "<s>" || c == "<bos>") this->special_tokens_.bos = id;
                if (c == "[EOS]" || c == "</s>" || c == "<eos>") this->special_tokens_.eos = id;
                if (c == "[UNK]" || c == "<unk>") this->special_tokens_.unk = id;
                auto bpe = std::dynamic_pointer_cast<BPEModel>(this->model_); if (bpe) { bpe->vocab_[c] = id; bpe->id_to_token_[id] = c; }
            }
            if (!cs.empty()) {
                std::sort(cs.begin(), cs.end(), [](const std::string& a, const std::string& b){ return a.length() > b.length(); });
                std::string p; for (size_t i=0; i<cs.size(); ++i) { if (i>0) p += "|"; p += OnigurumaRegexEscape(cs[i]); }
                this->added_tokens_regex_ = std::make_shared<OnigRegex>(p);
            }
        }
        if (j.contains("config_overrides")) {
            auto co = j["config_overrides"];
            if (co.contains("bos_token")) this->special_tokens_.bos = public_api->token_to_id(get_token_content(co["bos_token"]));
            if (co.contains("eos_token")) this->special_tokens_.eos = public_api->token_to_id(get_token_content(co["eos_token"]));
            if (co.contains("pad_token")) this->special_tokens_.pad = public_api->token_to_id(get_token_content(co["pad_token"]));
            if (co.contains("unk_token")) this->special_tokens_.unk = public_api->token_to_id(get_token_content(co["unk_token"]));
        }
        return true;
    }
};

// ==========================================
// PreTrainedTokenizer Public API Implementation
// ==========================================

PreTrainedTokenizer::PreTrainedTokenizer() : impl_(std_make_unique<Impl>()) {}
PreTrainedTokenizer::~PreTrainedTokenizer() = default;

std::vector<int> PreTrainedTokenizer::encode(const std::string& text, bool add_special_tokens) const {
    return impl_->encode(this, text, add_special_tokens);
}

std::string PreTrainedTokenizer::decode(const std::vector<int>& ids, bool skip_special_tokens) const {
    std::vector<std::string> tokens;
    for (int id : ids) {
        if (skip_special_tokens) {
            // Check if special token
            bool special = false;
            for (const auto& at : impl_->added_tokens_) {
                if (at.id == id && at.special) { special = true; break; }
            }
            if (special) continue;
        }
        std::string t = impl_->model_->id_to_token(id);
        if (!t.empty()) tokens.push_back(t);
    }
    if (impl_->decoder_) impl_->decoder_->decode(tokens);
    std::string out;
    for (const auto& t : tokens) out += t;
    return out;
}

int PreTrainedTokenizer::token_to_id(const std::string& t) const { return impl_->model_ ? impl_->model_->token_to_id(t) : -1; }
std::string PreTrainedTokenizer::id_to_token(int id) const { return impl_->model_ ? impl_->model_->id_to_token(id) : ""; }
int PreTrainedTokenizer::pad_token_id() const { return impl_->special_tokens_.pad; }
int PreTrainedTokenizer::bos_token_id() const { return impl_->special_tokens_.bos; }
int PreTrainedTokenizer::eos_token_id() const { return impl_->special_tokens_.eos; }
int PreTrainedTokenizer::unk_token_id() const { return impl_->special_tokens_.unk; }

void PreTrainedTokenizer::set_chat_template(const std::string& t) {
    impl_->chat_template_ = t;
    impl_->jinja_template_ = std::make_shared<jinja::Template>(t);
}

std::string PreTrainedTokenizer::apply_chat_template(const ChatMessages& msgs, bool add_gen) const {
    if (!impl_->jinja_template_) return "";
    nlohmann::json j_msgs = nlohmann::json::array();
    for (const auto& m : msgs) j_msgs.push_back({{"role", m.first}, {"content", m.second}});
    nlohmann::json extra;
    extra["bos_token"] = id_to_token(impl_->special_tokens_.bos);
    extra["eos_token"] = id_to_token(impl_->special_tokens_.eos);
    return impl_->jinja_template_->apply_chat_template(j_msgs, add_gen, nlohmann::json::array(), extra);
}

std::string PreTrainedTokenizer::apply_chat_template(const std::string& json_str, bool add_generation_prompt) const {
    if (!impl_->jinja_template_) return "";
    auto j_msgs = nlohmann::json::parse(json_str, nullptr, false);
    if (!j_msgs.is_array()) return "";
    nlohmann::json extra;
    extra["bos_token"] = id_to_token(impl_->special_tokens_.bos);
    extra["eos_token"] = id_to_token(impl_->special_tokens_.eos);
    return impl_->jinja_template_->apply_chat_template(j_msgs, add_generation_prompt, nlohmann::json::array(), extra);
}

bool PreTrainedTokenizer::load_from_json_str(const std::string& json_str) {
    auto j = nlohmann::json::parse(json_str, nullptr, false);
    if (j.is_discarded()) return false;
    return impl_->load_from_json(this, j);
}

// ==========================================
// AutoTokenizer Implementation
// ==========================================

std::shared_ptr<PreTrainedTokenizer> AutoTokenizer::from_pretrained(const std::string& path) {
    auto tok = std::make_shared<PreTrainedTokenizer>();
    std::ifstream f(path + "/tokenizer.json"); if (!f.is_open()) return nullptr;
    nlohmann::json j; f >> j;
    std::ifstream fc(path + "/tokenizer_config.json");
    if (fc.is_open()) {
        nlohmann::json jc; fc >> jc; if (jc.contains("chat_template")) tok->set_chat_template(jc["chat_template"].get<std::string>());
        j["config_overrides"] = jc;
    }
    std::stringstream ss; ss << j;
    if (!tok->load_from_json_str(ss.str())) return nullptr;
    return tok;
}

} // namespace tokenizer