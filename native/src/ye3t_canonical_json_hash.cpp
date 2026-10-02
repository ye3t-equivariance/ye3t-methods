/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: James M. Goff (Sandia National Laboratories)
------------------------------------------------------------------------- */

#include "ye3t_canonical_json_hash.h"

#include "ye3t_sha256.h"

#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace YE3T_LAMMPS {
namespace {

  struct ObjectMember {
    std::string key;
    std::size_t member_begin = 0;
    std::size_t member_end = 0;
    std::size_t value_begin = 0;
    std::size_t value_end = 0;
  };

  class CanonicalJSONScanner {
   public:
    explicit CanonicalJSONScanner(std::string source) : source_(std::move(source))
    {
      for (const unsigned char byte : source_)
        if (byte >= 0x80U) fail("execution-plan JSON must use ASCII or JSON Unicode escapes");
    }

    CanonicalExecutionPlanHashes hashes()
    {
      scan_root();

      const ObjectMember *plan_hash = member("plan_hash");
      const ObjectMember *tables = member("synthesis_tables");
      if (plan_hash == nullptr || tables == nullptr)
        fail("execution-plan root omits plan_hash or synthesis_tables");

      std::string plan_payload{"{"};
      bool first = true;
      for (const auto &entry : root_members_) {
        if (entry.key == "plan_hash") continue;
        if (!first) plan_payload.push_back(',');
        first = false;
        plan_payload.append(compact_, entry.member_begin, entry.member_end - entry.member_begin);
      }
      plan_payload.push_back('}');

      return {sha256_string(
                  compact_.substr(tables->value_begin, tables->value_end - tables->value_begin)),
              sha256_string(plan_payload)};
    }

    std::string hash_without_root_member(const std::string &omitted_member)
    {
      scan_root();
      if (member(omitted_member) == nullptr) fail("root omits member " + omitted_member);
      std::string payload{"{"};
      bool first = true;
      for (const auto &entry : root_members_) {
        if (entry.key == omitted_member) continue;
        if (!first) payload.push_back(',');
        first = false;
        payload.append(compact_, entry.member_begin, entry.member_end - entry.member_begin);
      }
      payload.push_back('}');
      return sha256_string(payload);
    }

    std::string root_member_value(const std::string &key)
    {
      scan_root();
      const ObjectMember *entry = member(key);
      if (entry == nullptr) fail("root omits member " + key);
      return compact_.substr(entry->value_begin, entry->value_end - entry->value_begin);
    }

   private:
    [[noreturn]] void fail(const std::string &message) const
    {
      throw std::runtime_error("canonical execution-plan JSON: " + message);
    }

    static bool whitespace(char value)
    {
      return value == ' ' || value == '\t' || value == '\n' || value == '\r';
    }

    static int hexadecimal_value(char value)
    {
      if (value >= '0' && value <= '9') return value - '0';
      if (value >= 'a' && value <= 'f') return value - 'a' + 10;
      if (value >= 'A' && value <= 'F') return value - 'A' + 10;
      return -1;
    }

    void scan_root()
    {
      skip_whitespace();
      if (peek() != '{') fail("root must be an object");
      parse_object(0, true);
      skip_whitespace();
      if (position_ != source_.size()) fail("trailing content after root");
    }

    void skip_whitespace()
    {
      while (position_ < source_.size() && whitespace(source_[position_])) ++position_;
    }

    char peek() const
    {
      if (position_ == source_.size()) fail("unexpected end of input");
      return source_[position_];
    }

    void punctuation(char expected)
    {
      skip_whitespace();
      if (peek() != expected) fail(std::string("expected '") + expected + "'");
      ++position_;
      compact_.push_back(expected);
    }

    std::string parse_string()
    {
      skip_whitespace();
      if (peek() != '"') fail("expected a JSON string");
      compact_.push_back(source_[position_++]);
      std::string decoded;
      while (position_ < source_.size()) {
        const char value = source_[position_++];
        compact_.push_back(value);
        if (value == '"') return decoded;
        if (static_cast<unsigned char>(value) < 0x20U) fail("unescaped control byte in string");
        if (value != '\\') {
          decoded.push_back(value);
          continue;
        }
        if (position_ == source_.size()) fail("truncated string escape");
        const char escape = source_[position_++];
        compact_.push_back(escape);
        switch (escape) {
          case '"':
          case '\\':
          case '/':
            decoded.push_back(escape);
            break;
          case 'b':
            decoded.push_back('\b');
            break;
          case 'f':
            decoded.push_back('\f');
            break;
          case 'n':
            decoded.push_back('\n');
            break;
          case 'r':
            decoded.push_back('\r');
            break;
          case 't':
            decoded.push_back('\t');
            break;
          case 'u': {
            int codepoint = 0;
            for (int digit = 0; digit < 4; ++digit) {
              if (position_ == source_.size()) fail("truncated Unicode escape");
              const char encoded = source_[position_++];
              compact_.push_back(encoded);
              const int nibble = hexadecimal_value(encoded);
              if (nibble < 0) fail("invalid Unicode escape");
              codepoint = 16 * codepoint + nibble;
            }
            if (codepoint > 0x7f) fail("non-ASCII Unicode escape is unsupported by this profile");
            decoded.push_back(static_cast<char>(codepoint));
            break;
          }
          default:
            fail("invalid string escape");
        }
      }
      fail("unterminated string");
    }

    void parse_number()
    {
      skip_whitespace();
      const std::size_t begin = position_;
      if (source_[position_] == '-') ++position_;
      if (position_ == source_.size()) fail("truncated number");
      if (source_[position_] == '0') {
        ++position_;
        if (position_ < source_.size() && source_[position_] >= '0' && source_[position_] <= '9')
          fail("leading zero in number");
      } else {
        if (source_[position_] < '1' || source_[position_] > '9') fail("invalid number");
        while (position_ < source_.size() && source_[position_] >= '0' && source_[position_] <= '9')
          ++position_;
      }
      if (position_ < source_.size() && source_[position_] == '.') {
        ++position_;
        const std::size_t fraction = position_;
        while (position_ < source_.size() && source_[position_] >= '0' && source_[position_] <= '9')
          ++position_;
        if (position_ == fraction) fail("fraction has no digits");
      }
      if (position_ < source_.size() && (source_[position_] == 'e' || source_[position_] == 'E')) {
        ++position_;
        if (position_ < source_.size() && (source_[position_] == '+' || source_[position_] == '-'))
          ++position_;
        const std::size_t exponent = position_;
        while (position_ < source_.size() && source_[position_] >= '0' && source_[position_] <= '9')
          ++position_;
        if (position_ == exponent) fail("exponent has no digits");
      }
      compact_.append(source_, begin, position_ - begin);
    }

    void parse_literal(const std::string &literal)
    {
      skip_whitespace();
      if (source_.compare(position_, literal.size(), literal) != 0) fail("invalid JSON literal");
      position_ += literal.size();
      compact_ += literal;
    }

    void parse_array(std::size_t depth)
    {
      punctuation('[');
      skip_whitespace();
      if (peek() == ']') {
        punctuation(']');
        return;
      }
      while (true) {
        parse_value(depth + 1);
        skip_whitespace();
        if (peek() == ']') {
          punctuation(']');
          return;
        }
        punctuation(',');
      }
    }

    void parse_object(std::size_t depth, bool root)
    {
      punctuation('{');
      skip_whitespace();
      if (peek() == '}') {
        punctuation('}');
        return;
      }
      std::string previous_key;
      bool first = true;
      while (true) {
        const std::size_t member_begin = compact_.size();
        const std::string key = parse_string();
        if (!first && key <= previous_key) fail("object keys must be unique and sorted");
        first = false;
        previous_key = key;
        punctuation(':');
        const std::size_t value_begin = compact_.size();
        parse_value(depth + 1);
        const std::size_t value_end = compact_.size();
        if (root) root_members_.push_back({key, member_begin, value_end, value_begin, value_end});
        skip_whitespace();
        if (peek() == '}') {
          punctuation('}');
          return;
        }
        punctuation(',');
      }
    }

    void parse_value(std::size_t depth)
    {
      if (depth > 512) fail("nesting depth exceeds 512");
      skip_whitespace();
      switch (peek()) {
        case '{':
          parse_object(depth, false);
          return;
        case '[':
          parse_array(depth);
          return;
        case '"':
          (void) parse_string();
          return;
        case 't':
          parse_literal("true");
          return;
        case 'f':
          parse_literal("false");
          return;
        case 'n':
          parse_literal("null");
          return;
        default:
          parse_number();
      }
    }

    const ObjectMember *member(const std::string &key) const
    {
      for (const auto &entry : root_members_)
        if (entry.key == key) return &entry;
      return nullptr;
    }

    std::string source_;
    std::size_t position_ = 0;
    std::string compact_;
    std::vector<ObjectMember> root_members_;
  };

  std::string read_file(const std::string &path)
  {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("could not open execution-plan JSON: " + path);
    std::string result((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    if (stream.bad()) throw std::runtime_error("failed while reading execution-plan JSON: " + path);
    return result;
  }

}    // namespace

CanonicalExecutionPlanHashes canonical_execution_plan_hashes(const std::string &path)
{
  return CanonicalJSONScanner(read_file(path)).hashes();
}

std::string canonical_json_hash_without_root_member(const std::string &path,
                                                    const std::string &omitted_member)
{
  return CanonicalJSONScanner(read_file(path)).hash_without_root_member(omitted_member);
}

std::string canonical_json_root_member_value(const std::string &path, const std::string &member)
{
  return CanonicalJSONScanner(read_file(path)).root_member_value(member);
}

std::string canonical_json_nested_member_value(const std::string &path,
                                               const std::string &object_member,
                                               const std::string &member)
{
  const std::string object = CanonicalJSONScanner(read_file(path)).root_member_value(object_member);
  return CanonicalJSONScanner(object).root_member_value(member);
}

std::string canonical_json_value_hash_without_root_member(const std::string &json_value,
                                                          const std::string &omitted_member)
{
  return CanonicalJSONScanner(json_value).hash_without_root_member(omitted_member);
}

std::string canonical_json_value_root_member(const std::string &json_value,
                                             const std::string &member)
{
  return CanonicalJSONScanner(json_value).root_member_value(member);
}

}    // namespace YE3T_LAMMPS
