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

#include "ye3t_sha256.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace YE3T_LAMMPS {
namespace {

  constexpr std::array<std::uint32_t, 64> round_constants{
      0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
      0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
      0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
      0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
      0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
      0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
      0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
      0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
      0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
      0xc67178f2U};

  std::uint32_t rotate_right(std::uint32_t value, unsigned int amount)
  {
    return (value >> amount) | (value << (32U - amount));
  }

  std::string encode_digest(const std::array<unsigned char, 32> &digest)
  {
    std::ostringstream encoded;
    encoded << std::hex << std::setfill('0');
    for (const unsigned char byte : digest)
      encoded << std::setw(2) << static_cast<unsigned int>(byte);
    return encoded.str();
  }

}    // namespace

void SHA256Builder::update(const void *data, std::size_t size)
{
  if (finished_) throw std::logic_error("cannot update a finished SHA-256 digest");
  if (size > (std::numeric_limits<std::uint64_t>::max() - bit_count_) / 8U)
    throw std::overflow_error("SHA-256 input is too large");
  if (size > 0 && data == nullptr) throw std::invalid_argument("SHA-256 received null input");
  bit_count_ += static_cast<std::uint64_t>(size) * 8U;
  const auto *bytes = static_cast<const unsigned char *>(data);
  for (std::size_t index = 0; index < size; ++index) {
    block_[block_size_++] = bytes[index];
    if (block_size_ == block_.size()) {
      transform();
      block_size_ = 0;
    }
  }
}

void SHA256Builder::update(const std::string &value)
{
  update(value.data(), value.size());
}

void SHA256Builder::transform()
{
  std::array<std::uint32_t, 64> words{};
  for (std::size_t index = 0; index < 16; ++index) {
    const std::size_t offset = index * 4;
    words[index] = (static_cast<std::uint32_t>(block_[offset]) << 24) |
        (static_cast<std::uint32_t>(block_[offset + 1]) << 16) |
        (static_cast<std::uint32_t>(block_[offset + 2]) << 8) |
        static_cast<std::uint32_t>(block_[offset + 3]);
  }
  for (std::size_t index = 16; index < words.size(); ++index) {
    const std::uint32_t s0 = rotate_right(words[index - 15], 7) ^
        rotate_right(words[index - 15], 18) ^ (words[index - 15] >> 3);
    const std::uint32_t s1 = rotate_right(words[index - 2], 17) ^
        rotate_right(words[index - 2], 19) ^ (words[index - 2] >> 10);
    words[index] = words[index - 16] + s0 + words[index - 7] + s1;
  }

  std::uint32_t a = state_[0];
  std::uint32_t b = state_[1];
  std::uint32_t c = state_[2];
  std::uint32_t d = state_[3];
  std::uint32_t e = state_[4];
  std::uint32_t f = state_[5];
  std::uint32_t g = state_[6];
  std::uint32_t h = state_[7];
  for (std::size_t index = 0; index < words.size(); ++index) {
    const std::uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
    const std::uint32_t choose = (e & f) ^ ((~e) & g);
    const std::uint32_t temporary1 = h + sum1 + choose + round_constants[index] + words[index];
    const std::uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
    const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    const std::uint32_t temporary2 = sum0 + majority;
    h = g;
    g = f;
    f = e;
    e = d + temporary1;
    d = c;
    c = b;
    b = a;
    a = temporary1 + temporary2;
  }
  state_[0] += a;
  state_[1] += b;
  state_[2] += c;
  state_[3] += d;
  state_[4] += e;
  state_[5] += f;
  state_[6] += g;
  state_[7] += h;
}

std::string SHA256Builder::finish()
{
  if (finished_) throw std::logic_error("SHA-256 digest was already finished");
  finished_ = true;
  block_[block_size_++] = 0x80U;
  if (block_size_ > 56) {
    while (block_size_ < block_.size()) block_[block_size_++] = 0;
    transform();
    block_size_ = 0;
  }
  while (block_size_ < 56) block_[block_size_++] = 0;
  for (int byte = 7; byte >= 0; --byte)
    block_[block_size_++] = static_cast<unsigned char>(bit_count_ >> (byte * 8));
  transform();

  std::array<unsigned char, 32> digest{};
  for (std::size_t word = 0; word < state_.size(); ++word)
    for (int byte = 0; byte < 4; ++byte)
      digest[word * 4 + static_cast<std::size_t>(byte)] =
          static_cast<unsigned char>(state_[word] >> (24 - byte * 8));
  return encode_digest(digest);
}

std::string sha256_file(const std::string &path)
{
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("could not open file for SHA-256: " + path);
  SHA256Builder sha;
  std::array<unsigned char, 1024 * 64> buffer{};
  while (stream) {
    stream.read(reinterpret_cast<char *>(buffer.data()), buffer.size());
    const std::streamsize count = stream.gcount();
    if (count > 0) sha.update(buffer.data(), static_cast<std::size_t>(count));
  }
  if (!stream.eof()) throw std::runtime_error("failed while hashing file: " + path);
  return sha.finish();
}

std::string sha256_string(const std::string &value)
{
  SHA256Builder sha;
  sha.update(value);
  return sha.finish();
}

}    // namespace YE3T_LAMMPS
