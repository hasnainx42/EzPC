#include "library_fixed.h"
#include "cleartext_library_fixed.h"
#include "defines.h"
#include "utils/io_channel.h"
#include "utils/net_io_channel.h"
#include "utils/ArgMapping/ArgMapping.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cassert>

using namespace std;
using namespace sci;

int main(int argc, char **argv) {
  /*********** Command line ***********/
  ArgMapping amap;
  int party;
  string address = "127.0.0.1";
  amap.arg("r", party, "Role: 0=client, 1=server");
  amap.arg("h", address, "Server IP address", string("127.0.0.1"));
  amap.parse(argc, argv);

  /*********** Parameters ***********/
  int llm_dim = 768;
  int latent_dim = 8;
  int bw = 64;        // bitlength
  int fracbits = 16;  // scaling factor
  int theta_range = 3;
  int a_range = 3;

  /*********** Initialize MPC ***********/
  NetIO *io = new NetIO(party == SERVER ? nullptr : address.c_str(), 32000);
  IOPack *iopack = new IOPack(party, io);
  OTPack *otpack = new OTPack(iopack, party);
  initialize();
  cout << "[Party " << party << "] BOLT runtime initialized." << endl;

  /*********** Allocate buffers ***********/
  vector<int64_t> llm(llm_dim), item(llm_dim);
  vector<int64_t> Wtheta(latent_dim * llm_dim), Wa(latent_dim * llm_dim),
      Wb(llm_dim);

  // each share is random; server loads true weights
  if (party == SERVER) {
    ifstream fWtheta("sec_export/Wtheta_i64.txt");
    ifstream fWa("sec_export/Wa_i64.txt");
    ifstream fWb("sec_export/Wb_i64.txt");
    for (int i = 0; i < latent_dim * llm_dim; i++)
      fWtheta >> Wtheta[i];
    for (int i = 0; i < latent_dim * llm_dim; i++)
      fWa >> Wa[i];
    for (int i = 0; i < llm_dim; i++)
      fWb >> Wb[i];
  } else {
    // dummy zeros
    fill(Wtheta.begin(), Wtheta.end(), 0);
    fill(Wa.begin(), Wa.end(), 0);
    fill(Wb.begin(), Wb.end(), 0);
  }

  // inputs
   // ---------- read additive shares from txt ----------
   auto read_vec_txt = [&](const std::string& path, int n) {
    std::vector<int64_t> v(n);
    std::ifstream f(path);
    for (int i = 0; i < n; i++) f >> v[i];
    return v;
  };

  // CLIENT reads its llm share and item share
  // SERVER reads its llm share and item share
  if (party == CLIENT) {
    llm  = read_vec_txt("sec_export/llm_client_share.txt", llm_dim);
    item = read_vec_txt("sec_export/item_client_share.txt", llm_dim); // OK even if meaningless to client
  } else {
    llm  = read_vec_txt("sec_export/llm_server_share.txt", llm_dim);
    item = read_vec_txt("sec_export/item_server_share.txt", llm_dim); // server owns item (its own share)
  }


  /*********** Buffers for intermediates ***********/
  vector<int64_t> theta(latent_dim);
  vector<int64_t> a(latent_dim);
  int64_t b = 0;
  int64_t z = 0;
  int64_t pred = 0;

  /*********** Helper: cubic sigmoid ***********/
  auto sigmoid_cubic = [&](int64_t *X, int64_t *Y, int n) {
    // Y = 0.5 + 0.2159x - 0.00822x^3
    // we reuse existing fixed ops
    vector<int64_t> x2(n), x3(n), t1(n), t3(n);

    MulCir(n, 1, fracbits, fracbits, 0, bw, bw, bw, bw, X, X, x2.data());
    MulCir(n, 1, fracbits, fracbits, 0, bw, bw, bw, bw, x2.data(), X, x3.data());

    ScalarMul(n, 1, fracbits, fracbits, 0, bw, bw, bw, bw,
              (int64_t)(0.2159198015 * (1 << fracbits)), X, t1.data());
    ScalarMul(n, 1, fracbits, fracbits, 0, bw, bw, bw, bw,
              (int64_t)(-0.0082176259 * (1 << fracbits)), x3.data(), t3.data());

    for (int i = 0; i < n; i++) {
      Y[i] = ((int64_t)(0.5 * (1 << fracbits)) + t1[i] + t3[i]);
    }
  };

  /*********** 1. θ = θ_range * σ(Wθ·llm) ***********/
  vector<int64_t> tmp(latent_dim);
  MatMul(latent_dim, llm_dim, 1,
         0, 0, 0, 0, 0,
         bw, bw, bw, bw,
         Wtheta.data(), llm.data(), tmp.data(), nullptr);
  sigmoid_cubic(tmp.data(), theta.data(), latent_dim);
  ScalarMul(latent_dim, 1, fracbits, fracbits, 0, bw, bw, bw, bw,
            (int64_t)(theta_range * (1 << fracbits)), theta.data(), theta.data());

  /*********** 2. a = a_range * σ(Wa·item) ***********/
  MatMul(latent_dim, llm_dim, 1,
         0, 0, 0, 0, 0,
         bw, bw, bw, bw,
         Wa.data(), item.data(), tmp.data(), nullptr);
  sigmoid_cubic(tmp.data(), a.data(), latent_dim);
  ScalarMul(latent_dim, 1, fracbits, fracbits, 0, bw, bw, bw, bw,
            (int64_t)(a_range * (1 << fracbits)), a.data(), a.data());

  /*********** 3. b = Wb·item ***********/
  MatMul(1, llm_dim, 1,
         0, 0, 0, 0, 0,
         bw, bw, bw, bw,
         Wb.data(), item.data(), &b, nullptr);

  /*********** 4. z = (a·θ) - b ***********/
  MatMul(1, latent_dim, 1,
         0, 0, 0, 0, 0,
         bw, bw, bw, bw,
         a.data(), theta.data(), &z, nullptr);
  z -= b;

  /*********** 5. pred = σ(z) ***********/
  sigmoid_cubic(&z, &pred, 1);

  /*********** 6. Reveal ***********/
  vector<int64_t> r_in(1), r_out(1);
  r_in[0] = pred;
  reconstruct(1, r_in.data(), r_out.data(), bw);
  if (party == CLIENT) {
    double fpred = (double)r_out[0] / (1 << fracbits);
    cout << "[CLIENT] Secure prediction = " << fpred << endl;
  }

  finalize();
  delete otpack;
  delete iopack;
  delete io;
  return 0;
}
