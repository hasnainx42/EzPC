#include "library_fixed.h"
#include "cleartext_library_fixed.h"
#include "defines.h"
#include "globals.h"  
#include "utils/io_channel.h"
#include "utils/net_io_channel.h"
#include "utils/ArgMapping/ArgMapping.h"

#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace sci;

static vector<int64_t> read_vec_txt(const string &path, int n) {
  vector<int64_t> v(n);
  ifstream f(path);
  if (!f) {
    cerr << "Failed to open: " << path << endl;
    exit(1);
  }
  for (int i = 0; i < n; i++) f >> v[i];
  return v;
}

static void sigmoid_cubic(const int64_t *X, int64_t *Y, int n, int bw, int fracbits) {
  // Y = 0.5 + 0.2159198015 * x - 0.0082176259 * x^3
  vector<int64_t> x2(n), x3(n), t1(n), t3(n);

  // x2 = X * X
  MulCir(n, 1, fracbits, fracbits, 0, bw, bw, bw, bw,
         const_cast<int64_t*>(X), const_cast<int64_t*>(X), x2.data());
  // x3 = x2 * X
  MulCir(n, 1, fracbits, fracbits, 0, bw, bw, bw, bw,
         x2.data(), const_cast<int64_t*>(X), x3.data());

  const int64_t C1  = (int64_t) llround( 0.2159198015 * (1LL << fracbits));
  const int64_t C3n = (int64_t) llround(-0.0082176259 * (1LL << fracbits));
  ScalarMul(n, 1, fracbits, fracbits, 0, bw, bw, bw, bw, C1,
            const_cast<int64_t*>(X), t1.data());
  ScalarMul(n, 1, fracbits, fracbits, 0, bw, bw, bw, bw, C3n, x3.data(), t3.data());

  const int64_t C05 = (int64_t)(0.5 * (1LL << fracbits));
  for (int i = 0; i < n; i++) Y[i] = C05 + t1[i] + t3[i];
}
int bitlength = 64;     // default; will set again after parsing
int num_threads = 1;    // default; will set again after parsing

int main(int argc, char **argv) {
  // --- Args ---
  ArgMapping amap;
  int party;                      // 0=client, 1=server
  string address = "127.0.0.1";   // set default yourself; ArgMapping doesn’t set defaults
  amap.arg("r", party, "Role: 0=client, 1=server");
  amap.arg("h", address, "Server IP address");
  amap.parse(argc, argv);

  // --- Model & numeric params (must match your exporters/scripts) ---
  const int D = 768;            // embedding dim
  const int H = 8;              // latent dim
  const int bw = 64;            // bitlength
  const int fracbits = 16;      // scale S = 2^16
  const int theta_range = 3;    // integer multiplier
  const int a_range = 3;        // integer multiplier
  const int bw = 64;
  const int fracbits = 16;

  // --- MPC init (use IOPack signature in your tree) ---
  IOPack *iopack = new IOPack(party, /*port*/32000, address);
  OTPack *otpack = new OTPack(iopack, party);
  initialize();
  +  // --- Wire SCI globals and let initialize() allocate I/OT packs ---
+  sci::party       = party;             // 0/1
+  sci::address     = address;           // e.g., "127.0.0.1"
+  sci::port        = 32000;             // same as you used before
+  sci::bitlength   = bw;                // 64
+  sci::num_threads = 1;
   sci::bitlength   = bw;
   sci::num_threads = 1;  
   
   bitlength   = bw;   // <--- bridge for files that use plain globals
   num_threads = 1;// or higher if you want
+  initialize();    
                       // sets up global iopack/otpack
  cout << "[Party " << party << "] BOLT init OK\n";

  // --- Load weights (server), zeros (client) ---
  vector<int64_t> Wtheta(H * D), Wa(H * D), Wb(D);
  if (party == SERVER) {
    Wtheta = read_vec_txt("sec_export/Wtheta_i64.txt", H * D);
    Wa     = read_vec_txt("sec_export/Wa_i64.txt",     H * D);
    Wb     = read_vec_txt("sec_export/Wb_i64.txt",     D);
  } else {
    fill(Wtheta.begin(), Wtheta.end(), 0);
    fill(Wa.begin(),     Wa.end(),     0);
    fill(Wb.begin(),     Wb.end(),     0);
  }

  // --- Load additive shares for inputs ---
  vector<int64_t> llm, item;
  if (party == CLIENT) {
    llm  = read_vec_txt("sec_export/llm_client_share.txt",  D);
    item = read_vec_txt("sec_export/item_client_share.txt", D);
  } else {
    llm  = read_vec_txt("sec_export/llm_server_share.txt",  D);
    item = read_vec_txt("sec_export/item_server_share.txt", D);
  }

  // --- Buffers ---
  vector<int64_t> theta(H), a(H);
  int64_t b = 0, z = 0, pred = 0;

  // tmp buffers (pattern used in repo): size >= I*K + K*J
  vector<int64_t> tmp1(H * D + D * 1);
  vector<int64_t> tmp2(H * D + D * 1);
  vector<int64_t> tmp3(1 * H + H * 1);
  vector<int64_t> tmpb(1 * D + D * 1);

  // 1) theta_lin = Wtheta * llm  --> [H,1] in theta
  MatMul(/*I*/H, /*K*/D, /*J*/1,
         /*shrA*/0, /*shrB*/0, /*H1*/0, /*H2*/0, /*demote*/0,
         /*bwA*/bw, /*bwB*/bw, /*bwTemp*/bw, /*bwC*/bw,
         Wtheta.data(), llm.data(), theta.data(), tmp1.data());

  // theta = theta_range * sigmoid_cubic(theta_lin)
  sigmoid_cubic(theta.data(), theta.data(), H, bw, fracbits);
  {
    const int64_t scale_tr = (int64_t)(theta_range * (1LL << fracbits));
    ScalarMul(H, 1, fracbits, fracbits, 0, bw, bw, bw, bw, scale_tr,
              theta.data(), theta.data());
  }

  // 2) a_lin = Wa * item  --> [H,1] in a
  MatMul(H, D, 1, 0, 0, 0, 0, 0, bw, bw, bw, bw,
         Wa.data(), item.data(), a.data(), tmp2.data());

  // a = a_range * sigmoid_cubic(a_lin)
  sigmoid_cubic(a.data(), a.data(), H, bw, fracbits);
  {
    const int64_t scale_ar = (int64_t)(a_range * (1LL << fracbits));
    ScalarMul(H, 1, fracbits, fracbits, 0, bw, bw, bw, bw, scale_ar,
              a.data(), a.data());
  }

  // 3) b = Wb * item  (1xD)(Dx1)->[1,1]
  MatMul(/*I*/1, /*K*/D, /*J*/1, 0, 0, 0, 0, 0, bw, bw, bw, bw,
         Wb.data(), item.data(), &b, tmpb.data());

  // 4) z = (a·theta) - b
  MatMul(/*I*/1, /*K*/H, /*J*/1, 0, 0, 0, 0, 0, bw, bw, bw, bw,
         a.data(), theta.data(), &z, tmp3.data());
  z -= b;

  // 5) pred = sigmoid_cubic(z)
  sigmoid_cubic(&z, &pred, 1, bw, fracbits);

  // 6) Reveal to client — reconstruct wants uint64_t*
  uint64_t rin[1], rout[1];
  rin[0]  = static_cast<uint64_t>(pred);
  rout[0] = 0;
  reconstruct(/*dim*/1, /*x*/rin, /*y*/rout, /*bw_x*/bw);

  if (party == CLIENT) {
    // interpret as signed two’s-complement fixed-point
    int64_t s = static_cast<int64_t>(rout[0]);
    double fpred = (double)s / (double)(1LL << fracbits);
    cout << "[CLIENT] Secure prediction = " << fpred << endl;
  }

  finalize();
  delete otpack;
  delete iopack;
  return 0;
  finalize();
  return 0;
}
