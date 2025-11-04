# Testing Notes

## 2024-04-27
- Attempted to compile `router.cpp` using `g++ -std=c++17` as a smoke test.
- Compilation failed because the LibTorch development headers (`torch/torch.h`) were not installed in this environment.
- Reworked the router to use an embedded lightweight tensor implementation so it can build without external dependencies while still modelling the secure routing flow.
- Successfully compiled and executed the secure router harness after the refactor (see 2024-04-28 entry).

## 2024-04-28
- `g++ -std=c++17 SCI/tests/bert_bolt/router.cpp -o /tmp/router`
- `echo "0.1 0.2 0.3" > /tmp/query_embedding.txt`
- `echo "0.3 0.2 0.1" > /tmp/llm_embedding.txt`
- `/tmp/router /tmp/query_embedding.txt /tmp/llm_embedding.txt`
- Output: `Selected secure route: llm_backend_2`
