---
description: Run correctness tests with make test-dev
---

Run the correctness tests using `make test-dev` and report the results.

**Note:** Since the package is installed in editable mode, any changes to `.pyx` files will be automatically recompiled when tests are executed.

After running the tests:
1. Check if the code compiled successfully (watch for Cython compilation output)
2. If any tests failed, analyze the error messages and identify the root cause
3. Suggest fixes for any failing tests
