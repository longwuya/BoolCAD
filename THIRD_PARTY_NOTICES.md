# Third-party notices

This repository contains or links against the following third-party software:

- **ANN (Approximate Nearest Neighbor Library)**, copyright University of
  Maryland, Sunil Arya and David Mount. The headers state that ANN is
  distributed under the Lesser GNU Public License (LGPL). The Windows binary
  runtime is included in `utils/ann/lib/`.
- **PQP**, copyright The University of North Carolina at Chapel Hill. The
  license text is included at the top of `utils/pqp/PQP.h`.
- **L-BFGS**, copyright Jorge Nocedal and Naoaki Okazaki, distributed under
  the MIT license. The license text is included in `utils/lbfgs/lbfgs.h`,
  `utils/lbfgs/lbfgs.cpp`, and the arithmetic helper headers.

CGAL, Eigen3, Boost, and OpenMP are resolved through the build environment and
retain their respective upstream licenses.
