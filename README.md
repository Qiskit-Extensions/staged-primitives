<!-- SHIELDS -->
<div align="left">

  ![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-informational)
  [![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-informational)](https://www.python.org/)
  [![Qiskit Terra](https://img.shields.io/badge/Qiskit%20Terra-%E2%89%A5%200.22.2-6133BD)](https://github.com/Qiskit/qiskit-terra)
<br />
  [![Tests](https://github.com/qiskit-community/staged-primitives/actions/workflows/test.yml/badge.svg)](https://github.com/qiskit-community/staged-primitives/actions/workflows/test.yml)
  [![Coverage](https://coveralls.io/repos/github/qiskit-community/staged-primitives/badge.svg?branch=main)](https://coveralls.io/github/qiskit-community/staged-primitives?branch=main)
  [![Release](https://img.shields.io/github/release/qiskit-community/staged-primitives.svg?include_prereleases&label=Release)](https://github.com/qiskit-community/staged-primitives/releases)
  [![License](https://img.shields.io/github/license/qiskit-community/staged-primitives?label=License)](LICENSE.txt)

</div>
<!-- PROJECT LOGO -->
<br />
<p align="center">
  <!-- <a href="README.md">
    <img src="https://github.com/qiskit-community/staged-primitives/blob/main/docs/media/cover.png?raw=true" alt="Logo" width="300">
  </a> -->
  <h2 align="center">Zero Noise Extrapolation (ZNE)</h2>
</p>
<!-- QUICK LINKS -->
<!-- <p align="center">
  <a href="https://mybinder.org/">
    <img src="https://ibm.biz/BdPq3s" alt="Launch Demo" hspace="5" vspace="10">
  </a>
  <a href="https://www.youtube.com/c/qiskit">
    <img src="https://img.shields.io/badge/watch-video-FF0000.svg?style=for-the-badge&logo=youtube" alt="Watch Video" hspace="5" vspace="10">
  </a>
</p> -->


----------------------------------------------------------------------

### Table of contents

1. [About This Project](#about-this-project)
2. [About Prototypes](#about-prototypes)
3. [Deprecation Policy](#deprecation-policy)
4. [Using Quantum Services](#using-quantum-services)
5. [Acknowledgements](#acknowledgements)
6. [References](#references)
7. [License](#license)

#### For users
1. [Installation](https://github.com/qiskit-community/staged-primitives/tree/main/INSTALL.md)
2. [Tutorials](https://github.com/qiskit-community/staged-primitives/tree/main/docs/tutorials/)
3. [Reference Guide](https://github.com/qiskit-community/staged-primitives/tree/main/docs/reference_guide.md)
4. [How-tos](https://github.com/qiskit-community/staged-primitives/tree/main/docs/how_tos/)
5. [Explanations](https://github.com/qiskit-community/staged-primitives/tree/main/docs/explanations/)
6. [How to Give Feedback](https://github.com/qiskit-community/staged-primitives/tree/main/CONTRIBUTING.md#giving-feedback)

#### For developers
1. [Contribution Guidelines](https://github.com/qiskit-community/staged-primitives/tree/main/CONTRIBUTING.md)


----------------------------------------------------------------------

### About This Project


----------------------------------------------------------------------

### About Prototypes

Prototypes is a collaboration between developers and researchers that will give users early access to solutions from cutting-edge research in areas like error mitigation, quantum simulation, and machine learning. These software packages are built on top of, and may eventually be integrated into the Qiskit SDK. They are a contribution as part of the Qiskit community.

Check out our [landing page](https://qiskit-community.github.io/prototypes/) and [blog post](https://medium.com/qiskit/try-out-the-latest-advances-in-quantum-computing-with-ibm-quantum-prototypes-11f51124cb61) for more information!


----------------------------------------------------------------------

### Deprecation Policy

Prototypes are meant to evolve rapidly and, as such, do not follow [Qiskit's deprecation policy](https://qiskit.org/documentation/contributing_to_qiskit.html#deprecation-policy). We may occasionally make breaking changes in order to improve the user experience. When possible, we will keep old interfaces and mark them as deprecated, as long as they can co-exist with the new ones. Each substantial improvement, breaking change, or deprecation will be documented in [`CHANGELOG.md`](https://github.com/qiskit-community/staged-primitives/tree/main/CHANGELOG.md).


----------------------------------------------------------------------

### Using Quantum Services

If you are interested in using quantum services (i.e. using a real quantum computer, not a simulator) you can look at the [Qiskit Partners program](https://qiskit.org/documentation/partners/) for partner organizations that have provider packages available for their offerings.

Importantly, *[Qiskit IBM Runtime](https://qiskit.org/documentation/partners/qiskit_ibm_runtime)* is a quantum computing service and programming model that allows users to optimize workloads and efficiently execute them on quantum systems at scale; extending the existing interface in Qiskit with a set of new *primitive* programs.


----------------------------------------------------------------------

### Acknowledgements


----------------------------------------------------------------------

### References


----------------------------------------------------------------------

### License
[Apache License 2.0](https://github.com/qiskit-community/staged-primitives/tree/main/LICENSE.txt)
