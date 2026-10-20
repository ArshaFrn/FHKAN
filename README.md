# Fractional Hermite Kolmogorov-Arnold Network (FHKAN)

Fractional Hermite Kolmogorov-Arnold Network (FHKAN) is a novel neural network architecture that combines the Kolmogorov-Arnold Network (KAN) framework with trainable fractional generalized Hermite functions as its basis functions.

FHKAN provides a globally defined, bounded, and differentiable basis on the real line, while introducing trainable fractional orders that allow the network to adapt its approximation behaviour to the underlying data. The proposed architecture is evaluated on both data-driven and physics-informed learning tasks.

## Installation

Clone the repository:

```bash
git clone https://github.com/ArshaFrn/FHKAN.git
cd FHKAN
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Trainable Fractional Hermite Basis

Each fractional Hermite block contains trainable parameters controlling:

* Hermite shape
* Fractional order
* Input scaling

These parameters are optimized jointly with the neural-network weights using gradient-based optimization.

Unlike polynomial-based KAN variants, FHKAN does not require its inputs to be restricted to a bounded interval.

## Contribution

Contributions are welcome.

If you find a bug, have a feature request, or would like to improve the implementation, please open an issue or submit a pull request.

## Citation

If you use FHKAN in your research, please cite our paper:

```bibtex
@article{farnoush2026fhkan,
    title   = {FHKAN: Fractional Hermite Kolmogorov-Arnold Networks
               for Data-Driven and Physics-Informed Learning},
    author  ={Farnoush, Arsha and Sadr, Amirhossein and Pakniyat, Aida and Rahmati, Dara and Gorgin, Saeid and Parand, Kourosh},
    journal = {arXiv preprint arXiv:2610.25148},
    year    = {2026}
}
```
