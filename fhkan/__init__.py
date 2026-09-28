from .hermite import fractional_map, generalized_hermite_functions, generalized_hermite_polynomials, hermite_normalization
from .layers import FHNB, FHKANLayer, ParallelFHKAN, SequentialFHKAN, DeepFHKAN
from .models import MLP, FHKANRegressor
from .pde import burgers_exact, burgers_residual, fde_forcing, lane_emden_exact
__all__ = ['fractional_map', 'generalized_hermite_polynomials', 'generalized_hermite_functions', 'hermite_normalization', 'FHNB', 'FHKANLayer', 'SequentialFHKAN', 'ParallelFHKAN', 'DeepFHKAN', 'MLP', 'FHKANRegressor', 'burgers_exact', 'burgers_residual', 'lane_emden_exact', 'fde_forcing']
__version__ = '0.1.0'
