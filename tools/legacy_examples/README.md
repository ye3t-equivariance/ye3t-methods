# Historical source demonstrations

These files preserve earlier APIs for regression work. The migration record
contains the original paths and SHA-256 hashes. Their configurations describe
the historical implementations; use the public examples for new calculations.

| Historical demonstration | Current workflow or coverage |
| --- | --- |
| `role_density_fit.py` | `examples/quickstart/shell_density_fit.py` uses the shell-role compiler and common linear model API. `tests/test_lifted_density_linear_fit.py` preserves the older fit. |
| `phi_fit.py` | `tests/test_compact_phi.py` checks the legacy scalar motif average, E/F/stress fitting and model loading. A general symmetry-adapted Φ orbit-sum linear potential requires further implementation. |
| `phi_depth2.py` | Legacy depth-two motif averaging; the selected explicit-Φ cluster example demonstrates the separate coupled tensor construction. |
| `parent_coefficient.py`, `parent_coupling.py` | `examples/quickstart/coupled_factors.py` shows the public compiled-factor interface. General representation examples belong to `ye3t/examples`. |
| `tagged_carriers_ase.py` | Historical raw carrier inspection. Scalar descriptor and fitting examples use the standardized `Basis` interface; raw carrier and message-passing workflows remain specialist APIs. |

Run historical files only when investigating their specific behavior. Their
small manufactured data and reference evaluators serve regression purposes.
They are not recommended model defaults.
