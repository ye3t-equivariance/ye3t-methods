Shell-role linear potentials
============================

``Basis`` can evaluate the same formal-Cauchy shell source used by the
message-passing workflow. Each density factor carries a physical shell
coordinate. For ``q`` shells, the compiler decomposes their ``S_q`` action
before selecting scalar energy features. Internal paired Young types can be
nontrivial. Each fitted coefficient multiplies one independent shell-invariant,
rotation-scalar, even-parity feature.

Use ``basis.tensor_product.kind = "shell_density"`` with a shell-filtered
radial family. The editable publication example is
``ye3t-ace/examples/publication/ni_ye3t_message_passing/shell_linear.py``.
It imports the public packages and can run outside the private checkout.

.. code-block:: python

   from ase.io import read
   from ye3t import YE3TRepresentation
   from ye3t_methods import Basis, LinearModel

   representation = YE3TRepresentation.from_config(config["representation"])
   basis = Basis.from_config(config["basis"], representation=representation,
                             runtime=config["runtime"])
   atoms = read("Ni_validation.extxyz")
   descriptors = basis.create(atoms)  # atoms by independent scalar features
   model = LinearModel(basis).fit(read("Ni_train.extxyz", ":"), config=config)
   path = model.write("shell_linear.ye3t.json")
   atoms.calc = LinearModel.read(path).ase_calculator()
   print(atoms.get_potential_energy(), atoms.get_forces())

The example supplies a complete editable seven-section config. Radial counts,
angular caps, ranks, shell windows, internal Young/rotation choices, device and
target fields are visible. Omitting shell occupations keeps the complete
occupation space. A selected occupation set must contain complete shell
permutation orbits. ``representation.intermediates`` controls paired block
types; ``representation.source_sectors`` selects the final shell-invariant
``L=0``, even sector, with an optional independent-copy budget.

Force and stress design rows differentiate the full frozen radial transform
and coupled features. Ridge regularization acts in the saved compiler
coordinates; comparing different coordinate systems requires transporting the
penalty. The current linear fit weights multiply total-energy and force-component
squared residuals, following the existing linear API. Neural physical per-atom
energy weighting is a separate training convention.

Saved JSON files include the frozen compiler catalogue, coefficient order,
radial source, species offsets and fitting metadata. Reloading verifies the
canonical source and coefficient identity. No training data are needed.
Reload selects CPU inference after checking the saved runtime identity. Use
``model.ase_calculator(device="cuda")`` to select CUDA explicitly, or
``model.ase_calculator(evaluator="torch")`` for the tensor reference.
``runtime.evaluator = "auto"`` uses the production native polynomial kernels;
``torch`` requests the independent tensor implementation for numerical checks.

ASE energies, forces, stresses and dynamics are supported. LAMMPS execution
for this shell-filtered radial source is a later task, conditional on model
quality comparisons. The linear energy workflow uses a scalar output;
complete nontrivial shell multiplets are available in the message-passing
workflow.
