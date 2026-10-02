"""Select a smaller set of exact one-hot neighbor channels."""

from ye3t_ace import YE3TDescriptors


config = {
    "elements": ["Li", "Na"], "type_map": {"Li": 0, "Na": 1},
    "cutoff": 4.0, "ranks": (1,), "nmax": (1,), "lmax": (0,),
    "lmin": (0,), "L_R": 0, "M_R_values": (0,),
    "basis_type": "no_charge", "k_o_max": 0, "k_max": (0,),
    "site_basis": {"mode": "explicit", "rc": 4.0, "lmbda": 0.25},
}
all_channels = YE3TDescriptors.ace(config)
li_neighbors_only = YE3TDescriptors.ace({**config, "restrict_neighbor_mu": (0,)})
print("all_columns", len(all_channels.descriptor_specs))
print("selected_columns", len(li_neighbors_only.descriptor_specs))
print("neighbor_type_indices", sorted({
    channel.mu for spec in li_neighbors_only.descriptor_specs for channel in spec.channels
}))

# TODO: expose a saved fixed embedding with genuinely reduced chemical width.
