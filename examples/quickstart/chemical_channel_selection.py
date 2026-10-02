"""Select which chemical neighbor channels enter an ACE descriptor basis."""

from ye3t_ace import YE3TDescriptors


config = {
    "metadata": {"name": "chemical_channel_selection"},
    "basis": {
        "elements": ["Li", "Na"], "type_map": {"Li": 0, "Na": 1},
        "cutoff": 4.0, "ranks": (1, 2, 3, 4),
        "nmax": (2, 2, 2, 2), "lmax": (1, 1, 1, 1),
        "lmin": (0, 0, 0, 0), "basis_type": "no_charge",
        "k_o_max": 0, "k_max": (0, 0, 0, 0),
        "site_basis": {"mode": "explicit", "rc": 4.0, "lmbda": 0.25},
    },
    "representation": {"L_R": 0, "M_R_values": (0,)},
    "runtime": {"backend": "pytorch"},
    "model": None,
    "targets": {"selected_neighbor_species": "Li"},
    "validation": {"show_channel_indices": True},
}
ace_settings = {**config["basis"], **config["representation"],
                "backend": config["runtime"]["backend"]}
selected_type = config["basis"]["type_map"][config["targets"]["selected_neighbor_species"]]
all_channels = YE3TDescriptors.ace(ace_settings)
selected_channels = YE3TDescriptors.ace(
    {**ace_settings, "restrict_neighbor_mu": (selected_type,)}
)
print("all_columns", len(all_channels.descriptor_specs))
print("selected_columns", len(selected_channels.descriptor_specs))
print("neighbor_type_indices", sorted({
    channel.mu for spec in selected_channels.descriptor_specs for channel in spec.channels
}))
