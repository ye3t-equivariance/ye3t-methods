"""Application-side composition for compiler-owned YE3T execution plans."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

from ye3t.execution_plan import (
    YE3TExecutionPlan,
    YE3T_SOURCE_ASSEMBLY_SCHEMA,
    YE3TCarrierLayout,
)
from ye3t.runtime import (
    YE3TFactorizedAngularModule,
    YE3THierarchicalRepeatedBlockModule,
    YE3TSourceAnalysisModule,
)
from ye3t_ace.equivariant_calc.labeling import SingleChannelLabel


YE3T_COMPILED_MODEL_SCHEMA = "ye3t_compiled_model_v2"
_YE3T_COMPILED_MODEL_LEGACY_SCHEMA = "ye3t_compiled_model_v1"
_VALIDATED_SHARED_SOURCE_RECORD = object()


def _canonical_payload(value):
    if hasattr(value, "to_dict"):
        return _canonical_payload(value.to_dict())
    if isinstance(value, dict):
        return {
            str(key): _canonical_payload(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_payload(item) for item in value]
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    return value


def _stable_hash(value):
    encoded = json.dumps(
        _canonical_payload(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _complex_values(values):
    result = []
    for value in values:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            result.append(complex(float(value[0]), float(value[1])))
        else:
            result.append(complex(value))
    return tuple(result)


def _complex_value(value):
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return complex(float(value[0]), float(value[1]))
    return complex(value)


def _terminal_instruction(execution_plan, instruction_id=None):
    if instruction_id is None:
        if len(execution_plan.forward_schedule) != 1:
            raise ValueError(
                "compiled-model v1 requires one terminal forward instruction"
            )
        instruction_id = execution_plan.forward_schedule[0]
    return next(
        instruction
        for instruction in execution_plan.instructions
        if instruction.instruction_id == str(instruction_id)
    )


def _instruction_output_dimension(execution_plan, instruction):
    layout = next(
        layout
        for layout in execution_plan.carrier_layouts
        if layout.key == instruction.output_carrier
    )
    return int(layout.width)


def _instruction_source_assembly(execution_plan, instruction):
    return next(
        assembly
        for assembly in execution_plan.source_assemblies
        if assembly.assembly_id == instruction.source_assembly_id
    )


def _atomistic_source_slot_records(execution_plan, instruction, metadata):
    assembly = _instruction_source_assembly(execution_plan, instruction)
    source = assembly.source_realization
    if source.kind != "ordinary_density":
        raise ValueError(
            "the initial compiled atomistic runtime supports ordinary_density; "
            "role-resolved and motif source buffers require their dedicated "
            "materializers"
        )
    instruction_metadata = dict(instruction.metadata)
    input_content = tuple(instruction_metadata.get("input_content", ()))
    input_Ls = tuple(
        int(value)
        for value in instruction_metadata.get("input_Ls", ())
    )
    if not input_Ls and instruction.factorized_angular_plan_id is not None:
        angular_plan = next(
            plan
            for plan in execution_plan.factorized_angular_plans
            if plan.plan_id == instruction.factorized_angular_plan_id
        )
        input_Ls = tuple(int(value) for value in angular_plan.input_Ls)
    if not input_content:
        input_content = tuple(source.content)
    if len(input_content) != int(source.rank):
        raise ValueError("compiled instruction is missing exact input_content")
    if len(input_Ls) != int(source.rank):
        raise ValueError("compiled instruction is missing exact input_Ls")

    records = tuple(dict(value) for value in metadata.get("source_slots", ()))
    if len(records) != int(source.rank):
        raise ValueError(
            "radial_angular_metadata.source_slots must contain one explicit "
            "binding per compiler input slot"
        )
    required = ("content", "mu0", "mu", "kappa0", "kappa", "n", "l")
    normalized = []
    for index, (record, content, angular_L) in enumerate(
        zip(records, input_content, input_Ls)
    ):
        missing = tuple(name for name in required if name not in record)
        if missing:
            raise ValueError(
                f"source slot {index} is missing fields {missing}"
            )
        if record["content"] != content:
            raise ValueError(
                f"source slot {index} content does not match compiler content"
            )
        if int(record["l"]) != int(angular_L):
            raise ValueError(
                f"source slot {index} l does not match compiler input_Ls"
            )
        normalized.append(record)
    return tuple(normalized)


def _slot_channel(record, magnetic):
    return SingleChannelLabel(
        mu0=int(record["mu0"]),
        mu=int(record["mu"]),
        kappa0=int(record["kappa0"]),
        kappa=int(record["kappa"]),
        n=int(record["n"]),
        l=int(record["l"]),
        m=int(magnetic),
        l_aux=record.get("l_aux"),
        m_aux=record.get("m_aux"),
        eta=None,
    )


class YE3TCompiledModelArtifact:
    """Portable compiled-model artifact independent of Python model pickles."""

    def __init__(
        self,
        execution_plan,
        radial_angular_metadata,
        species_mapping,
        normalization,
        readout,
        model_metadata=None,
        certificate=None,
        schema=YE3T_COMPILED_MODEL_SCHEMA,
        artifact_hash="",
        _validated_shared_record_token=None,
    ):
        if str(schema) not in {
            YE3T_COMPILED_MODEL_SCHEMA,
            _YE3T_COMPILED_MODEL_LEGACY_SCHEMA,
        }:
            raise ValueError("unsupported YE3T compiled-model schema")
        if not isinstance(execution_plan, YE3TExecutionPlan):
            execution_plan = YE3TExecutionPlan.from_dict(execution_plan)
        species_mapping = {
            str(symbol): int(index)
            for symbol, index in dict(species_mapping).items()
        }
        if not species_mapping:
            raise ValueError("species_mapping must not be empty")
        indices = tuple(sorted(species_mapping.values()))
        if indices != tuple(range(len(indices))):
            raise ValueError("species_mapping indices must be contiguous from zero")
        readout = dict(readout)
        weights = _complex_values(readout.get("weights", ()))
        if not weights:
            raise ValueError("readout weights must not be empty")
        instruction = _terminal_instruction(execution_plan)
        instruction_id = instruction.instruction_id
        output_dimension = _instruction_output_dimension(
            execution_plan,
            instruction,
        )
        radial_angular_metadata = dict(radial_angular_metadata)
        if any(str(key).startswith("symmetric_block_") for key in radial_angular_metadata):
            raise ValueError("unsupported source metadata")
        if len(weights) != output_dimension:
            raise ValueError(
                "readout weight count must match terminal plan output dimension"
            )
        bias = _complex_value(readout.get("bias", 0.0))
        self.schema = str(schema)
        self.execution_plan = execution_plan
        self.radial_angular_metadata = radial_angular_metadata
        self.species_mapping = species_mapping
        self.normalization = dict(normalization)
        self.readout = {
            "weights": weights,
            "bias": bias,
            "terminal_instruction_id": str(instruction_id),
        }
        self.model_metadata = dict(model_metadata or {})
        self.certificate = dict(certificate or {})
        if _validated_shared_record_token is _VALIDATED_SHARED_SOURCE_RECORD:
            if not str(artifact_hash):
                raise ValueError(
                    "validated shared source record requires an artifact hash"
                )
            self.artifact_hash = str(artifact_hash)
        else:
            if self.schema == _YE3T_COMPILED_MODEL_LEGACY_SCHEMA:
                computed_hash = _stable_hash(self._payload(include_hash=False))
            else:
                computed_hash = _stable_hash(self._hash_payload())
            if artifact_hash and str(artifact_hash) != computed_hash:
                raise ValueError(
                    "compiled-model artifact hash does not match payload"
                )
            self.artifact_hash = computed_hash

    def _application_payload(self):
        return {
            "radial_angular_metadata": dict(self.radial_angular_metadata),
            "species_mapping": dict(self.species_mapping),
            "normalization": dict(self.normalization),
            "readout": {
                "weights": [
                    [float(value.real), float(value.imag)]
                    for value in self.readout["weights"]
                ],
                "bias": [
                    float(self.readout["bias"].real),
                    float(self.readout["bias"].imag),
                ],
                "terminal_instruction_id": str(
                    self.readout["terminal_instruction_id"]
                ),
            },
            "model_metadata": dict(self.model_metadata),
            "certificate": dict(self.certificate),
        }

    def _hash_payload(self):
        return {
            "schema": str(self.schema),
            "execution_plan_hash": str(self.execution_plan.plan_hash),
            **self._application_payload(),
        }

    def _payload(self, include_hash):
        payload = {
            "schema": str(self.schema),
            "execution_plan": self.execution_plan.to_dict(),
            **self._application_payload(),
        }
        if self.schema == YE3T_COMPILED_MODEL_SCHEMA:
            payload["execution_plan_hash"] = str(
                self.execution_plan.plan_hash
            )
        if include_hash:
            payload["artifact_hash"] = str(self.artifact_hash)
        return payload

    def to_dict(self):
        return self._payload(include_hash=True)

    def to_shared_plan_record(self):
        """Serialize application data while referring to one exact plan hash."""

        return {
            "schema": str(self.schema),
            "execution_plan_hash": str(self.execution_plan.plan_hash),
            **self._application_payload(),
            "artifact_hash": str(self.artifact_hash),
        }

    @classmethod
    def from_dict(cls, payload):
        payload = dict(payload)
        stored_plan_hash = str(payload.get("execution_plan_hash", ""))
        if stored_plan_hash:
            plan_payload = payload["execution_plan"]
            plan_hash = str(
                plan_payload.plan_hash
                if isinstance(plan_payload, YE3TExecutionPlan)
                else plan_payload.get("plan_hash", "")
            )
            if plan_hash != stored_plan_hash:
                raise ValueError(
                    "compiled-model execution-plan hash does not match payload"
                )
        readout = dict(payload["readout"])
        return cls(
            execution_plan=payload["execution_plan"],
            radial_angular_metadata=payload["radial_angular_metadata"],
            species_mapping=payload["species_mapping"],
            normalization=payload["normalization"],
            readout={
                "weights": readout["weights"],
                "bias": readout["bias"],
            },
            model_metadata=payload.get("model_metadata", {}),
            certificate=payload.get("certificate", {}),
            schema=payload.get("schema", YE3T_COMPILED_MODEL_SCHEMA),
            artifact_hash=payload.get("artifact_hash", ""),
        )

    @classmethod
    def from_shared_plan_record(cls, payload, execution_plan):
        payload = dict(payload)
        if not isinstance(execution_plan, YE3TExecutionPlan):
            execution_plan = YE3TExecutionPlan.from_dict(execution_plan)
        expected_hash = str(payload.get("execution_plan_hash", ""))
        if expected_hash != str(execution_plan.plan_hash):
            raise ValueError(
                "shared source record references a different execution plan"
            )
        payload["execution_plan"] = execution_plan
        return cls.from_dict(payload)

    @classmethod
    def _from_validated_shared_plan_record(cls, payload, execution_plan):
        """Rebuild a source after its enclosing bundle hash was validated."""

        payload = dict(payload)
        if not isinstance(execution_plan, YE3TExecutionPlan):
            raise TypeError(
                "validated shared source reconstruction requires an exact plan"
            )
        expected_hash = str(payload.get("execution_plan_hash", ""))
        if expected_hash != str(execution_plan.plan_hash):
            raise ValueError(
                "shared source record references a different execution plan"
            )
        readout = dict(payload["readout"])
        artifact = cls(
            execution_plan=execution_plan,
            radial_angular_metadata=payload["radial_angular_metadata"],
            species_mapping=payload["species_mapping"],
            normalization=payload["normalization"],
            readout={
                "weights": readout["weights"],
                "bias": readout["bias"],
            },
            model_metadata=payload.get("model_metadata", {}),
            certificate=payload.get("certificate", {}),
            schema=payload.get("schema", YE3T_COMPILED_MODEL_SCHEMA),
            artifact_hash=payload.get("artifact_hash", ""),
            _validated_shared_record_token=_VALIDATED_SHARED_SOURCE_RECORD,
        )
        return artifact

    def save(self, path):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(
            json.dumps(
                _canonical_payload(self.to_dict()),
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
        return output

    @classmethod
    def load(cls, path):
        return cls.from_dict(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )

    def linear_readout(self, backend="auto", dtype=torch.float64, device=None):
        if any(value.imag != 0.0 for value in self.readout["weights"]):
            if dtype == torch.float64:
                dtype = torch.complex128
            elif dtype == torch.float32:
                dtype = torch.complex64
        return YE3TCompiledLinearReadout(
            self.execution_plan,
            weight=self.readout["weights"],
            bias=self.readout["bias"],
            instruction_id=self.readout["terminal_instruction_id"],
            backend=backend,
            dtype=dtype,
            device=device,
        )

    def atomistic_linear_model(
        self,
        site_basis,
        backend="auto",
        dtype=None,
        device=None,
    ):
        """Bind explicit physical source slots to the compiled linear plan."""

        if dtype is None:
            dtype = site_basis.cfg.complex_dtype
        return _YE3TCompiledAtomisticLinearModel(
            artifact=self,
            site_basis=site_basis,
            backend=backend,
            dtype=dtype,
            device=device,
        )

    def lifted_role_source_evaluator(
        self,
        role_density,
        backend="auto",
        dtype=torch.complex128,
        device=None,
        channel_transform=None,
        _shared_source_analysis=None,
        _build_linear_readout=True,
    ):
        """Bind a physical role-resolved density to the compiled ``L_v``."""

        return YE3TCompiledLiftedRoleSourceEvaluator(
            self,
            role_density,
            backend=backend,
            dtype=dtype,
            device=device,
            channel_transform=channel_transform,
            _shared_source_analysis=_shared_source_analysis,
            _build_linear_readout=_build_linear_readout,
        )

    def ase_calculator(
        self,
        site_basis,
        backend="auto",
        dtype=None,
        device=None,
        reference_energies=None,
        **kwargs,
    ):
        """Build an ASE calculator from this compiled atomistic artifact."""

        cutoff = self.radial_angular_metadata.get("cutoff")
        if cutoff is None or float(cutoff) <= 0.0:
            raise ValueError(
                "compiled artifact radial_angular_metadata requires a "
                "positive cutoff for ASE execution"
            )
        if dtype is None:
            dtype = site_basis.cfg.dtype
        source_backend = str(
            getattr(site_basis.cfg, "source_backend", "")
        )
        if device is None and source_backend == "native_cuda":
            device = "cuda"
        elif device is None and (
            str(backend) in {"native", "native_cpu"}
            or source_backend in {"native", "native_cpu"}
        ):
            device = "cpu"
        model = self.atomistic_linear_model(
            site_basis,
            backend=backend,
            dtype=site_basis.cfg.complex_dtype,
            device=device,
        )
        return YE3TCompiledArtifactCalculator(
            model,
            cutoff=float(cutoff),
            type_map=self.species_mapping,
            dtype=dtype,
            device=device,
            reference_energies=reference_energies,
            **kwargs,
        )


class YE3TCompiledSourceEvaluator(torch.nn.Module):
    """Evaluate application-materialized sources with a compiled YE3T plan."""

    def __init__(
        self,
        execution_plan,
        instruction_id=None,
        backend="auto",
        dtype=None,
        device=None,
        channel_mixing=False,
        strict=False,
        _shared_source_analysis=None,
    ):
        super().__init__()
        if not isinstance(execution_plan, YE3TExecutionPlan):
            execution_plan = YE3TExecutionPlan.from_dict(execution_plan)
        self.execution_plan = execution_plan
        self.instruction = _terminal_instruction(
            execution_plan,
            instruction_id=instruction_id,
        )
        self.output_layout = next(
            layout
            for layout in execution_plan.carrier_layouts
            if layout.key == self.instruction.output_carrier
        )
        self.factorized = (
            self.instruction.factorized_angular_plan_id is not None
        )
        self.hierarchical = bool(
            self.instruction.opcode == "block_symmetric_power"
            and dict(self.instruction.metadata).get("schema")
            == "ye3t_hierarchical_repeated_angular_blocks_v1"
        )
        self.backend = str(backend)
        if (self.factorized or self.hierarchical) and bool(channel_mixing):
            raise ValueError(
                "channel_mixing currently requires a dense source-analysis plan"
            )
        if _shared_source_analysis is not None:
            if self.hierarchical:
                expected_type = YE3THierarchicalRepeatedBlockModule
            elif self.factorized:
                expected_type = YE3TFactorizedAngularModule
            else:
                expected_type = YE3TSourceAnalysisModule
            if not isinstance(_shared_source_analysis, expected_type):
                raise ValueError(
                    "shared source analysis has the wrong runtime type"
                )
            shared_plan_hash = str(_shared_source_analysis.plan_hash)
            if shared_plan_hash != str(execution_plan.plan_hash):
                raise ValueError(
                    "shared source analysis references a different exact plan"
                )
            self.source_analysis = _shared_source_analysis
        elif self.hierarchical:
            self.source_analysis = YE3THierarchicalRepeatedBlockModule(
                execution_plan,
                instruction_id=self.instruction.instruction_id,
                backend=backend,
                dtype=dtype,
                device=device,
            )
        elif self.factorized:
            self.source_analysis = YE3TFactorizedAngularModule(
                execution_plan,
                factorized_angular_plan_id=(
                    self.instruction.factorized_angular_plan_id
                ),
                backend=backend,
                dtype=dtype,
                device=device,
            )
        else:
            self.source_analysis = YE3TSourceAnalysisModule(
                execution_plan,
                instruction_id=self.instruction.instruction_id,
                backend=backend,
                dtype=dtype,
                device=device,
                channel_mixing=channel_mixing,
                strict=strict,
            )

    @property
    def output_dimension(self):
        return _instruction_output_dimension(
            self.execution_plan,
            self.instruction,
        )

    def forward(self, source):
        if self.hierarchical:
            block_inputs = (
                tuple(source)
                if isinstance(source, (tuple, list))
                else (source,)
            )
            return self.source_analysis(*block_inputs)
        if self.factorized:
            return self.source_analysis(source)
        return self.source_analysis(source)

    def forward_logical(self, source):
        """Return separate channel/multiplicity, tableau, and magnetic axes."""

        value = self(source)
        if self.hierarchical:
            return value
        channel_count = int(self.output_layout.channel_count)
        tableau_count = int(self.output_layout.tableau_count)
        magnetic_count = int(self.output_layout.magnetic_count)
        return value.reshape(
            tuple(value.shape[:-2] if self.factorized else value.shape[:-1])
            + (channel_count, tableau_count, magnetic_count)
        )

    def runtime_report(self):
        if self.factorized:
            report = dict(self.source_analysis.runtime_report())
            report["opcode"] = str(self.instruction.opcode)
        else:
            report = dict(self.source_analysis.runtime_report())
        report.update(
            {
                "application_owner": "ye3t-ace",
                "source_materialization_owner": "ye3t-ace",
                "coupling_plan_owner": "ye3t",
                "logical_output_axes": (
                    "channel_or_multiplicity",
                    "tableau_t",
                    "magnetic_M",
                ),
            }
        )
        return report


class YE3TCompiledLinearReadout(torch.nn.Module):
    """Fixed linear readout over one compiler-owned source-analysis plan."""

    def __init__(
        self,
        execution_plan,
        weight,
        bias=0.0,
        instruction_id=None,
        backend="auto",
        dtype=torch.float64,
        device=None,
    ):
        super().__init__()
        self.source_evaluator = YE3TCompiledSourceEvaluator(
            execution_plan,
            instruction_id=instruction_id,
            backend=backend,
            dtype=dtype,
            device=device,
        )
        if dtype in (torch.float32, torch.float64):
            complex_weights = _complex_values(weight)
            if any(value.imag != 0.0 for value in complex_weights):
                raise ValueError("complex readout weights require a complex dtype")
            weight = tuple(float(value.real) for value in complex_weights)
            complex_bias = _complex_value(bias)
            if complex_bias.imag != 0.0:
                raise ValueError("complex readout bias requires a complex dtype")
            bias = float(complex_bias.real)
        weight = torch.as_tensor(weight, dtype=dtype, device=device)
        if weight.ndim != 1:
            raise ValueError("weight must be one-dimensional")
        if int(weight.shape[0]) != self.source_evaluator.output_dimension:
            raise ValueError("weight length must match coupled output dimension")
        self.weight = torch.nn.Parameter(weight.clone())
        self.bias = torch.nn.Parameter(
            torch.as_tensor(bias, dtype=dtype, device=device).reshape(())
        )
        self.last_linear_readout_backend = None

    def forward(self, source):
        output = self.source_evaluator.source_analysis.linear_readout(
            source,
            self.weight,
            self.bias,
        )
        self.last_linear_readout_backend = self.source_evaluator.source_analysis.last_linear_readout_backend
        return output

    def runtime_report(self):
        report = dict(self.source_evaluator.runtime_report())
        linear_backend = (
            self.last_linear_readout_backend
        )
        fused = linear_backend in {
            "native_fused_factorized_linear",
            "native_fused_source_analysis_linear",
        }
        native_materialized = (
            linear_backend
            == "native_materialized_measured_complex_role"
        )
        report.update(
            {
                "model": "YE3TCompiledLinearReadout",
                "linear_readout": True,
                "linear_readout_backend": linear_backend,
                "readout_fused_into_native_kernel": fused,
                "native_readout_feature_buffer_materialized": (
                    False if fused else native_materialized
                ),
                "force_path": (
                    "native_fused_readout_coupler_adjoint_then_source_chain"
                    if fused
                    else (
                        "native_materialized_readout_coupler_adjoint_then_source_chain"
                        if native_materialized
                        else "coupler_adjoint_then_source_chain"
                    )
                ),
            }
        )
        return report


class YE3TCompiledLiftedRoleSourceEvaluator(torch.nn.Module):
    """Bind physical ``A_s`` role tuples to exact induced source coordinates."""

    def __init__(
        self,
        artifact,
        role_density,
        backend="auto",
        dtype=torch.complex128,
        device=None,
        channel_transform=None,
        _shared_source_analysis=None,
        _build_linear_readout=True,
    ):
        super().__init__()
        self.artifact = artifact
        self.role_density = role_density
        object.__setattr__(self, "channel_transform", channel_transform)
        self.source_evaluator = YE3TCompiledSourceEvaluator(
            artifact.execution_plan,
            instruction_id=artifact.readout["terminal_instruction_id"],
            backend=backend,
            dtype=dtype,
            device=device,
            _shared_source_analysis=_shared_source_analysis,
        )
        instruction = self.source_evaluator.instruction
        assembly = _instruction_source_assembly(
            artifact.execution_plan,
            instruction,
        )
        source = assembly.source_realization
        if source.kind != "lifted_density_roles":
            raise ValueError(
                "lifted role source evaluation requires "
                "source_realization.kind='lifted_density_roles'"
            )
        if str(
            assembly.provenance.get("source_assembly_schema", "")
        ) != YE3T_SOURCE_ASSEMBLY_SCHEMA:
            raise ValueError(
                "compiled lifted role source uses an obsolete source "
                "assembly convention; rebuild the artifact with "
                + YE3T_SOURCE_ASSEMBLY_SCHEMA
            )
        coordinate_records = tuple(
            dict(record)
            for record in assembly.provenance.get(
                "source_coordinate_records",
                (),
            )
        )
        instruction_metadata = dict(instruction.metadata)
        input_content = tuple(
            instruction_metadata.get("input_content", source.content)
        )
        input_Ls = tuple(
            int(value)
            for value in instruction_metadata.get("input_Ls", ())
        )
        hierarchical_blocks = tuple()
        if self.source_evaluator.hierarchical:
            hierarchical_blocks = tuple(
                dict(block)
                for block in instruction_metadata.get("blocks", ())
            )
            resolved_input_Ls = [None] * int(source.rank)
            for block in hierarchical_blocks:
                for slot_index in block["slot_indices"]:
                    slot_index = int(slot_index)
                    if resolved_input_Ls[slot_index] is not None:
                        raise ValueError(
                            "hierarchical lifted source assigns one slot twice"
                        )
                    resolved_input_Ls[slot_index] = int(block["input_L"])
            if any(value is None for value in resolved_input_Ls):
                raise ValueError(
                    "hierarchical lifted source does not cover every slot"
                )
            input_Ls = tuple(int(value) for value in resolved_input_Ls)
        if not input_Ls and self.source_evaluator.factorized:
            angular_plan = next(
                plan
                for plan in artifact.execution_plan.factorized_angular_plans
                if plan.plan_id == instruction.factorized_angular_plan_id
            )
            input_Ls = tuple(int(value) for value in angular_plan.input_Ls)
        angular_input_dimension = math.prod(
            2 * int(angular_L) + 1 for angular_L in input_Ls
        )
        if angular_input_dimension <= 0:
            raise ValueError(
                "compiled lifted role source is missing exact input_Ls"
            )
        if self.source_evaluator.factorized or self.source_evaluator.hierarchical:
            source_coordinate_count = int(assembly.source_dimension)
        else:
            if int(assembly.source_dimension) % angular_input_dimension != 0:
                raise ValueError(
                    "lifted role source dimension is incompatible with its "
                    "ordered magnetic-product dimension"
                )
            source_coordinate_count = (
                int(assembly.source_dimension) // angular_input_dimension
            )
        if len(coordinate_records) != source_coordinate_count:
            raise ValueError(
                "lifted role L_v is missing exact source-coordinate records"
            )
        coordinate_records = tuple(
            sorted(
                coordinate_records,
                key=lambda record: int(record["source_index"]),
            )
        )
        if tuple(
            int(record["source_index"])
            for record in coordinate_records
        ) != tuple(range(int(source_coordinate_count))):
            raise ValueError(
                "lifted role source coordinates must be contiguous"
            )
        subgroup_partitions = tuple(
            tuple(int(value) for value in partition)
            for partition in assembly.provenance.get(
                "subgroup_partitions",
                (),
            )
        )
        if any(
            partition != (sum(partition),)
            for partition in subgroup_partitions
        ):
            raise ValueError(
                "physical A_s leaf binding currently requires fully "
                "symmetric child subgroup partitions; nontrivial child "
                "Young projection is not implemented"
            )
        subgroup_sizes = tuple(
            sorted(
                (sum(partition) for partition in subgroup_partitions),
                reverse=True,
            )
        )
        if not self.source_evaluator.hierarchical:
            pair_block_roles = {}
            for content, angular_L, role_label in zip(
                input_content,
                input_Ls,
                source.role_labels,
            ):
                pair_block_roles.setdefault(
                    (content, int(angular_L)), set()
                ).add(str(role_label))
            pair_block_sizes = tuple(
                sorted(
                    (
                        sum(
                            1
                            for content_value, angular_value in zip(
                                input_content, input_Ls
                            )
                            if (content_value, int(angular_value)) == pair_key
                        )
                        for pair_key in pair_block_roles
                    ),
                    reverse=True,
                )
            )
            if pair_block_sizes != subgroup_sizes:
                raise ValueError(
                    "physical A_s pair blocks do not match the compiled child "
                    "subgroup partitions"
                )
            if any(
                len(role_labels) != 1
                for role_labels in pair_block_roles.values()
            ):
                raise ValueError(
                    "physical A_s symmetric child blocks must be role "
                    "homogeneous until the internal child Young projector is "
                    "implemented"
                )
        if any(
            any(int(index) != 0 for index in record["child_tableau_indices"])
            for record in coordinate_records
        ):
            raise ValueError(
                "physical A_s leaf binding does not accept pre-coupled child "
                "tableau coordinates"
            )
        role_mapping = {
            str(key): int(value)
            for key, value in dict(
                artifact.radial_angular_metadata.get(
                    "role_label_to_index",
                    {},
                )
            ).items()
        }
        if not role_mapping:
            raise ValueError(
                "radial_angular_metadata.role_label_to_index is required"
            )
        role_count = int(
            role_density.model.config.lifted_density.num_filters
        )
        role_indices = []
        for record in coordinate_records:
            representative = tuple(
                int(index)
                for index in record["coset_representative"]
            )
            if tuple(sorted(representative)) != tuple(
                range(int(source.rank))
            ):
                raise ValueError(
                    "source coordinate coset representative is invalid"
                )
            labels = tuple(str(label) for label in record["role_tuple"])
            if len(labels) != int(source.rank):
                raise ValueError(
                    "source coordinate role tuple must bind every source slot"
                )
            try:
                indices = tuple(
                    int(role_mapping[str(label)])
                    for label in labels
                )
            except KeyError as exc:
                raise ValueError(
                    "role_label_to_index does not cover source role labels"
                ) from exc
            if any(index < 0 or index >= role_count for index in indices):
                raise ValueError(
                    "role_label_to_index contains an out-of-range role index"
                )
            role_indices.append(indices)
        self.register_buffer(
            "role_indices_by_source_slot",
            torch.tensor(
                role_indices,
                dtype=torch.long,
                device=device,
            ),
        )

        records = tuple(
            dict(record)
            for record in artifact.radial_angular_metadata.get(
                "source_slots",
                (),
            )
        )
        if (
            len(records) != int(source.rank)
            or len(input_content) != int(source.rank)
            or len(input_Ls) != int(source.rank)
        ):
            raise ValueError(
                "lifted role source_slots must bind every compiler leaf"
            )
        lifted_channels = tuple(
            role_density.model.config.lifted_density.channels
        )
        channel_lookup = {
            (
                int(channel.n),
                int(channel.l),
                int(channel.m),
                channel.neighbor_type,
            ): int(index)
            for index, channel in enumerate(lifted_channels)
        }
        transformed_lookup = {}
        if channel_transform is not None:
            transformed_lookup = {
                (
                    int(channel["output_channel"]),
                    int(channel["n"]),
                    int(channel["l"]),
                    int(channel["m"]),
                ): int(index)
                for index, channel in enumerate(
                    channel_transform.virtual_channels
                )
            }
        for slot_index, (record, content, angular_L) in enumerate(
            zip(records, input_content, input_Ls)
        ):
            if record.get("content") != content:
                raise ValueError(
                    f"lifted role source slot {slot_index} content mismatch"
                )
            if int(record.get("l", -1)) != int(angular_L):
                raise ValueError(
                    f"lifted role source slot {slot_index} angular mismatch"
                )
            indices = []
            for magnetic in range(-int(angular_L), int(angular_L) + 1):
                transform_binding = dict(
                    record.get("channel_transform", {})
                )
                if transform_binding:
                    if channel_transform is None:
                        raise ValueError(
                            "transformed source slot requires a channel-transform runtime"
                        )
                    key = (
                        int(transform_binding["output_channel"]),
                        int(record["n"]),
                        int(angular_L),
                        int(magnetic),
                    )
                    lookup = transformed_lookup
                else:
                    key = (
                        int(record["n"]),
                        int(angular_L),
                        int(magnetic),
                        record.get("neighbor_type"),
                    )
                    lookup = channel_lookup
                if key not in lookup:
                    raise ValueError(
                        f"lifted role source slot {slot_index} is missing "
                        f"channel {key}"
                    )
                indices.append(lookup[key])
            self.register_buffer(
                f"slot_channel_indices_{slot_index}",
                torch.tensor(
                    indices,
                    dtype=torch.long,
                    device=device,
                ),
            )
        if self.source_evaluator.hierarchical:
            def physical_binding(record):
                transform = dict(record.get("channel_transform", {}))
                if transform:
                    return (
                        "channel_transform",
                        str(transform["transform_id"]),
                        str(transform["scope"]),
                        int(transform["output_channel"]),
                    )
                return ("neighbor_type", record.get("neighbor_type"))

            for block in hierarchical_blocks:
                indices = tuple(int(value) for value in block["slot_indices"])
                first = indices[0]
                first_record = records[first]
                first_signature = (
                    input_content[first],
                    int(first_record["n"]),
                    int(first_record["l"]),
                    physical_binding(first_record),
                    str(source.role_labels[first]),
                )
                for slot_index in indices[1:]:
                    record = records[slot_index]
                    signature = (
                        input_content[slot_index],
                        int(record["n"]),
                        int(record["l"]),
                        physical_binding(record),
                        str(source.role_labels[slot_index]),
                    )
                    if signature != first_signature:
                        raise ValueError(
                            "hierarchical repeated block requires identical "
                            "content, radial, angular, chemical, and role bindings"
                        )
            object.__setattr__(
                self,
                "hierarchical_blocks",
                hierarchical_blocks,
            )
        self.slot_count = int(source.rank)
        self.source_dimension = int(source_coordinate_count)
        self.prepared_source_dimension = int(assembly.source_dimension)
        self.angular_input_dimension = int(angular_input_dimension)
        self.role_tuple_convention = str(
            assembly.provenance.get("role_tuple_convention", "")
        )
        expected_role_tuple_convention = (
            "source_role_tuple[f] = "
            "role_labels[inverse_coset_representative[f]]"
        )
        if self.role_tuple_convention != expected_role_tuple_convention:
            raise ValueError(
                "lifted role source assembly uses an obsolete or unknown "
                "coset action; rebuild the compiled artifact"
            )
        output_key = self.source_evaluator.output_layout.key
        self.scalar_invariant_output = bool(
            int(output_key.rotation_L) == 0
            and output_key.is_totally_symmetric
        )
        self.linear_readout = None
        if self.scalar_invariant_output and bool(_build_linear_readout):
            self.linear_readout = artifact.linear_readout(
                backend=backend,
                dtype=dtype,
                device=device,
            )

    def _prepared_slots(
        self, density, atom_types=None, density_is_transformed=False
    ):
        if self.channel_transform is not None and not bool(
            density_is_transformed
        ):
            density = self.channel_transform(density, atom_types=atom_types)
        if density.ndim != 3:
            raise ValueError(
                "lifted role density must have shape "
                "[atoms, roles, channels]"
            )
        if int(density.shape[1]) != int(
            self.role_density.model.config.lifted_density.num_filters
        ):
            raise ValueError("lifted role density role count mismatch")
        from ye3t.core.tesseral import (
            real_tesseral_to_complex_multiplet,
        )
        from ye3t_ace.lifted_density import (
            _site_real_block_to_ye3t_tesseral,
        )

        slots = []
        batch = int(density.shape[0])
        for slot_index in range(self.slot_count):
            channel_indices = getattr(
                self,
                f"slot_channel_indices_{slot_index}",
            )
            block = density.index_select(2, channel_indices)
            role_indices = self.role_indices_by_source_slot[
                :,
                int(slot_index),
            ]
            gather = role_indices.reshape(1, -1, 1).expand(
                batch,
                self.source_dimension,
                int(block.shape[2]),
            )
            selected = torch.gather(block, 1, gather)
            angular_L = (int(block.shape[2]) - 1) // 2
            ye3t_real = _site_real_block_to_ye3t_tesseral(
                selected,
                angular_L,
            )
            slots.append(
                real_tesseral_to_complex_multiplet(
                    ye3t_real,
                    angular_L,
                )
            )
        return tuple(slots)

    def _evaluation_source(self, slots):
        slots = tuple(slots)
        if len(slots) != int(self.slot_count):
            raise ValueError(
                "prepared lifted-role slots must bind every compiler leaf"
            )
        if self.source_evaluator.hierarchical:
            return tuple(
                slots[int(block["slot_indices"][0])][:, 0, :]
                for block in self.hierarchical_blocks
            )
        if self.source_evaluator.factorized:
            return slots
        source = slots[0]
        for slot in slots[1:]:
            source = torch.einsum(
                "...a,...b->...ab",
                source,
                slot,
            ).reshape(tuple(source.shape[:-1]) + (-1,))
        source = source.reshape(int(source.shape[0]), -1)
        if int(source.shape[-1]) != int(self.prepared_source_dimension):
            raise ValueError(
                "prepared lifted-role source does not match the compiled "
                "dense source-analysis dimension"
            )
        return source



    def forward(self, density, atom_types=None):
        return self.source_evaluator.forward_logical(
            self._evaluation_source(
                self._prepared_slots(density, atom_types=atom_types)
            )
        )

    def _forward_prepared_slots(self, slots):
        return self.source_evaluator.forward_logical(
            self._evaluation_source(slots)
        )

    def per_atom_energy_from_density(self, density, atom_types=None):
        if self.linear_readout is None:
            raise ValueError(
                "linear energy readout requires parent lambda=(N) and L=0"
            )
        source = self._evaluation_source(
            self._prepared_slots(density, atom_types=atom_types)
        )
        return self.linear_readout(source).real

    def energy_forces(
        self,
        positions,
        atom_types=None,
        *,
        cell=None,
        pbc=None,
        edge_index=None,
        shifts=None,
    ):
        """Return a scalar role-density energy and analytic atomic forces."""

        materialized = self.role_density.materialize(
            positions,
            atom_types,
            cell=cell,
            pbc=pbc,
            edge_index=edge_index,
            shifts=shifts,
            include_derivatives=True,
        )
        differentiable_density = (
            materialized["density"].detach().requires_grad_(True)
        )
        per_atom = self.per_atom_energy_from_density(
            differentiable_density,
            atom_types=atom_types,
        )
        energy = per_atom.sum()
        density_adjoint = torch.autograd.grad(
            energy,
            differentiable_density,
            create_graph=False,
        )[0]
        position_gradient = materialized[
            "density_cache"
        ].position_vjp(density_adjoint)
        return {
            "energy": energy,
            "per_atom_energy": per_atom,
            "forces": -position_gradient,
            "density": materialized["density"],
            "source_runtime": materialized["source_runtime"],
        }

    def materialize_and_evaluate(
        self,
        positions,
        atom_types=None,
        *,
        cell=None,
        pbc=None,
        edge_index=None,
        shifts=None,
        include_derivatives=False,
    ):
        materialized = self.role_density.materialize(
            positions,
            atom_types,
            cell=cell,
            pbc=pbc,
            edge_index=edge_index,
            shifts=shifts,
            include_derivatives=include_derivatives,
        )
        output = dict(materialized)
        output["features"] = self(
            materialized["density"], atom_types=atom_types
        )
        output["execution_runtime"] = self.runtime_report()
        return output

    def runtime_report(self):
        report = dict(self.source_evaluator.runtime_report())
        report.update(
            {
                "physical_source_realization": "lifted_density_roles",
                "physical_source_owner": "ye3t-ace",
                "source_coordinate_owner": "ye3t",
                "role_tuple_convention": self.role_tuple_convention,
                "typed_role_relabeling_is_formal_slot_action": False,
                "role_automorphism_policy": "not_declared",
                "source_coordinate_count": int(self.source_dimension),
                "prepared_source_dimension": int(
                    self.prepared_source_dimension
                ),
                "ordered_magnetic_product_dimension": int(
                    self.angular_input_dimension
                ),
                "factorized_angular_execution": bool(
                    self.source_evaluator.factorized
                ),
                "role_axis_retained_until_coupling": True,
                "site_real_to_primary_complex_transform": True,
                "scalar_invariant_output": bool(
                    self.scalar_invariant_output
                ),
                "analytic_role_density_force_path": bool(
                    self.scalar_invariant_output
                ),
                "channel_transform": (
                    None
                    if self.channel_transform is None
                    else self.channel_transform.report()
                ),
            }
        )
        return report


class _YE3TCompiledAtomisticLinearModel(torch.nn.Module):
    """Ordinary-density atomistic binding for a compiled scalar readout."""

    def __init__(
        self,
        artifact,
        site_basis,
        backend,
        dtype,
        device,
    ):
        super().__init__()
        self.artifact = artifact
        self.site_basis = site_basis
        instruction = _terminal_instruction(artifact.execution_plan)
        output_layout = next(
            layout
            for layout in artifact.execution_plan.carrier_layouts
            if layout.key == instruction.output_carrier
        )
        if int(output_layout.key.rotation_L) != 0:
            raise ValueError(
                "compiled atomistic energy readout requires parent L=0"
            )
        if str(site_basis.cfg.spherical_backend) != "complex":
            raise ValueError(
                "compiled plans use the primary complex convention; real "
                "SiteBasis execution requires a serialized unitary transform"
            )
        possible_types = tuple(
            sorted(int(value) for value in site_basis.cfg.possible_types)
        )
        mapped_types = tuple(sorted(artifact.species_mapping.values()))
        if possible_types != mapped_types:
            raise ValueError(
                "SiteBasis possible_types must match artifact species_mapping"
            )
        declared_normalization = artifact.normalization.get(
            "atomic_base",
            artifact.normalization.get("density"),
        )
        actual_normalization = str(
            site_basis.cfg.atomic_base_normalization
        )
        if declared_normalization == "raw":
            declared_normalization = "none"
        if (
            declared_normalization is not None
            and str(declared_normalization) != actual_normalization
        ):
            raise ValueError(
                "SiteBasis atomic-base normalization does not match artifact"
            )
        slot_records = _atomistic_source_slot_records(
            artifact.execution_plan,
            instruction,
            artifact.radial_angular_metadata,
        )
        channels = []
        channel_indices = {}
        slot_indices = []
        for record in slot_records:
            indices = []
            angular_L = int(record["l"])
            for magnetic in range(-angular_L, angular_L + 1):
                channel = _slot_channel(record, magnetic)
                if channel not in channel_indices:
                    channel_indices[channel] = len(channels)
                    channels.append(channel)
                indices.append(channel_indices[channel])
            slot_indices.append(tuple(indices))
        self.channels = tuple(channels)
        self.slot_count = len(slot_indices)
        for index, indices in enumerate(slot_indices):
            self.register_buffer(
                f"slot_channel_indices_{index}",
                torch.tensor(indices, dtype=torch.int64, device=device),
            )
        self.readout = artifact.linear_readout(
            backend=backend,
            dtype=dtype,
            device=device,
        )

    def _slot_values(self, atomic_base):
        return tuple(
            atomic_base.index_select(
                1,
                getattr(self, f"slot_channel_indices_{index}"),
            )
            for index in range(self.slot_count)
        )

    def _prepared_source(self, atomic_base):
        slots = self._slot_values(atomic_base)
        if self.readout.source_evaluator.factorized:
            return slots
        source = slots[0]
        for slot in slots[1:]:
            source = torch.einsum(
                "...a,...b->...ab",
                source,
                slot,
            ).reshape(tuple(source.shape[:-1]) + (-1,))
        return source

    def per_atom_energy_from_atomic_base(self, atomic_base):
        return self.readout(self._prepared_source(atomic_base)).real

    def fixed_basis_from_atomic_base(self, atomic_base):
        """Return per-atom compiler-plan basis values before linear readout."""

        values = self.readout.source_evaluator(
            self._prepared_source(atomic_base)
        )
        return values.reshape(int(atomic_base.shape[0]), -1).real

    def per_atom_energy(
        self,
        x_ij,
        edge_index,
        atom_types,
        charges=None,
        aux_tensor_basis=None,
    ):
        _, _, atomic_base = self.site_basis.compute_channels_raw_and_final(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            channels=self.channels,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
        )
        return self.per_atom_energy_from_atomic_base(atomic_base)

    def forward(
        self,
        x_ij,
        edge_index,
        atom_types,
        charges=None,
        aux_tensor_basis=None,
    ):
        return self.per_atom_energy(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
        ).sum()

    def energy_forces_virial(
        self,
        x_ij,
        edge_index,
        atom_types,
        charges=None,
        aux_tensor_basis=None,
        create_graph=False,
    ):
        """Return energy, forces, and virial through the compact source VJP."""

        _, raw_atomic_base, final_atomic_base = (
            self.site_basis.compute_channels_raw_and_final(
                x_ij=x_ij,
                edge_index=edge_index,
                atom_types=atom_types,
                channels=self.channels,
                charges=charges,
                aux_tensor_basis=aux_tensor_basis,
            )
        )
        differentiable_base = final_atomic_base.detach().requires_grad_(True)
        per_atom = self.per_atom_energy_from_atomic_base(differentiable_base)
        energy = per_atom.sum()
        channel_adjoint = torch.autograd.grad(
            energy,
            differentiable_base,
            create_graph=bool(create_graph),
            retain_graph=True,
        )[0]
        # SiteBasis VJPs use algebraic adjoints; PyTorch complex grads are conjugate.
        if torch.is_complex(channel_adjoint):
            channel_adjoint = channel_adjoint.conj()
        position_gradient, strain_derivative = (
            self.site_basis.position_vjp_from_raw_channel_adjoint_streaming(
                x_ij=x_ij,
                edge_index=edge_index,
                atom_types=atom_types,
                channels=self.channels,
                raw_atomic_base=raw_atomic_base,
                final_channel_adjoint=channel_adjoint,
                charges=charges,
                aux_tensor_basis=aux_tensor_basis,
                return_strain_derivative=True,
            )
        )
        return {
            "energy": energy,
            "per_atom_energy": per_atom,
            "forces": -position_gradient,
            "strain_derivative": strain_derivative,
            "virial": -strain_derivative,
        }

    def fixed_basis_design(
        self,
        x_ij,
        edge_index,
        atom_types,
        charges=None,
        aux_tensor_basis=None,
        feature_chunk_size=64,
        volume=None,
    ):
        """Build one structure's energy, force, and optional stress rows.

        The returned columns are the compiler-owned invariant basis before
        readout weights. Derivatives are lowered through source-analysis
        adjoints and the compact SiteBasis VJP in bounded feature chunks.
        Only this structure's design rows are retained.
        """

        _, final_atomic_base, record = (
            self.site_basis.compute_channels_with_vjp_record(
                x_ij=x_ij,
                edge_index=edge_index,
                atom_types=atom_types,
                channels=self.channels,
                charges=charges,
                aux_tensor_basis=aux_tensor_basis,
            )
        )
        differentiable_base = (
            final_atomic_base.detach().requires_grad_(True)
        )
        total_basis = self.fixed_basis_from_atomic_base(
            differentiable_base
        ).sum(dim=0)
        feature_count = int(total_basis.numel())
        chunk_size = max(1, int(feature_chunk_size))
        position_blocks = []
        strain_blocks = []
        for start in range(0, feature_count, chunk_size):
            stop = min(feature_count, start + chunk_size)
            width = stop - start
            grad_outputs = torch.zeros(
                (width, feature_count),
                dtype=total_basis.dtype,
                device=total_basis.device,
            )
            indices = torch.arange(
                width,
                dtype=torch.long,
                device=total_basis.device,
            )
            grad_outputs[indices, start + indices] = 1.0
            channel_adjoint = torch.autograd.grad(
                total_basis,
                differentiable_base,
                grad_outputs=grad_outputs,
                is_grads_batched=True,
                retain_graph=stop < feature_count,
                create_graph=False,
            )[0]
            # SiteBasis VJPs consume algebraic rather than PyTorch adjoints.
            if torch.is_complex(channel_adjoint):
                channel_adjoint = channel_adjoint.conj()
            position_blocks.append(
                self.site_basis.position_vjp_from_record_batched(
                    record,
                    channel_adjoint,
                )
            )
            strain_blocks.append(
                self.site_basis.strain_vjp_from_record_batched(
                    record,
                    channel_adjoint,
                )
            )
        position_gradient = torch.cat(position_blocks, dim=0)
        strain_derivative = torch.cat(strain_blocks, dim=0)
        force_design = -position_gradient.permute(1, 2, 0).contiguous()
        strain_design = strain_derivative.permute(1, 2, 0).contiguous()
        stress_design = None
        if volume is not None:
            volume = float(volume)
            if volume <= 0.0:
                raise ValueError(
                    "stress design rows require a positive cell volume"
                )
            symmetric = 0.5 * (
                strain_design + strain_design.transpose(0, 1)
            ) / volume
            stress_design = torch.stack(
                (
                    symmetric[0, 0],
                    symmetric[1, 1],
                    symmetric[2, 2],
                    symmetric[1, 2],
                    symmetric[0, 2],
                    symmetric[0, 1],
                ),
                dim=0,
            )
        return {
            "energy": total_basis.detach(),
            "forces": force_design,
            "strain_derivative": strain_design,
            "stress": stress_design,
            "report": {
                "backend": "compiled_source_analysis_compact_site_basis_vjp",
                "feature_count": feature_count,
                "feature_chunk_size": chunk_size,
                "materializes_dataset_design_matrix": False,
                "materializes_structure_force_rows": True,
                "materializes_edge_derivative_buffer": True,
                "coupling_plan_owner": "ye3t",
                "source_materialization_owner": "ye3t-ace",
            },
        }

    def accumulate_fixed_basis_normal_equations(
        self,
        x_ij,
        edge_index,
        atom_types,
        energy_reference=None,
        force_reference=None,
        stress_reference=None,
        energy_weight=1.0,
        force_weight=1.0,
        stress_weight=1.0,
        include_bias_column=True,
        feature_chunk_size=64,
        volume=None,
        accumulator=None,
        charges=None,
        aux_tensor_basis=None,
    ):
        """Accumulate one structure into fixed-basis normal equations."""

        if stress_reference is not None and volume is None:
            raise ValueError(
                "stress_reference requires the structure cell volume"
            )
        design = self.fixed_basis_design(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
            feature_chunk_size=feature_chunk_size,
            volume=volume,
        )
        feature_count = int(design["energy"].numel())
        include_bias_column = bool(include_bias_column)
        column_count = feature_count + int(include_bias_column)
        if accumulator is None:
            accumulator = {
                "XtX": torch.zeros(
                    (column_count, column_count),
                    dtype=design["energy"].dtype,
                    device=design["energy"].device,
                ),
                "Xty": torch.zeros(
                    column_count,
                    dtype=design["energy"].dtype,
                    device=design["energy"].device,
                ),
                "yty": torch.zeros(
                    (),
                    dtype=design["energy"].dtype,
                    device=design["energy"].device,
                ),
                "n_rows": 0,
                "n_structures": 0,
            }
        if tuple(accumulator["XtX"].shape) != (
            column_count,
            column_count,
        ):
            raise ValueError(
                "normal-equation accumulator column count does not match "
                "the compiled fixed basis"
            )

        def add_rows(rows, targets, weight):
            sqrt_weight = float(max(float(weight), 0.0) ** 0.5)
            if targets is None or sqrt_weight == 0.0:
                return
            rows = rows.reshape(-1, feature_count).to(
                dtype=accumulator["XtX"].dtype,
                device=accumulator["XtX"].device,
            )
            targets = torch.as_tensor(
                targets,
                dtype=accumulator["Xty"].dtype,
                device=accumulator["Xty"].device,
            ).reshape(-1)
            if int(rows.shape[0]) != int(targets.numel()):
                raise ValueError(
                    "reference target count does not match its design rows"
                )
            if include_bias_column:
                rows = torch.cat(
                    (
                        rows,
                        torch.zeros(
                            (int(rows.shape[0]), 1),
                            dtype=rows.dtype,
                            device=rows.device,
                        ),
                    ),
                    dim=1,
                )
            rows = rows * sqrt_weight
            targets = targets * sqrt_weight
            accumulator["XtX"].add_(rows.transpose(0, 1) @ rows)
            accumulator["Xty"].add_(rows.transpose(0, 1) @ targets)
            accumulator["yty"].add_(torch.dot(targets, targets))
            accumulator["n_rows"] += int(targets.numel())

        if energy_reference is not None and float(energy_weight) > 0.0:
            energy_row = design["energy"]
            if include_bias_column:
                energy_row = torch.cat(
                    (
                        energy_row,
                        torch.as_tensor(
                            [float(atom_types.numel())],
                            dtype=energy_row.dtype,
                            device=energy_row.device,
                        ),
                    )
                )
            sqrt_weight = float(float(energy_weight) ** 0.5)
            target = torch.as_tensor(
                energy_reference,
                dtype=accumulator["Xty"].dtype,
                device=accumulator["Xty"].device,
            ).reshape(())
            row = energy_row * sqrt_weight
            target = target * sqrt_weight
            accumulator["XtX"].add_(torch.outer(row, row))
            accumulator["Xty"].add_(row * target)
            accumulator["yty"].add_(target * target)
            accumulator["n_rows"] += 1
        add_rows(design["forces"], force_reference, force_weight)
        if stress_reference is not None:
            stress_reference = torch.as_tensor(stress_reference)
            if tuple(stress_reference.shape) == (3, 3):
                stress_reference = torch.stack(
                    (
                        stress_reference[0, 0],
                        stress_reference[1, 1],
                        stress_reference[2, 2],
                        stress_reference[1, 2],
                        stress_reference[0, 2],
                        stress_reference[0, 1],
                    )
                )
            add_rows(
                design["stress"],
                stress_reference,
                stress_weight,
            )
        accumulator["n_structures"] += 1
        accumulator["n_cols"] = column_count
        accumulator["feature_count"] = feature_count
        accumulator["include_bias_column"] = include_bias_column
        accumulator["last_structure_report"] = dict(design["report"])
        return accumulator

    def runtime_report(self):
        report = dict(self.readout.runtime_report())
        report.update(
            {
                "atomistic_source_binding": "ordinary_density_source_slots",
                "source_slot_count": int(self.slot_count),
                "atomic_channel_count": len(self.channels),
                "descriptor_jacobian_materialized": False,
                "edge_derivative_buffer_materialized": False,
                "source_reverse_path": "streaming_channel_group_vjp",
                "source_runtime": self.site_basis.source_runtime_report(),
            }
        )
        return report


class YE3TCompiledArtifactCalculator(Calculator):
    """ASE adapter for a compiled ordinary-density linear artifact."""

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, model, *, cutoff, type_map, dtype=torch.float64,
                 device=None, reference_energies=None, **kwargs):
        super().__init__(**kwargs)
        from ye3t_ace.ace.linear_ace import _normalize_reference_energies

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device is None else torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this environment.")
        self.model = model.to(self.device)
        self.model.eval()
        self.cutoff = float(cutoff)
        if self.cutoff <= 0.0:
            raise ValueError("A compiled artifact calculator requires a positive cutoff.")
        self.type_map = dict(type_map)
        self.dtype = dtype
        self.reference_energies = _normalize_reference_energies(reference_energies)

    def calculate(self, atoms=None, properties=("energy", "forces"),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        from ye3t_ace.ace.linear_ace import _reference_energy_offset_from_atoms
        from ye3t_ace.equivariant_calc.neighbors import neighbor_data_from_ase_atoms

        neighbor_data = neighbor_data_from_ase_atoms(atoms, self.cutoff, self.type_map)
        edge_vectors = torch.as_tensor(
            neighbor_data.x_ij, dtype=self.dtype, device=self.device,
        )
        edge_index = torch.as_tensor(
            neighbor_data.edge_index, dtype=torch.long, device=self.device,
        )
        atom_types = torch.as_tensor(
            neighbor_data.atom_types, dtype=torch.long, device=self.device,
        )
        needs_derivatives = "forces" in properties or "stress" in properties
        with torch.enable_grad():
            if needs_derivatives:
                output = self.model.energy_forces_virial(
                    edge_vectors, edge_index, atom_types,
                )
                energy = output["energy"]
            else:
                output = None
                energy = self.model(edge_vectors, edge_index, atom_types)
        self.results["energy"] = (
            float(energy.detach().cpu())
            + _reference_energy_offset_from_atoms(atoms, self.reference_energies)
        )
        if "forces" in properties:
            self.results["forces"] = output["forces"].detach().cpu().numpy()
        if "stress" in properties:
            volume = float(atoms.get_volume())
            if volume <= 0.0:
                raise ValueError("ASE stress requires a positive periodic cell volume.")
            derivative = output["strain_derivative"]
            stress = (0.5 * (derivative + derivative.T) / volume).detach().cpu().numpy()
            self.results["stress"] = np.asarray(
                (stress[0, 0], stress[1, 1], stress[2, 2],
                 stress[1, 2], stress[0, 2], stress[0, 1]), dtype=float,
            )


__all__ = [
    "YE3T_COMPILED_MODEL_SCHEMA",
    "YE3TCompiledModelArtifact",
    "YE3TCompiledLinearReadout",
    "YE3TCompiledLiftedRoleSourceEvaluator",
    "YE3TCompiledSourceEvaluator",
    "YE3TCompiledArtifactCalculator",
]
