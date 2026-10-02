#!/bin/sh

set -eu

usage()
{
  cat <<'EOF'
Usage: run_end_to_end.sh --lmp PATH --ye3t-lammps-root PATH [options]

Fit a new tagged-Cauchy Ta residual model, export it, and validate the largest
joint arm through one- and two-rank LAMMPS. This script does not install
packages, patch LAMMPS, or modify system configuration.

Options:
  --config PATH             Config file (default: config_quick.json).
  --output PATH             New workflow directory (default: unique run under
                            $YE3T_WORKFLOW_ROOT or ~/ye3t-workflows).
  --cache-root PATH         Reusable YE3T cache root.
  --mpi-exec PATH           MPI launcher (default: $YE3T_MPIEXEC or mpiexec).
  --timeout SECONDS         Timeout per native validation run (default: 600).
  --with-kokkos             Also run the experimental unqualified /kk validator.
  --kokkos-gpus N           GPUs per node for the Kokkos check (default: 1).
EOF
}

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
config=$script_dir/config_quick.json
lmp=
ye3t_lammps_root=
output=
cache_root=
mpi_exec=${YE3T_MPIEXEC:-mpiexec}
timeout_seconds=600
with_kokkos=no
kokkos_gpus=1

while test "$#" -gt 0; do
  case "$1" in
    --config|--lmp|--ye3t-lammps-root|--output|--cache-root|--mpi-exec|--timeout|--kokkos-gpus)
      option=$1
      shift
      if test "$#" -eq 0; then
        echo "$option requires a value" >&2
        exit 2
      fi
      case "$option" in
        --config) config=$1 ;;
        --lmp) lmp=$1 ;;
        --ye3t-lammps-root) ye3t_lammps_root=$1 ;;
        --output) output=$1 ;;
        --cache-root) cache_root=$1 ;;
        --mpi-exec) mpi_exec=$1 ;;
        --timeout) timeout_seconds=$1 ;;
        --kokkos-gpus) kokkos_gpus=$1 ;;
      esac
      ;;
    --with-kokkos) with_kokkos=yes ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if test -z "$lmp" || test -z "$ye3t_lammps_root"; then
  usage >&2
  exit 2
fi
case "$timeout_seconds" in *[!0-9]*|'') echo "--timeout must be a positive integer" >&2; exit 2 ;; esac
case "$kokkos_gpus" in *[!0-9]*|'') echo "--kokkos-gpus must be a positive integer" >&2; exit 2 ;; esac
if test "$timeout_seconds" -lt 1 || test "$kokkos_gpus" -lt 1; then
  echo "--timeout and --kokkos-gpus must be positive" >&2
  exit 2
fi

case "$lmp" in
  */*) ;;
  *) lmp=$(command -v "$lmp" || true) ;;
esac
if test -z "$lmp" || test ! -x "$lmp"; then
  echo "LAMMPS executable not found: $lmp" >&2
  exit 1
fi
lmp=$(CDPATH= cd -- "$(dirname -- "$lmp")" && pwd -P)/$(basename -- "$lmp")
ye3t_lammps_root=$(CDPATH= cd -- "$ye3t_lammps_root" && pwd -P)
validator=$ye3t_lammps_root/tests/validate_tagged_cauchy_v3_lammps.py
if test ! -f "$validator"; then
  echo "Tagged V3 validator not found: $validator" >&2
  exit 1
fi
config=$(CDPATH= cd -- "$(dirname -- "$config")" && pwd -P)/$(basename -- "$config")
if test ! -f "$config"; then
  echo "Config not found: $config" >&2
  exit 1
fi

if test -z "$output"; then
  workflow_root=${YE3T_WORKFLOW_ROOT:-$HOME/ye3t-workflows}
  output=$workflow_root/MLIP/ta_tagged_cauchy_image_linear/run-$(date -u +%Y%m%dT%H%M%SZ)-$$
fi
output_parent=$(dirname -- "$output")
mkdir -p "$output_parent"
output=$(CDPATH= cd -- "$output_parent" && pwd -P)/$(basename -- "$output")
if test -e "$output"; then
  echo "Output path already exists: $output" >&2
  exit 1
fi

python_command=${PYTHON:-python3}
if ! command -v "$python_command" >/dev/null 2>&1; then
  echo "Python executable not found: $python_command" >&2
  exit 1
fi

echo "Training and exporting into $output"
if test -n "$cache_root"; then
  "$python_command" "$script_dir/train_export.py" --config "$config" \
    --lammps "$lmp" --ye3t-lammps-root "$ye3t_lammps_root" \
    --cache-root "$cache_root" --output "$output"
else
  "$python_command" "$script_dir/train_export.py" --config "$config" \
    --lammps "$lmp" --ye3t-lammps-root "$ye3t_lammps_root" \
    --output "$output"
fi

model=
for candidate in "$output"/models/train_*/s02_joint.ye3t.json; do
  if test -f "$candidate"; then
    model=$candidate
  fi
done
if test -z "$model"; then
  echo "No exported s02_joint model found under $output/models" >&2
  exit 1
fi

echo "Validating CPU/MPI model $model"
"$python_command" "$validator" --lmp "$lmp" --mpiexec "$mpi_exec" \
  --model "$model" --output-dir "$output.lammps_cpu" \
  --timeout "$timeout_seconds"

if test "$with_kokkos" = yes; then
  echo "Validating the experimental unqualified Kokkos reference"
  "$python_command" "$validator" --lmp "$lmp" --mpiexec "$mpi_exec" \
    --model "$model" --output-dir "$output.lammps_kokkos" \
    --timeout "$timeout_seconds" --kokkos --kokkos-gpus "$kokkos_gpus"
fi

echo "Training result: $output"
echo "CPU validation: $output.lammps_cpu/tagged_cauchy_v3_lammps_report.json"
if test "$with_kokkos" = yes; then
  echo "Kokkos validation: $output.lammps_kokkos/tagged_cauchy_v3_lammps_report.json"
fi
