#!/bin/bash
#SBATCH --job-name=rl-main
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=vorname.nachnamefallsihrwollt@stud.uni-hannover.de

# ============================================================
# GPU-Partition wählen: aktiv ist "kisski-h100"
# (größte verfügbare GPU: H100, 94 GiB VRAM).
# Zum Wechseln: aktive Zeilen zu "##SBATCH" machen,
# gewünschte Alternative von "##SBATCH" zu "#SBATCH" ändern.
# Für einen vollwertigen Submission-Lauf eine echte GPU nehmen,
# NICHT die grete:interactive MIG-Slice.
# ============================================================

# #SBATCH -p kisski-h100
# #SBATCH -G H100:1

## Alternative 1: kisski (A100, 80 GiB) - falls kisski-h100 voll ist
#SBATCH -p kisski
#SBATCH -G A100:1

## Alternative 2: grete:interactive (A100 MIG-Slice, 20 GiB)
## Nur für schnelles Debuggen/Testen, NICHT für vollwertiges Training!
##SBATCH -p grete:interactive
##SBATCH -G 1g.20gb

# --- Wir haben aktuell keinen Zugriff auf diese Nodes ---
# Ich habe sie der Vollständigkeit halber und zur Dokumentation hier gelassen ---
## Alternative 3: grete:shared (A100, 80 GiB) - nur falls du zusätzlich NHR-Zugriff hast
##SBATCH -p grete:shared
##SBATCH -G A100:1

## Alternative 4: grete-h100:shared (H100, 94 GiB) - nur falls du zusätzlich NHR-Zugriff hast
##SBATCH -p grete-h100:shared
##SBATCH -G H100:1

# ============================================================

set -euo pipefail

# In das Verzeichnis wechseln, aus dem sbatch aufgerufen wurde (Repo-Root),
# damit relative Pfade (envs/, Makefile, datasets/) stimmen.
cd "${SLURM_SUBMIT_DIR}"

echo "Job gestartet: $(date)"
echo "Job-ID: ${SLURM_JOB_ID}"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "Node: ${SLURM_NODELIST}"

# Conda-Umgebung laden (siehe SETUP.md: envs/automl-env, Python 3.12 + CUDA 12.8)
module load miniforge3
source activate ../../envs/rl-env

# Debug-Infos
python --version
python -m torch.utils.collect_env
nvidia-smi

# Python-Ausgabe unbuffered machen: ohne das puffert Python stdout blockweise,
# sobald es nicht an ein Terminal geschrieben wird - sattach würde dann lange
# nichts anzeigen. PYTHONUNBUFFERED wirkt auf den Python-Prozess selbst,
# `srun --unbuffered` zusätzlich auf die I/O-Weiterleitung von Slurm.
export PYTHONUNBUFFERED=1

# Running the main als eigener Job-Step (via srun), damit man sich mit
# `sattach <jobid>.0` an die laufende Ausgabe hängen kann, z.B.:
#   squeue -u $USER            # Job-ID herausfinden
#   sattach <jobid>.0          # live an stdout/stderr andocken (mehrfach möglich)
srun --unbuffered python main.py