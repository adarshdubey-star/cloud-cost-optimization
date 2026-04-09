#!/bin/bash
# =============================================================
# Load Generator for GCP Workload VMs
#
# Simulates realistic cloud workload patterns by SSH'ing into
# MIG instances and running stress-ng with varying intensity.
#
# Usage:
#   ./scripts/generate_load.sh <PROJECT_ID> <ZONE> <MIG_NAME> [DURATION_MINUTES]
#
# Prerequisites:
#   - gcloud CLI authenticated
#   - stress-ng installed on VMs (done via Terraform startup script)
# =============================================================

set -e

PROJECT_ID=${1:?"Usage: $0 <PROJECT_ID> <ZONE> <MIG_NAME> [DURATION_MIN]"}
ZONE=${2:?"Usage: $0 <PROJECT_ID> <ZONE> <MIG_NAME> [DURATION_MIN]"}
MIG_NAME=${3:?"Usage: $0 <PROJECT_ID> <ZONE> <MIG_NAME> [DURATION_MIN]"}
DURATION_MIN=${4:-30}

echo "============================================"
echo "  Cloud Workload Load Generator"
echo "  Project: $PROJECT_ID"
echo "  Zone: $ZONE"
echo "  MIG: $MIG_NAME"
echo "  Duration: ${DURATION_MIN} minutes"
echo "============================================"

# Get instance names in the MIG
INSTANCES=$(gcloud compute instance-groups managed list-instances "$MIG_NAME" \
    --zone="$ZONE" --project="$PROJECT_ID" \
    --format="value(instance)" 2>/dev/null)

if [ -z "$INSTANCES" ]; then
    echo "ERROR: No instances found in MIG '$MIG_NAME'"
    exit 1
fi

echo "Found instances:"
echo "$INSTANCES" | while read inst; do echo "  - $inst"; done

# Workload phases simulating diurnal pattern
# Each phase: (cpu_load_percent, duration_seconds)
TOTAL_SECONDS=$((DURATION_MIN * 60))
PHASE_DURATION=$((TOTAL_SECONDS / 6))

PHASES=(
    "20:Low load (off-peak)"
    "50:Medium load (ramp-up)"
    "80:High load (peak)"
    "95:Spike load (burst)"
    "60:Declining load"
    "25:Low load (recovery)"
)

echo ""
echo "Starting workload generation..."
echo ""

for phase_info in "${PHASES[@]}"; do
    CPU_LOAD="${phase_info%%:*}"
    PHASE_NAME="${phase_info##*:}"
    WORKERS=$(nproc 2>/dev/null || echo "2")

    echo "[$(date '+%H:%M:%S')] Phase: $PHASE_NAME (CPU target: ${CPU_LOAD}%) for ${PHASE_DURATION}s"

    for INSTANCE in $INSTANCES; do
        gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT_ID" \
            --command="stress-ng --cpu $WORKERS --cpu-load $CPU_LOAD --timeout ${PHASE_DURATION}s --quiet" \
            --quiet -- -o StrictHostKeyChecking=no &
    done

    sleep "$PHASE_DURATION"
    echo "  Phase complete."
    echo ""
done

echo "============================================"
echo "  Load generation complete!"
echo "  Total duration: ${DURATION_MIN} minutes"
echo "============================================"
