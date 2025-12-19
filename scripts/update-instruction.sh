#!/bin/bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop
#
# Update VLA task instruction via ROS 2 topic
#
# Usage:
#   scripts/update-instruction "pick up the red cube"
#   scripts/update-instruction -s vla-sim "pick up the object"
#   scripts/update-instruction --service lift-vla "place the object"

set -euo pipefail

# Default values
SERVICE=""  # Auto-detect if not specified
TOPIC="/vla/update_instruction"
VLA_SERVICES=("vla-real" "vla-sim" "lift-vla")

# Help message
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] <instruction>

Send a task instruction to the VLA inference node via ROS 2 topic.

Arguments:
  instruction     The task instruction text (required)

Options:
  -s, --service SERVICE   Docker Compose service name (auto-detect if omitted)
                          Available: vla-real, vla-sim, lift-vla
  -t, --topic TOPIC       ROS 2 topic name (default: /vla/update_instruction)
  -h, --help              Show this help message

Examples:
  $(basename "$0") "pick up the red cube"
  $(basename "$0") -s vla-sim "place the object on the table"
  $(basename "$0") --service lift-vla "pick up the object"
EOF
    exit 0
}

# Auto-detect running VLA service
detect_vla_service() {
    for svc in "${VLA_SERVICES[@]}"; do
        if docker compose ps --status running "$svc" 2>/dev/null | grep -q "$svc"; then
            echo "$svc"
            return 0
        fi
    done
    return 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--service)
            SERVICE="$2"
            shift 2
            ;;
        -t|--topic)
            TOPIC="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        -*)
            echo "Error: Unknown option: $1" >&2
            echo "Use --help for usage information" >&2
            exit 1
            ;;
        *)
            INSTRUCTION="$1"
            shift
            ;;
    esac
done

# Validate instruction
if [[ -z "${INSTRUCTION:-}" ]]; then
    echo "Error: Instruction text is required" >&2
    echo "Use --help for usage information" >&2
    exit 1
fi

# Change to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Auto-detect service if not specified
if [[ -z "$SERVICE" ]]; then
    SERVICE=$(detect_vla_service) || {
        echo "Error: No VLA service is running" >&2
        echo "Start one with: docker compose --profile vla up" >&2
        echo "             or: docker compose --profile vla-sim up" >&2
        exit 1
    }
    echo "Auto-detected service: $SERVICE"
else
    # Check if specified service is running
    if ! docker compose ps --status running "$SERVICE" 2>/dev/null | grep -q "$SERVICE"; then
        echo "Error: Service '$SERVICE' is not running" >&2
        echo "Start it with: docker compose --profile <profile> up" >&2
        exit 1
    fi
fi

echo "Sending instruction to $SERVICE..."
echo "  Topic: $TOPIC"
echo "  Instruction: $INSTRUCTION"

# Send the instruction via ros2 topic pub (--once for single message)
# Use bash -c to source ROS 2 environment
docker compose exec -e VLA_INSTRUCTION="$INSTRUCTION" "$SERVICE" \
    bash -c 'source /opt/ros/humble/setup.bash && ros2 topic pub --once /vla/update_instruction std_msgs/msg/String "{data: \"$VLA_INSTRUCTION\"}"'

echo "Done."
