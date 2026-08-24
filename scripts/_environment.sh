#!/usr/bin/env bash

# Activate a virtual environment only when VENV is explicitly provided.
# VENV may point either to the environment directory or to its activate file.
repair_activate_environment() {
  local activate_path="${VENV:-}"
  if [[ -z "$activate_path" ]]; then
    return 0
  fi
  if [[ -d "$activate_path" ]]; then
    activate_path="$activate_path/bin/activate"
  fi
  if [[ ! -f "$activate_path" ]]; then
    echo "ABORT: VENV does not contain an activation script: $activate_path" >&2
    return 2
  fi
  # shellcheck disable=SC1090
  source "$activate_path"
}
