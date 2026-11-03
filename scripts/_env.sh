# Sourced by the scripts in this folder: repo folders and settings from .env (see .env.example).
# Values: environment variable, else the repo's .env, else ~/.config/dobot-pnp/.env, else the default.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
N="$ROOT/src"
envget() {   # envget KEY [default]
  local k="$1" v="${!1:-}" f
  if [ -z "$v" ]; then
    for f in "${DOBOT_ENV_FILE:-}" "$ROOT/.env" "$HOME/.config/dobot-pnp/.env"; do
      [ -n "$f" ] && [ -f "$f" ] || continue
      v="$(sed -n -E "s/^[[:space:]]*(export[[:space:]]+)?$k[[:space:]]*=[[:space:]]*//p" "$f" | tail -n 1)"
      case "$v" in
        \"*\"*) v="${v#\"}"; v="${v%%\"*}" ;;
        \'*\'*) v="${v#\'}"; v="${v%%\'*}" ;;
        *) v="$(printf '%s' "$v" | sed -E 's/[[:space:]]+#.*$//; s/[[:space:]]+$//')" ;;
      esac
      break
    done
  fi
  printf '%s' "${v:-${2:-}}"
}
abspath() {  # ~ and repo-relative paths -> absolute
  local p="${1/#\~/$HOME}"
  case "$p" in /*) printf '%s' "$p" ;; *) printf '%s' "$ROOT/$p" ;; esac
}
ASSETS="$(abspath "$(envget DOBOT_ASSETS assets)")"
DATA="$(abspath "$(envget DOBOT_DATA .)")"
RPY="$(abspath "$(envget DOBOT_REAL_PY "$HOME/dobot_real/venv/bin/python")")"
