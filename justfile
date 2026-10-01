# Default recipe: show available commands.
default:
    @just --list

# Show this repo's notes vault in the browser (needs the  tool on PATH).
vault *ARGS:
    vault-serve {{ARGS}}
