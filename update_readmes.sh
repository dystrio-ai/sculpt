#!/usr/bin/env bash
set -e

REPOS=(
  "Mistral-7B-v0.1-sculpt-conservative"
  "Mistral-7B-v0.1-sculpt-balanced"
)

for REPO in "${REPOS[@]}"; do
  echo "Updating README for $REPO"

  cd "$REPO"

  sed -i.bak 's/from_pretrained(repo, subfolder="model")/from_pretrained(repo)/g' README.md

  rm -f README.md.bak

  git add README.md
  git commit -m "Update quickstart: remove subfolder=\"model\" (files now at repo root)" || true
  git push

  cd ..
done

echo "Done."
