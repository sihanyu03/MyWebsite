#!/usr/bin/env bash
# bash publish.sh
set -e
git checkout --orphan _publish
git add -A
git commit -m "Site snapshot $(date +%d-%m-%Y)"
git push public _publish:main --force-with-lease
git checkout main
git branch -D _publish