#!/bin/bash
rm -rf .git/refs/original/
rm -rf .git/logs/
git gc --aggressive --prune=now
git count-objects -v
