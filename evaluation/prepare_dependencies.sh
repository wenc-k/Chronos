#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
checkout="$here/.deps/OpenS2V-Nexus"
revision=85d4c3bb7e4ca3297d4e96e581454535b668fb56
if [[ ! -d "$checkout/.git" ]]; then
    mkdir -p "$here/.deps"
    git clone --no-checkout https://github.com/PKU-YuanGroup/OpenS2V-Nexus.git "$checkout"
    git -C "$checkout" checkout --detach "$revision"
fi
if [[ $(git -C "$checkout" rev-parse HEAD) != "$revision" ]]; then
    echo "Unexpected OpenS2V-Nexus revision at $checkout" >&2
    exit 1
fi
mkdir -p "$here/eval_opens2v/utils"
for component in Q-Align yoloworld; do
    target="$here/eval_opens2v/utils/$component"
    expected="$checkout/eval/utils/$component"
    [[ -d "$expected" ]]
    if [[ -L "$target" && "$(readlink -f "$target")" == "$expected" ]]; then
        continue
    fi
    ln -s -- "$expected" "$target"
done
