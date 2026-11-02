#!/usr/bin/env bash
# Genesis + rsl_rl による Unitree Go2 歩行学習環境のセットアップ
# 参考: https://qiita.com/hEnka/items/cc5fd872eb0bf7cd3abc
#
# 記事との差分:
#  - Ubuntu 24.04 はシステム pip へのインストールが禁止 (PEP 668) のため uv の venv を使う
#  - 現行の Genesis examples は rsl-rl-lib>=5.0.0 を要求する (記事の v1.0.2 は使えない)
#  - Genesis は clone したものを editable install するので、examples とバージョンが一致し
#    記事の rendered_envs_idx エラー対処は不要
set -euo pipefail
cd "$(dirname "$0")"

# 動作を確認した Genesis のコミット (genesis-world 1.4.5)
GENESIS_COMMIT=1daee8d5210cdfdfb3e238c592bfabe7f7b11814
if [ ! -d Genesis ]; then
    git init -q Genesis
    git -C Genesis remote add origin https://github.com/Genesis-Embodied-AI/Genesis.git
    git -C Genesis fetch -q --depth 1 origin "$GENESIS_COMMIT"
    git -C Genesis checkout -q FETCH_HEAD
fi

uv venv --python 3.12 .venv
source .venv/bin/activate

uv pip install torch
uv pip install -e ./Genesis "rsl-rl-lib==5.5.2" tensorboard markdown

python -c "import torch, genesis; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('genesis', genesis.__version__)"
