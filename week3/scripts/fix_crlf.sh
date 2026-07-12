#!/bin/bash
# 去除 week3 下所有 .py/.sh 文件的 CRLF（被 scp 从 Windows 加上 \r）
find /home/ubuntu/amazon/week3 -type f \( -name '*.py' -o -name '*.sh' \) -exec sed -i 's/\r$//' {} +
echo "--- verify ---"
find /home/ubuntu/amazon/week3 -name '*.py' -print0 | xargs -0 file | head -8
