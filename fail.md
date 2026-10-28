- 代理 `http://141.2.239.180:3128`（`Proxy-agent: netentsec`）
- 华为内网源 `mirrors.tools.huawei.com` 直连可达，外网（github / jsdelivr / pypi 公共源）直连超时

***

## 问题 1：curl 下载 GitHub Release 报 SSL 60

**命令**：`curl -fL https://github.com/... -o k.tar.gz`

**根因**：代理 `141.2.239.180:3128` 做 HTTPS 中间人，注入自签名 CA，curl 证书校验失败。

**错误**：`curl: (60) SSL certificate problem: self-signed certificate in certificate chain`

**修复命令**（显式指定代理 + `-k`）：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
curl -fLk --connect-timeout 30 -x http://141.2.239.180:3128 \
  -o /tmp/ascend-bench-kit.tar.gz \
  https://github.com/ZYY-ing/ascend-bench-web/releases/latest/download/ascend-bench-kit.tar.gz
```

> 关键：必须 `unset` 环境变量再用 `-x`，防止两者打架；直接 export 代理变量在这台机器上 curl 有时不生效。

***

## 问题 3：setup.sh pip install 慢 → 华为源 → 重复 index-url

**根因 1**：setup.sh 第 40 行硬编码阿里云源，通过项目自身 `pip.conf` + `PIP_CONFIG_FILE` 强制 pip 只用它，绕过宿主机配好的华为内网源。

```bash
# setup.sh 原始第 40 行
docker exec -w / "$CONTAINER" bash -c 'printf "[global]\nindex-url = https://mirrors.aliyun.com/pypi/simple\nprogress-bar = on\n" > "$1/pip.conf"' _ "$TARGET"
```

**修复**（改 printf 模板，用华为内网源）：

```bash
sed -i '40c\  docker exec -w / "$CONTAINER" bash -c '\''printf "[global]\\nindex-url = http://mirrors.tools.huawei.com/pypi/simple\\ntrusted-host = mirrors.tools.huawei.com\\nprogress-bar = on\\ntimeout = 30\\n" > "$1/pip.conf"'\'' _ "$TARGET"' \
  /home/z00949185/ascend-bench-kit/setup.sh
```

**根因 2**：第一次手动改 pip.conf 后 setup.sh 重跑会用 printf 覆盖。用 `sed` 直接改 setup.sh 里的模板行才是根治。

**最终 pip.conf 模板内容**：

```ini
[global]
index-url = http://mirrors.tools.huawei.com/pypi/simple
trusted-host = mirrors.tools.huawei.com
progress-bar = on
timeout = 30
```

> **教训**：`sed -i` 在 macOS BSD sed 和 GNU sed（Linux）上语法有细微差异。Linux 下用 `sed -i '40c\  ...'` 替换整行时，反斜杠续行必须精确到目标行号，否则会插出重复段（本次因 sed 匹配范围过宽导致 pip.conf 里出现两个 `index-url`，触发 `option already exists`）。

***

## 网络速查表

| 目标                                         | 直连        | 走代理 `-x 141.2.239.180:3128` | 备注             |
| ------------------------------------------ | --------- | --------------------------- | -------------- |
| `mirrors.tools.huawei.com` (华为内网源)         | ✅ 快       | ✅                           | 优先             |
| `mirrors.aliyun.com` (阿里云)                 | ❌ timeout | ✅ 慢                         | 外网             |
| `github.com` / `raw.githubusercontent.com` | ❌ timeout | ✅ + `-k`                    | 需 `-k` 跳过中间人证书 |
| `cdn.jsdelivr.net`                         | ❌ timeout | —                           | 外网             |

