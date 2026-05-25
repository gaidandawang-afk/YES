# SGLang AutoDL Git 工作流

更新时间：2026-05-21

这份文档记录如何从 Windows 本地仓库向 AutoDL 服务器上的 SGLang 仓库提交代码、如何让服务器工作区切到对应代码，以及如何把同一份代码推送到 GitHub review remote。

## 当前仓库

本地 clone：

```text
F:\theend\repo\sglang-autodl
```

服务器仓库：

```text
/root/autodl-tmp/public/iws/projects/sglang
```

本地 `origin`：

```text
ssh://root@region-42.seetacloud.com:38651/root/autodl-tmp/public/iws/projects/sglang
```

本地 `github`：

```text
git@github.com:gaidandawang-afk/sglang.git
```

当前已确认：

```text
本地分支:     codex/first-ft-sentinel-registration
本地 HEAD:     03034c166 Fix FT sentinel registration for DP workers
AutoDL 分支:   codex/first-ft-sentinel-registration
GitHub 分支:   codex/first-ft-sentinel-registration
GitHub commit: 03034c166e388409c37d517ab63ff550d00c14a1
```

## Windows Git 可执行文件

当前 Windows `PATH` 里没有普通 `git`。使用 Visual Studio bundled Git：

```powershell
$GIT = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\CommonExtensions\Microsoft\TeamFoundation\Team Explorer\Git\cmd\git.exe'
```

后续命令都可以写成：

```powershell
& $GIT -C F:\theend\repo\sglang-autodl <git-subcommand>
```

## SSH Host 分流配置

当前采用 `C:\Users\59699\.ssh\config` 做 host-aware SSH 分流：

```text
Host github.com
  HostName ssh.github.com
  Port 443
  User git
  IdentityFile C:/Users/59699/.ssh/id_rsa
  IdentitiesOnly yes
  BatchMode yes
  StrictHostKeyChecking accept-new

Host region-42.seetacloud.com
  HostName region-42.seetacloud.com
  Port 38651
  User root
  IdentityFile C:/Users/59699/Documents/sglang/.ssh/sglang_autodl_git_key
  IdentitiesOnly yes
  BatchMode yes
  StrictHostKeyChecking accept-new
```

关键点：

```text
GitHub SSH 22 端口在当前网络下会被关闭/reset。
GitHub SSH over 443 已验证可用，认证账号为 gaidandawang-afk。
AutoDL 继续使用 region-42.seetacloud.com:38651 和 AutoDL 专用 key。
```

AutoDL 私钥：

```text
C:\Users\59699\Documents\sglang\.ssh\sglang_autodl_git_key
```

GitHub 私钥：

```text
C:\Users\59699\.ssh\id_rsa
```

当前 repo-local Git config 不再设置 `core.sshCommand`，让 Git 正常读取 `C:\Users\59699\.ssh\config`：

```powershell
& $GIT -C F:\theend\repo\sglang-autodl config --unset core.sshCommand
& $GIT -C F:\theend\repo\sglang-autodl config ssh.variant ssh
```

历史上曾使用过 AutoDL 专用 wrapper：

```text
C:\Users\59699\Documents\sglang\.ssh\sglang_autodl_git_ssh.cmd
```

现在不要再把它配置到 `core.sshCommand`，否则 GitHub remote 也会被迫使用 AutoDL key，导致 GitHub SSH 推送失败。

远端服务器已配置：

```text
/root/.ssh/authorized_keys 包含对应 public key
/root/.ssh 权限 700
/root/.ssh/authorized_keys 权限 600
```

## 连通性检查

```powershell
$GIT = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\CommonExtensions\Microsoft\TeamFoundation\Team Explorer\Git\cmd\git.exe'
$REPO = 'F:\theend\repo\sglang-autodl'

& $GIT -C $REPO status --short
& $GIT -C $REPO remote -v
& $GIT -C $REPO ls-remote origin HEAD
& $GIT -C $REPO ls-remote github HEAD
```

GitHub SSH 认证检查：

```powershell
$SSH = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\CommonExtensions\Microsoft\TeamFoundation\Team Explorer\Git\usr\bin\ssh.exe'

& $SSH -T git@github.com
```

成功时会看到类似输出，并以非 0 退出，这是 GitHub 不提供 shell 的正常行为：

```text
Hi gaidandawang-afk! You've successfully authenticated, but GitHub does not provide shell access.
```

已经验证过的 push 流程：

```powershell
& $GIT -C $REPO push --dry-run origin HEAD:refs/heads/codex/push-connectivity-test
& $GIT -C $REPO push origin HEAD:refs/heads/codex/push-connectivity-test-20260520
& $GIT -C $REPO push origin --delete codex/push-connectivity-test-20260520
& $GIT -C $REPO push github HEAD:refs/heads/codex/first-ft-sentinel-registration
```

结果：AutoDL 创建和删除临时远端分支成功；GitHub review 分支推送成功。

## 推荐开发流程

### 1. 本地创建工作分支

当前本地 `main` 已和服务器 `origin/main` 对齐。开发新任务时先创建分支：

```powershell
$GIT = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\CommonExtensions\Microsoft\TeamFoundation\Team Explorer\Git\cmd\git.exe'
$REPO = 'F:\theend\repo\sglang-autodl'

& $GIT -C $REPO switch -c codex/<task-name>
```

如果已经有分支：

```powershell
& $GIT -C $REPO switch codex/<task-name>
```

### 2. 修改、检查、提交

```powershell
& $GIT -C $REPO status --short
& $GIT -C $REPO diff
& $GIT -C $REPO add <files>
& $GIT -C $REPO commit -m "<message>"
```

不要把无关文件、日志、模型、缓存、venv 内容加入 Git。

### 3. 推送到 AutoDL 服务器

推荐先推到任务分支：

```powershell
& $GIT -C $REPO push origin HEAD:refs/heads/codex/<task-name>
```

如果明确要更新服务器 `main`：

```powershell
& $GIT -C $REPO push origin HEAD:refs/heads/main
```

注意：服务器仓库不是 bare repo。push 到 `refs/heads/main` 或 `refs/heads/codex/...` 通常只更新服务器 repo 的 ref；运行中的服务也不会因为 push 自动切换代码。部署时仍需要 SSH 到服务器显式检查并切换工作区。

### 4. 推送到 GitHub review remote

确认 remote：

```powershell
& $GIT -C $REPO remote -v
```

应包含：

```text
github git@github.com:gaidandawang-afk/sglang.git
```

如果没有，添加：

```powershell
& $GIT -C $REPO remote add github git@github.com:gaidandawang-afk/sglang.git
```

推送当前分支到 GitHub：

```powershell
& $GIT -C $REPO push github HEAD:refs/heads/codex/<task-name>
```

当前已验证：

```powershell
& $GIT -C $REPO push github HEAD:refs/heads/codex/first-ft-sentinel-registration
```

PR 创建地址：

```text
https://github.com/gaidandawang-afk/sglang/pull/new/codex/<task-name>
```

## 服务器切换代码

push 之后，SSH 到服务器显式切换代码：

```bash
cd /root/autodl-tmp/public/iws/projects/sglang
git status --short
git branch --show-current || true
git log -1 --oneline
```

如果工作区干净，可以切到远端已推分支：

```bash
git switch codex/<task-name>
git log -1 --oneline
```

或者部署精确 commit：

```bash
git switch --detach <commit-sha>
git log -1 --oneline
```

如果 `git status --short` 显示有本地修改，先判断是不是当前任务产生的修改。不要随手 `git reset --hard`，除非明确知道这些改动可以丢弃并已得到确认。

## 启动验证

服务器切到目标 commit 后，按 `sglang_manual_ops.md` 的 clean-env 命令启动：

```bash
curl -fsS http://127.0.0.1:30000/get_model_info
curl -fsS http://127.0.0.1:30000/fault_tolerance/status
curl -fsS http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello, my name is","sampling_params":{"max_new_tokens":8,"temperature":0}}'
```

## 与 GitHub remote 交互

当前推荐方式是通过 `C:\Users\59699\.ssh\config` 做 Host 分流，并在 repo 中保留两个 remote：

```text
origin -> AutoDL 非 bare 工作区仓库
github -> GitHub review 仓库
```

当前配置：

```text
origin  ssh://root@region-42.seetacloud.com:38651/root/autodl-tmp/public/iws/projects/sglang
github  git@github.com:gaidandawang-afk/sglang.git
```

GitHub 使用 SSH over 443：

```text
github.com -> ssh.github.com:443
```

这避免了当前环境下 GitHub SSH 22 端口被关闭/reset 的问题。

检查 GitHub 分支：

```powershell
& $GIT -C $REPO ls-remote github refs/heads/codex/<task-name>
```

不要重新设置 repo-local `core.sshCommand` 到 AutoDL wrapper；它会绕过 host-aware SSH config。

## 常见问题

`git` 命令找不到：

```text
使用 Visual Studio bundled Git 的完整路径。
```

push 成功但服务器代码没变：

```text
这是正常的。push 更新 refs，不自动切换服务器 working tree。SSH 到服务器 git switch/git switch --detach。
```

push 到 main 被拒绝：

```text
检查服务器当前是否正在 checkout main。非 bare repo 不能安全 push 到当前 checked-out 分支。推荐推 codex/<task-name> 分支，再 SSH 到服务器切换。
```

GitHub SSH 失败：

```text
先检查 C:\Users\59699\.ssh\config 是否存在 github.com -> ssh.github.com:443。
再检查 repo-local config 是否误设 core.sshCommand。
如果 core.sshCommand 指向 AutoDL wrapper，执行 git config --unset core.sshCommand。
```

## 安全边界

不要提交：

```text
passwd.txt
.ssh/*
venv/
caches/
logs/
models/
```

不要把服务器密码、私钥内容或 token 写入 commit message、文档或日志。
