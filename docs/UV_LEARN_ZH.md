# uv 学习笔记（对比 Go / Java 技术栈）

> 面向已有 Golang / Java 背景的开发者，学习 Python 的现代包+项目管理工具 `uv`。
> 以本仓库 `OmniVoice` 为示例项目。

---

## 0. 一句话理解 uv

**uv ≈ Rust 写的 `cargo` for Python**，一个工具同时干完：

- 包管理（pip）
- 虚拟环境（venv）
- Python 解释器管理（pyenv）
- 锁文件 + 可复现构建（pip-tools / poetry）
- 全局 CLI 工具安装（pipx）
- 打包发布（build + twine）

由 [Astral](https://astral.sh)（也写了 `ruff`）用 Rust 实现，速度比 pip 快 10–100 倍。

---

## 1. 跨语言生态对照表

理解 uv 在 Python 生态的位置，最快的方式是和 Go / Java 横向对比：

| 能力 | Python（旧） | **Python（uv）** | Golang | Java（Maven） | Java（Gradle） |
|---|---|---|---|---|---|
| 项目元信息文件 | `setup.py` / `requirements.txt` | `pyproject.toml` | `go.mod` | `pom.xml` | `build.gradle` |
| 依赖锁文件 | 无（或 `requirements.txt`） | `uv.lock` | `go.sum` | 无（靠 `pom.xml` 精确版本） | `gradle.lockfile` |
| 安装/同步依赖 | `pip install -r requirements.txt` | `uv sync` | `go mod tidy` + `go build` | `mvn install` | `gradle build` |
| 添加依赖 | 手改 + `pip install` | `uv add 包名` | `go get 包名` | 手改 `pom.xml` | 手改 `build.gradle` |
| 删除依赖 | 手改 + `pip uninstall` | `uv remove 包名` | `go mod tidy`（自动） | 手改 `pom.xml` | 手改 `build.gradle` |
| 运行入口程序 | `python xxx.py` | `uv run python xxx.py` | `go run main.go` | `mvn exec:java` | `gradle run` |
| 升级所有依赖 | 手动 | `uv lock --upgrade` | `go get -u ./...` | `mvn versions:use-latest-versions` | `gradle dependencyUpdates` |
| 环境隔离 | `venv` / `conda` | `.venv`（uv 自动建） | **无需**（编译型语言） | **无需** | **无需** |
| 语言版本管理 | `pyenv` | `uv python install 3.11` | `gvm` / `goenv` | `sdkman` / `jenv` | `sdkman` / `jenv` |
| 全局 CLI 工具 | `pipx install ruff` | `uv tool install ruff` | `go install xxx@latest` | N/A | N/A |
| 临时跑工具（不安装） | N/A | `uvx ruff check .` | N/A（要先 `go install`） | N/A | N/A |
| 私有/镜像源 | `pip.conf` | `[[tool.uv.index]]` | `GOPROXY` | `<repository>` in `settings.xml` | `repositories { }` |
| 全局缓存 | `~/.cache/pip` | `~/.cache/uv`（硬链接复用） | `$GOPATH/pkg/mod` | `~/.m2/repository` | `~/.gradle/caches` |
| 构建 + 发布 | `python -m build` + `twine` | `uv build` + `uv publish` | `goreleaser` | `mvn deploy` | `gradle publish` |

### 关键心智差异

| 维度 | Go / Java | Python（uv） |
|---|---|---|
| 是否需要"虚拟环境" | **不需要**，依赖直接编译进二进制 / classpath | **必须**，因为 Python 是动态解释 + 全局 site-packages 易污染 |
| 是否需要管理语言版本 | 偶尔（多项目用不同 Go/JDK 版本） | **常态**（项目要 3.10，系统 3.13） |
| 锁文件强制性 | Go 强制（`go.sum`）；Maven 默认无 | uv 强烈推荐（自动维护 `uv.lock`） |
| 一个项目一个根 | `go.mod` / `pom.xml` 即项目根 | `pyproject.toml` 即项目根 |

---

## 2. 文件对照：pyproject.toml ↔ pom.xml / go.mod

以本项目 `OmniVoice` 的 `pyproject.toml` 为例：

```toml
[project]
name = "omnivoice"
version = "0.1.5"
requires-python = ">=3.10"

dependencies = [
    "torch>=2.4",
    "transformers>=5.3.0",
    "librosa",
]

[project.optional-dependencies]
eval = ["jiwer==3.1.0", "s3prl", "funasr"]

[project.scripts]
omnivoice-demo = "omnivoice.cli.demo:main"

[tool.uv.sources]
torch = [{ index = "pytorch-cuda", marker = "sys_platform == 'linux'" }]

[[tool.uv.index]]
name = "pytorch-cuda"
url = "https://download.pytorch.org/whl/cu128"
```

对应到其他生态：

### Go 版（`go.mod`）

```go
module github.com/k2-fsa/omnivoice

go 1.21

require (
    github.com/some/torch v2.4.0
    github.com/some/transformers v5.3.0
)
```

差异：
- Go 没有"可选依赖组"（`[eval]`）概念，要拆模块或用 build tag
- Go 没有"自定义 index 源"（除了 `GOPROXY` 一刀切）
- Go 没有"项目入口脚本注册"，靠 `cmd/xxx/main.go` 约定

### Java Maven 版（`pom.xml`）

```xml
<project>
    <groupId>com.k2fsa</groupId>
    <artifactId>omnivoice</artifactId>
    <version>0.1.5</version>

    <properties>
        <maven.compiler.source>17</maven.compiler.source>
    </properties>

    <dependencies>
        <dependency>
            <groupId>org.pytorch</groupId>
            <artifactId>torch</artifactId>
            <version>[2.4,)</version>
        </dependency>
    </dependencies>

    <profiles>
        <profile>
            <id>eval</id>
            <dependencies>
                <dependency>
                    <groupId>com.jiwer</groupId>
                    <artifactId>jiwer</artifactId>
                    <version>3.1.0</version>
                </dependency>
            </dependencies>
        </profile>
    </profiles>
</project>
```

差异：
- 可选依赖在 Maven 里叫 **profile**，激活方式 `mvn -P eval install`
- Maven 的私有源在 `<repositories>` 配置
- Java 入口在 `META-INF/MANIFEST.MF` 或主类参数指定

---

## 3. 常用命令速查（对照表）

### 3.1 项目层（推荐主用）

| 场景 | uv | Go | Maven | Gradle |
|---|---|---|---|---|
| 装齐项目依赖 | `uv sync` | `go mod download` | `mvn install -DskipTests` | `gradle build -x test` |
| 装含可选依赖 | `uv sync --extra eval` | N/A（build tag） | `mvn install -P eval` | `gradle build -PuseEval` |
| 添加依赖 | `uv add requests` | `go get rsc.io/quote` | 改 pom + `mvn install` | 改 build.gradle + `gradle build` |
| 添加开发依赖 | `uv add --dev pytest` | （test 文件自动） | `<scope>test</scope>` | `testImplementation 'xxx'` |
| 删除依赖 | `uv remove requests` | 删 import + `go mod tidy` | 改 pom | 改 build.gradle |
| 升级所有依赖 | `uv lock --upgrade` | `go get -u ./...` | `mvn versions:use-latest-versions` | `gradle dependencyUpdates` |
| 升级单个依赖 | `uv lock --upgrade-package torch` | `go get xxx@latest` | `mvn versions:use-latest-versions -Dincludes=xxx` | 改单个版本号 |
| 跑入口程序 | `uv run python xxx.py` | `go run main.go` | `mvn exec:java -Dexec.mainClass=...` | `gradle run` |
| 跑项目里的 entry point | `uv run omnivoice-demo` | `go run ./cmd/demo` | `mvn exec:java -Pdemo` | `gradle :demo:run` |
| 看依赖树 | `uv tree` | `go mod graph` | `mvn dependency:tree` | `gradle dependencies` |

### 3.2 环境/解释器层

| 场景 | uv | Go | Java |
|---|---|---|---|
| 装某个语言版本 | `uv python install 3.11` | `gvm install go1.21` | `sdk install java 21-tem` |
| 固定项目语言版本 | `uv python pin 3.11` （写 `.python-version`） | `go 1.21` 写进 `go.mod` | `pom.xml` 里 `<maven.compiler.source>21</source>` |
| 列出可用版本 | `uv python list` | `gvm listall` | `sdk list java` |
| 创建/重建环境 | `uv venv` | N/A | N/A |

### 3.3 工具层

| 场景 | uv | Go | Java |
|---|---|---|---|
| 全局装一个 CLI 工具 | `uv tool install ruff` | `go install xxx@latest` | N/A（一般用系统包） |
| 临时跑一次工具 | `uvx ruff check .` | N/A | N/A |
| 列已装工具 | `uv tool list` | `ls $GOBIN` | N/A |

### 3.4 兼容/底层

| 场景 | 命令 | 说明 |
|---|---|---|
| 当 pip 用 | `uv pip install xxx` | 完全兼容 pip CLI，零迁移成本，速度快 10–100 倍 |
| 当 venv 用 | `uv venv` | 单独创建 `.venv` |
| 替代 pip-tools | `uv pip compile requirements.in -o requirements.txt` | 编译锁文件 |
| 清理缓存 | `uv cache clean` | 类似 `go clean -modcache` |

---

## 4. 完整工作流（用 OmniVoice 串一遍）

### Python (uv) 版

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

cd /Users/bytedance/codespace/OmniVoice
uv python install 3.11
uv python pin 3.11
uv sync                              # 自动建 .venv + 装齐所有依赖（含 GPU torch）

uv run omnivoice-demo                # 跑 entry point
uv run python omnivoice/cli/demo.py  # 直接跑脚本

uv add rich                          # 加新依赖
uv sync --extra eval                 # 加可选依赖组
uv tree --depth 1                    # 看依赖树
uv lock --upgrade && uv sync         # 升级所有依赖
```

### 对应的 Go 版

```bash
brew install go
gvm install go1.21 && gvm use go1.21

cd my-go-project
go mod download                      # 装齐依赖

go run ./cmd/demo                    # 跑入口

go get github.com/rivo/tview         # 加新依赖
go mod tidy                          # 清理无用依赖
go mod graph                         # 看依赖树
go get -u ./...                      # 升级所有依赖
```

### 对应的 Java (Maven) 版

```bash
sdk install java 21-tem
sdk use java 21-tem

cd my-java-project
mvn install -DskipTests              # 装齐依赖

mvn exec:java -Dexec.mainClass=com.example.Demo   # 跑入口

# 加依赖：手改 pom.xml 然后 mvn install
mvn dependency:tree                  # 看依赖树
mvn versions:use-latest-versions     # 升级所有依赖
```

---

## 5. 关键概念映射

### "虚拟环境" 在 Go / Java 里对应什么？

**答：什么都不对应。** Go / Java 不需要虚拟环境，因为：

- **Go**：依赖在 `$GOPATH/pkg/mod` 全局缓存，但编译时按 `go.mod` 选版本，最终编译进单个二进制，不会运行时冲突
- **Java**：依赖在 classpath 里指定，JVM 启动时按需加载，每个 `java -jar xxx.jar` 是独立进程
- **Python**：解释器全局只有一份 `site-packages`，装了 `numpy 2.x` 后所有项目都用这个版本 → 必须虚拟环境隔离

uv 的 `.venv/` 本质就是给每个项目一个独立的 `site-packages`，Go/Java 工程师不必为此焦虑，把它理解成"项目本地依赖目录"即可，类似 Node 的 `node_modules`。

### "锁文件" 在三个生态的对应

| 生态 | 锁文件 | 谁生成 | 是否提交 git |
|---|---|---|---|
| uv | `uv.lock` | `uv sync` / `uv add` / `uv lock` | **必须提交** |
| Go | `go.sum` | `go mod download` / `go get` | **必须提交** |
| Maven | 无（默认） | — | — |
| Gradle | `gradle.lockfile` | 启用 dep locking 后生成 | 必须提交 |
| npm/yarn | `package-lock.json` / `yarn.lock` | `npm install` | 必须提交 |

uv 的 `uv.lock` 更接近 `package-lock.json` 的体验：跨平台、确定性、自动维护。

### "项目脚本" 在三个生态的对应

uv 的：
```toml
[project.scripts]
omnivoice-demo = "omnivoice.cli.demo:main"
```

`uv sync` 后会在 `.venv/bin/` 生成可执行文件 `omnivoice-demo`，`uv run omnivoice-demo` 直接跑。

类比：
- **Go**：`cmd/demo/main.go` + `go install ./cmd/demo` → 生成 `$GOBIN/demo` 二进制
- **Maven**：`<plugin>` 配 `exec-maven-plugin` 或打 `<packaging>jar</packaging>` + 入口 manifest

---

## 6. 速查便签（贴墙版）

```text
日常 5 个：
  uv sync                            装/同步项目依赖
  uv run xxx                         在项目环境里跑命令
  uv add 包名                         加依赖
  uv remove 包名                      删依赖
  uv lock --upgrade                  升级所有依赖（再 sync 生效）

环境 3 个：
  uv python install 3.11             装语言版本（替代 pyenv）
  uv python pin 3.11                 项目固定版本
  uv venv                            手动建虚拟环境

工具 2 个：
  uv tool install ruff               全局装 CLI 工具（替代 pipx）
  uvx ruff check .                   临时跑工具（替代 npx 风格）

兼容 1 个：
  uv pip install xxx                 当 pip 用，速度快 10-100 倍
```

---

## 7. 给 Go / Java 工程师的建议

1. **把 `pyproject.toml` 当 `pom.xml` / `go.mod` 看**，是项目唯一真相
2. **永远提交 `uv.lock`**，等价于提交 `go.sum`
3. **永远用 `uv run`**，等价于 `go run` / `mvn exec:java`，**别再手动 `source .venv/bin/activate`**
4. **不要全局 `pip install`**，所有依赖都走 `uv add`，等价于 Go 不会全局乱 `go install`
5. **`uv python pin 3.11`** 写进项目，等价于 Java 项目锁 JDK 21，确保团队一致
6. **CI 里只用 uv**：
   ```yaml
   - run: curl -LsSf https://astral.sh/uv/install.sh | sh
   - run: uv sync --frozen        # 严格按 lock 装，等价 mvn install --offline
   - run: uv run pytest
   ```
   `--frozen` 表示不更新 lock，CI 失败就说明开发同学忘了提交 `uv.lock`

---

## 8. 延伸阅读

- 官方文档：https://docs.astral.sh/uv/
- 安装：`curl -LsSf https://astral.sh/uv/install.sh | sh`
- 对照 Cargo（如果你也写 Rust）：uv 的命令风格几乎照搬 cargo，`uv add` ↔ `cargo add`，`uv run` ↔ `cargo run`，`uv sync` ↔ `cargo build`
