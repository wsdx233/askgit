[English](README_EN.md)

# AskGit：用 LLM 智能探索 Git 仓库

AskGit 是一个命令行工具，它允许你利用大型语言模型（LLM）的力量来通过“询问”了解你的 Git 仓库。通过两种强大的模式——检索增强生成（RAG）和全项目上下文模式，AskGit 可以帮助你理解代码库、查找信息，甚至生成文档。

## 特性

- **智能问答：** 提出关于你的代码库的自然语言问题，并获得 LLM 生成的答案。

- **两种问答模式：**

  - **RAG 模式（推荐）：** 扫描你的仓库以创建知识库（使用 FAISS 向量存储），然后利用检索增强生成（RAG）技术，只将相关的代码片段提供给 LLM，从而提高效率和准确性。

  - **全项目上下文模式：** 将整个 Git 仓库的内容（非二进制文件）作为上下文发送给 LLM。适用于小型项目或需要全面项目概述的场景（可能会消耗大量 Token）。

- **多轮研究模式：** 进入交互式会话，连续提问，LLM 会记住之前的对话历史。

- **文档生成：** 将问答结果直接保存为 Markdown 文档。

- **可配置的 LLM 和 Embedding 提供商：** 支持 OpenAI 和 Google Gemini 模型。

- **Git 感知：** 自动检测 Git 仓库的根目录，并仅处理受 Git 跟踪的文件。

## 安装

### 使用二进制文件（推荐）

从releases中下载对应的二进制文件，放在PATH中

### 使用源代码

1. **克隆仓库（如果适用）或下载 `askgit.py` 文件。**

   ```bash
   git clone https://github.com/your-username/askgit.git # 如果是开源项目
   cd askgit
   ```

2. **安装必要的 Python 依赖：**

   ```bash
   pip install langchain langchain-openai faiss-cpu tiktoken python-dotenv configparser tqdm
   ```

   - 如果计划使用 **Google Gemini** 模型，还需要额外安装：

     ```bash
     pip install langchain-google-genai
     ```

## 配置

首次在 Git 仓库内运行 `askgit.py` 时，它会自动引导你完成配置过程，并在 `.askgit/config.ini` 文件中保存设置。

1. **进入你的 Git 仓库目录：**

   ```bash
   cd /path/to/your/git/repo
   ```

2. **首次运行（例如，运行 `scan` 命令）：**

   ```bash
   askgit scan
   ```

   程序将提示你选择 LLM 和 Embedding 模型的 API 提供商（OpenAI 或 Gemini），并输入相应的 API 密钥和模型名称。

   **`config.ini` 示例：**

   ```ini
   [GENERAL]
   ASSISTANT_API_PROVIDER = openai
   EMBEDDING_API_PROVIDER = openai
   
   [OPENAI]
   API_KEY = sk-YOUR_OPENAI_API_KEY
   BASE_URL = https://api.openai.com/v1
   ASSISTANT_MODEL = gpt-4o
   EMBEDDING_MODEL = text-embedding-3-small
   
   ; [GEMINI]
   ; API_KEY = YOUR_GEMINI_API_KEY
   ; ASSISTANT_MODEL = gemini-1.5-pro-latest
   ; EMBEDDING_MODEL = models/embedding-001
   ```

   - **`ASSISTANT_API_PROVIDER`**: 用于问答和研究模式的 LLM 提供商。

   - **`EMBEDDING_API_PROVIDER`**: 用于 RAG 模式下生成 Embedding 的提供商。

   - **API 密钥：** 请确保你的 API 密钥是有效的。建议使用环境变量来管理敏感信息，而不是直接写入配置文件。

   - **模型名称：** 根据你使用的模型和服务，可能需要调整模型名称（例如，OpenAI 的 `gpt-4o`，Gemini 的 `gemini-1.5-pro-latest`）。

## 使用方法

在你的 Git 仓库根目录下运行 `askgit.py`。

### 1. `scan`：扫描仓库并构建 RAG 知识库

这是使用 RAG 模式问答和研究的**前提**。该命令会遍历你的 Git 仓库中的文件（跳过二进制文件），将文本内容分块并生成嵌入，然后将其存储为 FAISS 向量数据库。

```bash
askgit scan
```

**注意：**

- 这可能需要一些时间，具体取决于你的仓库大小。

- 此命令将创建一个 `.askgit/db/git_kb.faiss` 和 `.askgit/db/git_kb.pkl` 文件来存储知识库。

### 2. `ask`：提出单个问题

向你的仓库提出一个一次性的问题。

- **RAG 模式（默认，推荐）：** 利用之前扫描生成的知识库。

  ```bash
  askgit ask "这个项目如何处理用户认证？"
  ```

- **全项目上下文模式：** 将整个仓库内容（非二进制文件）发送给 LLM。适用于小型项目或需要全局视角的问题。**请注意：** 这可能导致高昂的 Token 消耗和更长的响应时间，并且如果项目过大，可能会超出模型的上下文限制。

  ```bash
  askgit ask --whole "请给我一个关于项目架构的高层概述。"
  ```

### 3. `research`：进入多轮研究模式

进入一个交互式会话，你可以连续提问，LLM 会记住对话历史。

- **RAG 模式（默认，推荐）：**

  ```bash
  askgit research
  ```

  在提示符后输入你的问题。输入 `exit` 或 `quit` 结束会话。

- **全项目上下文模式：**

  ```bash
  askgit research --whole
  ```

  同样，输入 `exit` 或 `quit` 结束会话。请注意此模式下的潜在 Token 消耗。

### 4. `doc`：将问答结果生成为 Markdown 文档

根据你的问题和 LLM 的回答，生成一个 Markdown 文档并保存到 `.askgit/docs/` 目录中。

- **RAG 模式（默认）：**

  ```bash
  askgit doc my_auth_summary "总结一下项目中用户认证的实现细节。"
  ```

  这将在 `.askgit/docs/` 目录下创建一个名为 `my_auth_summary.md` 的文件。

- **全项目上下文模式：**

  ```bash
  askgit doc project_overview_doc --whole "请提供一个详细的项目概述，包括主要模块和它们之间的关系。"
  ```

  这将在 `.askgit/docs/` 目录下创建一个名为 `project_overview_doc.md` 的文件。

## 目录结构

当你在 Git 仓库中使用 AskGit 时，它会在仓库的根目录下创建一个 `.askgit/` 隐藏目录：

```
your_git_repo/
├── .git/
├── .askgit/
│   ├── config.ini      # 存储 API 配置和模型偏好
│   ├── db/             # 存储 RAG 模式的 FAISS 向量数据库
│   │   ├── git_kb.faiss
│   │   └── git_kb.pkl
│   └── docs/           # 存储由 `doc` 命令生成的 Markdown 文档
└── (你的代码文件和目录)
```

## 常见问题

- **`Error: This command must be run inside a git repository.`**请确保你在一个由 Git 管理的目录下运行 `askgit.py`。

- **`Core Langchain/OpenAI libraries not found.`**请运行 `pip install langchain langchain-openai faiss-cpu tiktoken python-dotenv configparser tqdm` 安装所有必要的依赖。

- **RAG 模式没有响应或给出不相关的答案。**

  - 确保你已经运行了 `scan` 命令来构建知识库。

  - 知识库可能过时。如果你的代码库有重大更改，请重新运行 `scan`。

  - 你的问题可能太模糊，导致检索到的上下文不相关。尝试更具体的问题。

  - 检查你的 Embedding 模型配置是否正确。

- **全项目模式遇到 Token 限制错误。**

  - 你的项目可能太大，超过了 LLM 的最大上下文窗口。

  - 在这种情况下，强烈建议使用默认的 RAG 模式，因为它只将相关的代码片段发送给 LLM。

  - 尝试更短的问题，或将 LLM 配置为使用更大的上下文模型（如果可用）。

- **API 密钥或模型配置错误。**

  - 检查 `.askgit/config.ini` 文件中的 API 密钥和模型名称是否正确。

  - 确保你的 API 密钥具有访问所选模型的权限。

## 贡献

欢迎通过 Pull Request 贡献代码或提出 Issue！

---