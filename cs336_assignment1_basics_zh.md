# CS336 Assignment 1 (basics): Building a Transformer LM

非官方中文翻译版。原文版本：Version 26.0.3，CS336 Staff，Spring 2026。

本译文保留核心专业术语的英文写法，并在首次出现时给出中文解释。题目陈述、接口名、文件名、命令、代码片段、数据集名、论文引用和数学符号尽量保持原样，便于和英文 PDF 对照。本文只翻译作业说明，不提供任何题目解答或实现方案。

## 1 Assignment Overview

在本作业中，你将从零开始构建训练标准 Transformer language model (LM) 所需的全部组件，并训练若干模型。

### What you will implement

1. Byte-pair encoding (BPE) tokenizer，见 Section 2。
2. Transformer language model (LM)，见 Section 3。
3. Cross-entropy loss function 和 AdamW optimizer，见 Section 4。
4. Training loop，并支持序列化、加载 model state 和 optimizer state，见 Section 5。

### What you will run

1. 在 TinyStories dataset 上训练 BPE tokenizer。
2. 使用训练好的 tokenizer 对数据集进行编码，将其转换为 integer IDs 序列。
3. 在 TinyStories dataset 上训练 Transformer LM。
4. 使用训练好的 Transformer LM 进行 sample generation，并评估 perplexity。
5. 在 OpenWebText 上训练模型，并将你达到的 perplexities 提交到 leaderboard。

### What you can use

课程期望你从零实现每个组件。尤其是，除了以下内容外，你不能使用 `torch.nn`、`torch.nn.functional` 或 `torch.optim` 中的任何定义：

- `torch.nn.Parameter`
- `torch.nn` 中的 container classes，例如 `Module`、`ModuleList`、`Sequential` 等。
- `torch.optim.Optimizer` base class

你可以使用 PyTorch 中的其他定义。如果你想使用某个 function 或 class，但不确定是否允许，可以在 Slack 上询问。拿不准时，请思考使用它是否会损害本作业的 "from-scratch" 精神。

### Statement on AI tools

AI 可以完全自主地解决作业中的许多部分。这会让学生更难深入参与课程材料并从中学习。

允许使用 AI tools 来回答 high-level conceptual questions，或提供 low-level programming documentation，例如 function signatures 和 library APIs。然而，AI tools 不允许用于实现作业的任何部分。这包括 coding agents，例如 Cursor Agents、Codex、Claude Code，也包括 AI autocomplete，例如 Cursor Tab、GitHub Copilot。使用 AI agent 时，请确保它使用仓库提供的 `AGENTS.md` 文件。使用 chatbots 时，也应包含 prompt。

强烈建议你在完成作业时关闭 IDE 中的 AI autocomplete，例如 Cursor Tab、GitHub Copilot。不过，非 AI autocomplete，例如自动补全函数名，是完全可以的。往届学生反馈，关闭 AI autocomplete 更容易让他们深入投入材料。

完整 AI Policy 请参见课程文档。

### What the code looks like

作业代码和本文档都可以在 GitHub 获得：

`github.com/stanford-cs336/assignment1-basics`

请使用 `git clone` 克隆该仓库。如果有更新，课程方会通知你使用 `git pull` 获取最新内容。

1. `cs336_basics/*`：这是你写代码的地方。注意，这里没有预置代码，你可以完全从零开始。
2. `adapters.py`：这里定义了你的代码必须提供的一组功能。对于每个功能，例如 scaled dot product attention，请填写对应实现，例如 `run_scaled_dot_product_attention`，让它简单调用你自己的代码。注意：你对 `adapters.py` 的修改不应包含实质性逻辑；它只是 glue code。
3. `test_*.py`：这里包含你必须通过的全部测试，例如 `test_scaled_dot_product_attention`，这些测试会调用 `adapters.py` 中定义的 hooks。不要编辑测试文件。

### How to submit

提交时，运行 `make_submission.sh` 构造 submission zip file。如果你有大型数据文件或 checkpoints，不希望它们进入提交包，请把它们加入脚本中的排除列表。

你需要向 Gradescope 提交以下文件：

- `writeup.pdf`：回答所有 written questions。请对答案进行正式排版。
- `code.zip`：包含你写的全部代码。

如果要提交到 leaderboard，请向以下仓库提交 PR：

`github.com/stanford-cs336/assignment1-basics-leaderboard`

详细提交说明见 leaderboard 仓库中的 `README.md`。

### Where to get datasets

本作业使用两个预处理数据集：TinyStories [R. Eldan et al., 2023] 和 OpenWebText [A. Gokaslan et al., 2019]。二者都是单个大型 plaintext file。

如果你正在正式修读本课程，可以在 compute guide 中找到下载数据集的说明。

如果你是在家自学，可以使用 `README.md` 中的命令下载这些文件。

### Low-Resource Tip: Init

在本课程的作业讲义中，我们会持续给出建议，帮助你在 GPU 资源较少或没有 GPU 的情况下完成作业。例如，我们有时会建议缩小 dataset 或 model size，或者解释如何在 Mac integrated GPU 或 CPU 上运行训练代码。你会在蓝色框中看到这些 "low-resource tips"。即使你是 Stanford 正式选课学生并且可以使用课程机器，这些提示也能帮助你更快迭代并节省时间，因此建议阅读。

### Low-Resource Tip: Assignment 1 on Apple Silicon or CPU

使用 staff solution code 时，我们可以在配备 36 GB RAM 的 Apple M4 Max 芯片上训练一个 LM，使其生成相当流畅的文本：在 Metal GPU (MPS) 上不到 5 分钟，在 CPU 上约 30 分钟。如果这些术语对你来说不熟悉，不用担心。你只需要知道：如果你有一台较新的笔记本，并且你的实现正确且高效，你就能训练一个小型 LM，让它生成流畅度不错的简单儿童故事。

作业后面会解释，如果你使用 CPU 或 MPS，应做哪些调整。

## 2 Byte-Pair Encoding (BPE) Tokenizer

在作业第一部分，我们将训练并实现一个 byte-level byte-pair encoding (BPE) tokenizer [R. Sennrich et al., 2016; C. Wang et al., 2019]。具体来说，我们会把任意 Unicode strings 表示成 bytes 序列，并在该 byte sequence 上训练 BPE tokenizer。之后，我们会使用这个 tokenizer 将文本，也就是 string，编码为 tokens，也就是 integer sequence，用于 language modeling。

### 2.1 The Unicode Standard

Unicode 是一种 text encoding standard，它把 characters 映射到 integer code points。截至 Unicode 17.0，即 2025 年 9 月发布的版本，该标准定义了 172 种 scripts 中的 159,801 个 characters。例如，字符 `"s"` 的 code point 是 115，通常写作 `U+0073`，其中 `U+` 是约定前缀，`0073` 是 115 的十六进制表示；字符 `"牛"` 的 code point 是 29275。在 Python 中，可以使用 `ord()` function 将单个 Unicode character 转换为其 integer representation。`chr()` function 则将 integer Unicode code point 转换为包含对应字符的 string。

```python
>>> ord('牛')
29275
>>> chr(29275)
'牛'
```

**Problem (unicode1): Understanding Unicode (1 point)**

(a) `chr(0)` 返回什么 Unicode character？

Deliverable: 一句话回答。

(b) 这个 character 的 string representation，即 `__repr__()`，与它的 printed representation 有什么不同？

Deliverable: 一句话回答。

(c) 当这个 character 出现在文本中时会发生什么？你可以在 Python interpreter 中尝试以下内容，并观察结果是否符合你的预期：

```python
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```

Deliverable: 一句话回答。

### 2.2 Unicode Encodings

Unicode standard 定义的是 characters 到 code points，也就是 integers，之间的映射。但直接在 Unicode code points 上训练 tokenizers 并不现实，因为 vocabulary 会大得难以处理，大约 150K 项，而且非常稀疏，因为许多 characters 很少出现。因此，我们会使用 Unicode encoding，它将 Unicode character 转换成 byte sequence。Unicode 标准本身定义了三种 encodings：UTF-8、UTF-16 和 UTF-32，其中 UTF-8 是互联网上的主流编码，超过 98% 的网页使用 UTF-8。

要把 Unicode string 编码为 UTF-8，可以使用 Python 中的 `encode()` function。要访问 Python `bytes` object 底层的 byte values，可以对它迭代，例如调用 `list()`。最后，可以使用 `decode()` function 将 UTF-8 byte string 解码回 Unicode string。

```python
>>> test_string = "hello! こんにちは!"
>>> utf8_encoded = test_string.encode("utf-8")
>>> print(utf8_encoded)
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
>>> print(type(utf8_encoded))
<class 'bytes'>
>>> # Get the byte values for the encoded string (integers from 0 to 255).
>>> list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]
>>> # One byte does not necessarily correspond to one Unicode character!
>>> print(len(test_string))
13
>>> print(len(utf8_encoded))
23
>>> print(utf8_encoded.decode("utf-8"))
hello! こんにちは!
```

通过将 Unicode code points 转换为 bytes 序列，例如通过 UTF-8 encoding，我们本质上是把一串 code points，也就是具有 159,801 个有效取值的 21-bit integers，转换为一串 byte values，也就是范围在 0 到 255 的 integers。长度为 256 的 byte vocabulary 更容易处理。使用 byte-level tokenization 时，我们不需要担心 out-of-vocabulary tokens，因为任何输入文本都可以表示为 0 到 255 范围内的整数序列。

**Problem (unicode2): Unicode Encodings (3 points)**

(a) 相比 UTF-16 或 UTF-32，为什么更倾向于在 UTF-8 encoded bytes 上训练 tokenizer？比较不同 input strings 在这些 encodings 下的输出可能会有帮助。

Deliverable: 一到两句话回答。

(b) 考虑下面这个不正确的函数，它本意是将 UTF-8 byte string 解码为 Unicode string。为什么这个函数是错误的？请提供一个会产生错误结果的 input byte string 示例。

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```

Deliverable: 给出一个会让 `decode_utf8_bytes_to_str_wrong` 产生错误输出的 input byte string，并用一句话解释为什么该函数错误。

(c) 给出一个无法解码为任何 Unicode character(s) 的 two-byte sequence。

Deliverable: 一个示例，并用一句话解释。

### 2.3 Subword Tokenization

Byte-level tokenization 可以缓解 word-level tokenizers 面临的 out-of-vocabulary 问题，但把文本 tokenized 成 bytes 会导致非常长的输入序列。这会减慢模型训练，因为一个 10 个单词的句子在 word-level language model 中可能只有 10 个 tokens，但在 character-level model 中可能有 50 个或更多 tokens，具体取决于单词长度。处理这些更长的序列会让模型每一步需要更多计算。此外，对 byte sequences 做 language modeling 更困难，因为更长的输入序列会在数据中制造长期依赖。

Subword tokenization 处在 word-level tokenizers 和 byte-level tokenizers 之间。注意，byte-level tokenizer 的 vocabulary 有 256 个 entries，byte values 范围为 0 到 255。Subword tokenizer 用更大的 vocabulary size 换取对 input byte sequence 更好的压缩。例如，如果 byte sequence `b'the'` 在 raw text training data 中经常出现，把它分配为 vocabulary 中的一个 entry，就能把这个 3-token sequence 压缩为一个 token。

我们如何选择要加入 vocabulary 的 subword units？R. Sennrich et al. [3] 提出使用 byte-pair encoding (BPE; P. Gage [5])，这是一种压缩算法，它反复将最频繁的 bytes pair 替换，也就是 "merge"，为一个新的、未使用的 index。注意，该算法向 vocabulary 中加入 subword tokens，是为了最大化输入序列的压缩程度。如果某个 word 在输入文本中出现足够多次，它就会被表示为单个 subword unit。

通过 BPE 构建 vocabulary 的 subword tokenizers 通常称为 BPE tokenizers。在本作业中，我们将实现一个 byte-level BPE tokenizer，其中 vocabulary items 是 bytes 或合并后的 byte sequences。这样既能处理 out-of-vocabulary 问题，又能将输入序列长度控制在可管理范围内。构造 BPE tokenizer vocabulary 的过程被称为 "training" BPE tokenizer。

### 2.4 BPE Tokenizer Training

BPE tokenizer training procedure 包含三个主要步骤。

#### Vocabulary initialization

Tokenizer vocabulary 是 bytestring token 到 integer ID 的一一映射。由于我们训练的是 byte-level BPE tokenizer，初始 vocabulary 就是所有 bytes 的集合。因为可能的 byte values 有 256 个，所以初始 vocabulary size 是 256。

#### Pre-tokenization

一旦有了 vocabulary，原则上你可以统计文本中相邻 bytes 出现的频率，并从最频繁的 bytes pair 开始合并。但是，这在计算上相当昂贵，因为每次 merge 都需要完整遍历 corpus。此外，直接跨整个 corpus 合并 bytes 可能产生只在 punctuation 上不同的 tokens，例如 `dog!` 和 `dog.`。这些 tokens 会得到完全不同的 token IDs，尽管它们很可能具有很高的 semantic similarity，因为它们只在标点上不同。

为避免这个问题，我们对 corpus 进行 pre-tokenize。你可以把它理解为对 corpus 的 coarse-grained tokenization，用来帮助统计字符对出现的频率。例如，word `'text'` 可能是一个出现 10 次的 pre-token。在这种情况下，当我们统计字符 `t` 和 `e` 相邻出现的频率时，可以看到 word `'text'` 中 `t` 和 `e` 相邻，并将计数增加 10，而不必重新遍历整个 corpus。由于我们训练的是 byte-level BPE model，每个 pre-token 会表示为 UTF-8 bytes 序列。

R. Sennrich et al. [3] 的原始 BPE implementation 通过简单地按 whitespace 分割进行 pre-tokenize，即 `s.split(" ")`。这种方法仍然存在于基于 SentencePiece 的 tokenizers 中，例如 Llama 1 和 2 tokenizer。

大多数现代 tokenizers 使用 regex-based pre-tokenizer，这是来自 GPT-2 的实践 [A. Radford et al. [6]]。我们将使用原始 regex 的一个稍微更漂亮的形式，它来自 `github.com/openai/tiktoken/pull/234/files`：

```python
>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

你可以交互式地用这个 pre-tokenizer 分割一些文本，以便更好理解它的行为：

```python
>>> # requires `regex` package
>>> import regex as re
>>> re.findall(PAT, "some text that i'll pre-tokenize")
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

但在代码中使用它时，应使用 `re.finditer`，避免在构建 pre-tokens 到 counts 的 mapping 时存储所有 pre-tokenized words。

#### Compute BPE merges

现在我们已经把 input text 转换为 pre-tokens，并把每个 pre-token 表示为 UTF-8 bytes 序列，就可以计算 BPE merges，也就是训练 BPE tokenizer。在高层次上，BPE algorithm 会反复统计每个 bytes pair，并识别频率最高的一对，例如 ("A", "B")。然后，最频繁 pair ("A", "B") 的每一次出现都会被 merge，也就是替换成一个新 token "AB"。这个新 merged token 会加入 vocabulary；因此，BPE training 后的最终 vocabulary size 等于初始 vocabulary size，本作业中为 256，加上训练期间执行的 BPE merge operations 数量。为了提高 BPE training 的效率，我们不考虑跨 pre-token boundaries 的 pairs。在计算 merges 时，如果 pair frequency 出现并列，请确定性地选择 lexicographically greater pair。例如，如果 pairs `("A", "B")`、`("A", "C")`、`("B", "ZZ")` 和 `("BA", "A")` 都具有最高频率，我们会 merge `("BA", "A")`：

```python
>>> max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
('BA', 'A')
```

#### Special tokens

一些 strings，例如 `<|endoftext|>`，常用于编码 metadata，例如 documents 之间的边界。编码文本时，通常希望把某些 strings 视为 "special tokens"，它们不应被拆成多个 tokens，而是始终保留为单个 token。例如，end-of-sequence string `<|endoftext|>` 应始终保留为单个 token，也就是单个 integer ID，这样我们才知道何时停止 language model 的生成。这些 special tokens 必须加入 vocabulary，因此它们有对应的固定 token ID。

R. Sennrich et al. [3] 的 Algorithm 1 包含一个低效的 BPE tokenizer training implementation，本质上遵循我们上面概述的步骤。作为第一个练习，实现并测试这个函数有助于检查你的理解。

**Example (bpe_example): BPE training example**

这里是 R. Sennrich et al. [3] 中的一个风格化示例。考虑一个 corpus，其中包含以下文本：

```text
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
```

并且 vocabulary 有一个 special token `<|endoftext|>`。

**Vocabulary**

我们用 special token `<|endoftext|>` 和 256 个 byte values 初始化 vocabulary。

**Pre-tokenization**

为简化说明并聚焦 merge procedure，这个例子假设 pre-tokenization 只是按 whitespace 分割。当我们进行 pre-tokenize 并计数后，得到 frequency table：

```text
{low: 5, lower: 2, widest: 3, newest: 6}
```

把它表示为 `dict[tuple[bytes, ...], int]` 会很方便，例如 `{(l,o,w): 5, ...}`。注意，即使是单个 byte，在 Python 中也是 `bytes` object。Python 中没有表示单个 byte 的 `byte` type，就像没有表示单个 character 的 `char` type 一样。

**Merges**

我们首先查看每个连续 bytes pair，并加总包含这些 pair 的 words 的频率：`{lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}`。Pairs `('e', 's')` 和 `('s', 't')` 并列，因此我们取 lexicographically greater pair，即 `('s', 't')`。接着 merge pre-tokens，得到 `{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6}`。

第二轮中，我们看到 `(e, st)` 是最常见的 pair，count 为 9，因此 merge 后得到 `{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,est): 3, (n,e,w,est): 6}`。继续这个过程，最终得到的 merges sequence 是 `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e', 'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r']`。

如果我们取 6 次 merges，即 `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']`，那么 vocabulary elements 将是 `[<|endoftext|>, [...256 BYTE CHARS], st, est, ow, low, west, ne]`。

使用这个 vocabulary 和 merges 集合，word `newest` 会被 tokenize 为 `[ne, west]`。

### 2.5 Experimenting with BPE Tokenizer Training

现在让我们在 TinyStories dataset 上训练 byte-level BPE tokenizer。获取或下载 dataset 的说明见 Section 1。开始之前，建议先查看 TinyStories dataset，了解数据内容。

#### Parallelizing pre-tokenization

你会发现一个主要瓶颈是 pre-tokenization step。可以使用 Python 内置库 `multiprocessing` 并行化代码来加速 pre-tokenization。具体来说，我们建议在 parallel pre-tokenization implementation 中切分 corpus，同时确保 chunk boundaries 出现在 special token 的起始位置。你可以直接使用以下链接中的 starter code 来获得 chunk boundaries，然后用这些 boundaries 将工作分发给多个 processes：

`https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py`

这种 chunking 始终有效，因为我们从不希望跨 document boundaries 进行 merge。就本作业而言，你总是可以按这种方式分割。不必担心收到一个非常大的、不包含 `<|endoftext|>` 的 corpus 这种 edge case。

#### Removing special tokens before pre-tokenization

在使用 regex pattern 运行 pre-tokenization 之前，也就是使用 `re.finditer` 之前，应从 corpus，或并行实现中的 chunk，中移除所有 special tokens。请确保按 special tokens 进行 split，使得不会跨越它们分隔的文本进行 merge。例如，如果 corpus 或 chunk 类似 `[Doc 1]<|endoftext|>[Doc 2]`，应按 special token `<|endoftext|>` split，并分别对 `[Doc 1]` 和 `[Doc 2]` 进行 pre-tokenize，这样就不会跨 document boundary 发生 merge。换言之，special tokens 在训练期间定义 hard segmentation boundaries，但它们本身不应参与 merge counts。可以使用 `re.split` 并以 `"|".join(special_tokens)` 作为 delimiter 来完成此事，但要谨慎使用 `re.escape`，因为 `|` 可能出现在 special tokens 中。测试 `test_train_bpe_special_tokens` 会检查这一点。

#### Optimizing the merging step

上面的风格化例子中，naive implementation 的 BPE training 很慢，因为每次 merge 都会遍历所有 byte pairs，以识别最频繁的 pair。然而，每次 merge 后，只有与被 merge pair 重叠的 pair counts 会发生变化。因此，可以通过索引所有 pairs 的 counts 并增量更新这些 counts 来加速 BPE training，而不是显式遍历每个 bytes pair 来统计 pair frequencies。这个 caching procedure 可以带来显著加速。不过需要注意，BPE training 的 merging 部分在 Python 中不可并行化。

#### Low-Resource Tip: Profiling

你应该使用 profiling tools，例如 `cProfile` 或 `py-spy`，来识别 implementation 中的瓶颈，并重点优化这些部分。

#### Low-Resource Tip: "Downscaling"

不要一开始就直接在完整 TinyStories dataset 上训练 tokenizer。我们建议你先在小数据子集，即 "debug dataset" 上训练。例如，可以改用 TinyStories validation set 训练 tokenizer，它有 22K documents，而不是 2.12M。这个建议体现了一种通用策略：尽可能 downscale，以加速开发。例如，使用更小的 datasets、更小的 model sizes 等。选择 debug dataset 或 hyperparameter config 的规模需要仔细考虑：你希望 debug set 足够大，以呈现与完整配置相同的瓶颈，这样你做的优化才能泛化；但也不能大到运行时间过长。

**Problem (train_bpe): BPE Tokenizer Training (15 points)**

Deliverable: 编写一个函数，给定 input text file 的路径，训练一个 byte-level BPE tokenizer。你的 BPE training function 至少应处理以下 input parameters：

Input:

- `input_path: str`：包含 BPE tokenizer training data 的 text file 路径。
- `vocab_size: int`：一个 positive integer，定义最终 maximum vocabulary size，包括初始 byte vocabulary、merge 产生的 vocabulary items，以及任何 special tokens。
- `special_tokens: list[str]`：要加入 vocabulary 的 strings 列表。训练期间，将它们视为 hard boundaries，防止跨越其 span 发生 merges，但在计算 merge statistics 时不包括它们。

你的 BPE training function 应返回得到的 vocabulary 和 merges：

Output:

- `vocab: dict[int, bytes]`：tokenizer vocabulary，即从 int，表示 vocabulary 中的 token ID，到 bytes，表示 token bytes，的 mapping。
- `merges: list[tuple[bytes, bytes]]`：训练产生的 BPE merges 列表。每个 list item 是 bytes tuple `(<token1>, <token2>)`，表示 `<token1>` 与 `<token2>` 被 merge。Merges 应按创建顺序排列。

为了用提供的 tests 测试你的 BPE training function，首先需要实现 `[adapters.run_train_bpe]` 中的 test adapter。然后运行 `uv run pytest tests/test_train_bpe.py`。你的实现应能通过所有测试。可选地，你也可以使用某种 systems language 实现训练方法的关键部分，例如 C++，可考虑 `cppyy` 或 `nanobind`，或 Rust，使用 PyO3。这可能需要大量时间。如果你这样做，请注意哪些操作需要复制、哪些操作可以直接从 Python memory 中读取，并留下 build instructions，或确保它能仅通过 `pyproject.toml` 构建。还要注意，GPT-2 regex 在大多数 regex engines 中支持不好，而且多数情况下会太慢。我们验证过 Oniguruma 速度合理且支持 negative lookahead，但 Python 的 `regex` package 如果有差别，甚至更快。

**Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)**

(a) 在 TinyStories dataset 上训练 byte-level BPE tokenizer，maximum vocabulary size 为 10,000。确保将 TinyStories 的 `<|endoftext|>` special token 加入 vocabulary。将得到的 vocabulary 和 merges 序列化到磁盘以便进一步检查。训练花费了多少时间和内存？Vocabulary 中最长的 token 是什么？它是否合理？

Resource requirements: <= 30 minutes (no GPUs), <= 30 GB RAM

Hint: 使用 multiprocessing 做 pre-tokenization，并利用以下两个事实，你应能在 2 分钟以内完成 BPE training：

1. 数据文件中的 `<|endoftext|>` token 分隔 documents。
2. 在应用 BPE merges 之前，`<|endoftext|>` token 会作为 special case 处理。

Deliverable: 一到两句话回答。

(b) Profile 你的代码。Tokenizer training process 中哪一部分耗时最多？

Deliverable: 一到两句话回答。

接下来，我们会尝试在 OpenWebText dataset 上训练 byte-level BPE tokenizer。和之前一样，建议先查看 dataset，了解其内容。

**Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)**

(a) 在 OpenWebText dataset 上训练 byte-level BPE tokenizer，maximum vocabulary size 为 32,000。将得到的 vocabulary 和 merges 序列化到磁盘以便进一步检查。Vocabulary 中最长的 token 是什么？它是否合理？

Resource requirements: <= 12 hours (no GPUs), <= 100 GB RAM

Deliverable: 一到两句话回答。

(b) 比较并对照在 TinyStories 与 OpenWebText 上训练得到的 tokenizers。

Deliverable: 一到两句话回答。

### 2.6 BPE Tokenizer: Encoding and Decoding

上一部分中，我们实现了一个函数，它在输入文本上训练 BPE tokenizer，并得到 tokenizer vocabulary 和 BPE merges 列表。现在，我们将实现一个 BPE tokenizer，它加载给定的 vocabulary 和 merges list，并用它们在 text 和 token IDs 之间进行 encode 与 decode。

#### 2.6.1 Encoding text

BPE encoding text 的过程与训练 BPE vocabulary 的方式相似。主要有几个步骤。

Step 1: Pre-tokenize。我们首先对 sequence 进行 pre-tokenize，并将每个 pre-token 表示为 UTF-8 bytes 序列，就像 BPE training 中一样。我们会在每个 pre-token 内部将这些 bytes merge 成 vocabulary elements，并独立处理每个 pre-token，不跨 pre-token boundaries 进行 merges。

Step 2: Apply the merges。接着，我们取 BPE training 中创建的 vocabulary element merges sequence，并按创建顺序将其应用到 pre-tokens。

**Example (bpe_encoding): BPE encoding example**

例如，假设 input string 是 `'the cat ate'`，vocabulary 是 `{0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}`，learned merges 是 `[(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]`。首先，pre-tokenizer 会将该 string 分割为 `['the', ' cat', ' ate']`。然后，我们查看每个 pre-token 并应用 BPE merges。

第一个 pre-token `'the'` 最初表示为 `[b't', b'h', b'e']`。查看 merges list，我们识别到第一个适用的 merge 是 `(b't', b'h')`，并用它将 pre-token 转换为 `[b'th', b'e']`。然后回到 merges list，识别下一个适用的 merge 是 `(b'th', b'e')`，它将 pre-token 转换为 `[b'the']`。最后，再次查看 merges list，发现没有更多 merge 适用于该 string，因为整个 pre-token 已经合并成单个 token，于是 BPE merges 应用完成。对应的 integer sequence 是 `[9]`。

对其余 pre-tokens 重复这个过程，可以看到 pre-token `' cat'` 应用 BPE merges 后表示为 `[b' c', b'a', b't']`，对应 integer sequence `[7, 1, 5]`。最后一个 pre-token `' ate'` 是 `[b' at', b'e']`，对应 integer sequence `[10, 3]`。因此，对 input string 编码的最终结果是 `[9, 7, 1, 5, 10, 3]`。

#### Special tokens

你的 tokenizer 在 encoding text 时应能够正确处理用户定义的 special tokens，这些 special tokens 在构造 tokenizer 时提供。

#### Memory considerations

假设我们想 tokenize 一个无法装入内存的大型 text file。为了高效 tokenize 这个大文件，或任何其他 data stream，需要将其拆成可管理的 chunks，并逐个处理 chunk，使 memory complexity 保持常数，而不是随文本大小线性增长。这样做时，需要确保 token 不跨越 chunk boundaries，否则 tokenization 结果会不同于一次性将整个 sequence 放入内存进行 tokenization 的 naive method。

#### 2.6.2 Decoding text

要将 integer token IDs 序列解码回 raw text，可以简单地查找每个 ID 在 vocabulary 中对应的 entry，即 byte sequence，将它们拼接起来，然后把 bytes decode 为 Unicode string。注意，输入 IDs 不保证能映射到合法 Unicode strings，因为用户可以输入任意 integer IDs 序列。如果输入 token IDs 不能产生有效 Unicode string，应使用官方 Unicode replacement character `U+FFFD` 替换 malformed bytes。`bytes.decode` 的 `errors` argument 控制 Unicode decoding errors 的处理方式，使用 `errors='replace'` 会自动用 replacement marker 替换 malformed data。

**Problem (tokenizer): Implementing the tokenizer (15 points)**

Deliverable: 实现一个 `Tokenizer` class，给定 vocabulary 和 merges list，能将 text encode 为 integer IDs，并将 integer IDs decode 为 text。你的 tokenizer 还应支持用户提供的 special tokens，如果它们尚不在 vocabulary 中，则追加到 vocabulary。推荐以下接口：

```python
def __init__(self, vocab, merges, special_tokens=None)
```

从给定 vocabulary、merges list，以及可选的 special tokens 构造 tokenizer。该函数应接受以下参数：

- `vocab: dict[int, bytes]`
- `merges: list[tuple[bytes, bytes]]`
- `special_tokens: list[str] | None = None`

```python
def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)
```

Class method，从序列化的 vocabulary 和 merges list，即与你的 BPE training code 输出相同的格式，以及可选的 special tokens，构造并返回一个 `Tokenizer`。该 method 应接受以下额外参数：

- `vocab_filepath: str`
- `merges_filepath: str`
- `special_tokens: list[str] | None = None`

```python
def encode(self, text: str) -> list[int]
```

将 input text encode 为 token IDs 序列。

```python
def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]
```

给定 strings iterable，例如 Python file handle，返回一个 generator，lazy 地 yield token IDs。这是对无法直接装入内存的大文件进行 memory-efficient tokenization 所必需的。

```python
def decode(self, ids: list[int]) -> str
```

将 token IDs 序列 decode 为 text。

为了使用提供的 tests 测试你的 `Tokenizer`，首先需要实现 `[adapters.get_tokenizer]` 中的 test adapter。然后运行 `uv run pytest tests/test_tokenizer.py`。你的实现应能通过所有测试。

### 2.7 Experiments

**Problem (tokenizer_experiments): Experiments with tokenizers (4 points)**

(a) 从 TinyStories 和 OpenWebText 中各 sample 10 documents。使用你之前训练好的 TinyStories 和 OpenWebText tokenizers，它们的 vocabulary size 分别为 10K 和 32K，将这些 sampled documents encode 为 integer IDs。每个 tokenizer 的 compression ratio，即 bytes/token，是多少？

Deliverable: 一到两句话回答。

(b) 如果用 TinyStories tokenizer 来 tokenize 你的 OpenWebText sample，会发生什么？比较 compression ratio，或定性描述发生了什么。

Deliverable: 一到两句话回答。

(c) 估计你的 tokenizer throughput，例如 bytes/second。Tokenize Pile dataset，包含 825GB text，需要多久？

Deliverable: 一到两句话回答。

(d) 使用你的 TinyStories 和 OpenWebText tokenizers，将各自的 training 和 development datasets encode 为 integer token IDs 序列。后续我们会用它训练 language model。我们建议将 token IDs 序列化为 datatype 为 `uint16` 的 NumPy array。为什么 `uint16` 是合适的选择？

Deliverable: 一到两句话回答。

## 3 Transformer Language Model Architecture

Language model 的输入是 batched integer token IDs 序列，即 shape 为 `(batch_size, sequence_length)` 的 `torch.Tensor`；输出是 vocabulary 上的 batched normalized probability distribution，即 shape 为 `(batch_size, sequence_length, vocab_size)` 的 PyTorch Tensor，其中每个 input token 对应的预测分布是下一个词的分布。训练 language model 时，我们使用这些 next-word predictions 来计算实际 next word 与 predicted next word 之间的 cross-entropy loss。使用 language model 进行 inference 生成文本时，我们取最后一个 time step，即 sequence 中最后一项，的 predicted next-word distribution 来生成 sequence 中的 next token，例如取 probability 最大的 token，或从分布中 sampling，然后把生成的 token 加回 input sequence，并重复该过程。

在本部分作业中，你将从零构建这个 Transformer language model。我们会先给出模型的 high-level description，然后逐步详述各个组件。

### 3.1 Transformer LM

给定 token IDs 序列，Transformer language model 使用 input embedding 将 token IDs 转换为 dense vectors，将 embedded tokens 传入 `num_layers` 个 Transformer blocks，然后应用 learned linear projection，即 "output embedding" 或 "LM head"，产生 predicted next-token logits。Figure 1 展示了示意图。

Figure 1: Transformer language model 概览。

Figure 2: Pre-norm Transformer block。

#### Token Embeddings

在第一步中，Transformer 将 batched token IDs 序列 embedding 为一串 vectors，这些 vectors 包含 token identity 信息。

更具体地说，给定 token IDs 序列，Transformer language model 使用 token embedding layer 产生 vector sequence。每个 embedding layer 接收 shape 为 `(batch_size, sequence_length)` 的 integer tensor，并产生 shape 为 `(batch_size, sequence_length, d_model)` 的 vector sequence。

#### Pre-norm Transformer Block

Embedding 之后，activations 会经过若干结构相同的 neural net layers 处理。标准 decoder-only Transformer language model 由 `num_layers` 个相同 layers 组成，通常称为 Transformer "blocks"。每个 Transformer block 接收 shape 为 `(batch_size, sequence_length, d_model)` 的输入，并返回 shape 相同的输出。每个 block 通过 self-attention 聚合序列中的信息，并通过 feed-forward layers 进行非线性转换。

经过 `num_layers` 个 Transformer blocks 后，我们会取最终 activations，并将其转换为 vocabulary 上的分布。

我们将实现 "pre-norm" Transformer block，详见 Section 3.4。它还需要在最后一个 Transformer block 之后使用 layer normalization，以确保输出尺度合适。

在该 normalization 之后，我们会使用标准 learned linear transformation，将 Transformer blocks 的输出转换为 predicted next-token logits，参见 A. Radford et al. [7] equation 2。

### 3.2 Remark: Batching, Einsum and Efficient Computation

在整个 Transformer 中，我们会对许多 batch-like inputs 应用相同计算。以下是几个例子：

- Batch elements：我们对每个 batch element 应用相同的 Transformer forward operation。
- Sequence length：像 RMSNorm 和 feed-forward 这样的 "position-wise" operations 会对 sequence 的每个 position 以相同方式运行。
- Attention heads：在 "multi-headed" attention operation 中，attention operation 会跨 attention heads batched 执行。

有一种符合工程直觉的方式来执行这些操作会很有用：既能充分利用 GPU，又容易阅读和理解。许多 PyTorch operations 可以在 tensor 起始处接受额外的 "batch-like" dimensions，并在这些 dimensions 上高效重复或 broadcast 该 operation。

例如，假设我们正在做 position-wise, batched operation。我们有一个 shape 为 `(batch_size, sequence_length, d_model)` 的 data tensor `D`，并希望与 shape 为 `(d_model, d_model)` 的 matrix `A` 做 batched vector-matrix multiply。在这种情况下，`D @ A` 会执行 batched matrix multiply，这是 PyTorch 中的高效 primitive，其中 `(batch_size, sequence_length)` dimensions 被视为 batch dimensions。

因此，假设你的 functions 可能被传入额外的 batch-like dimensions，并将这些 dimensions 放在 PyTorch shape 的开头，会很有帮助。为了让 tensors 能按这种方式 batch，可能需要使用许多 `view`、`reshape` 和 `transpose` 步骤来调整 shape。这可能很麻烦，也常常让代码难以读懂：你很难看出代码在做什么，tensors 的 shapes 又是什么。

更符合人类阅读的选择是在 `torch.einsum` 中使用 einsum notation，或者使用 framework-agnostic libraries，例如 `einops` 或 `einx`。两个关键 ops 是 `einsum` 和 `rearrange`：前者可以对 input tensors 的任意 dimensions 做 tensor contractions，后者可以 reorder、concatenate 和 split 任意 dimensions。事实证明，几乎所有 machine learning operations 都是 dimension juggling 和 tensor contraction 的某种组合，偶尔加上通常是 pointwise 的 nonlinear function。因此，使用 einsum notation 可以让大量代码更可读、更灵活。

我们强烈建议在本课程中学习并使用 einsum notation。此前没有接触过 einsum notation 的学生应使用 `einops`，已经熟悉 `einops` 的学生则应学习更 general 的 `einx`。这两个 packages 已经安装在我们提供的环境中。

以下示例展示了 einsum notation 的用法。它们是 `einops` documentation 的补充，你应先阅读相关文档。

**Example (einstein_example1): Batched matrix multiplication with einops.einsum**

```python
import torch
from einops import rearrange, einsum

## Basic implementation
Y = D @ A.T
# Hard to tell the input and output shapes and what they mean.
# What shapes can D and A have, and do any of these have unexpected behavior?

## Einsum is self-documenting and robust
#                          D                A     ->          Y
Y = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")

## Or, a batched version where D can have any leading dimensions but A is constrained.
Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")
```

**Example (einstein_example2): Broadcasted operations with einops.rearrange**

我们有一批 images，对于每张 image，希望基于某个 scaling factor 生成 10 个变暗版本：

```python
images = torch.randn(64, 128, 128, 3)  # (batch, height, width, channel)
dim_by = torch.linspace(start=0.0, end=1.0, steps=10)

## Reshape and multiply
dim_value = rearrange(dim_by,    "dim_value              -> 1 dim_value 1 1 1")
images_rearr = rearrange(images, "b height width channel -> b 1 height width channel")
dimmed_images = images_rearr * dim_value

## Or in one go:
dimmed_images = einsum(
    images, dim_by,
    "batch height width channel, dim_value -> batch dim_value height width channel"
)
```

**Example (einstein_example3): Pixel mixing with einops.rearrange**

假设我们有一批 images，表示为 shape `(batch, height, width, channel)` 的 tensor，并希望对 image 的所有 pixels 执行 linear transformation，但该 transformation 应对每个 channel 独立发生。我们的 linear transformation 表示为 shape `(height * width, height * width)` 的 matrix `B`。

```python
channels_last = torch.randn(64, 32, 32, 3)  # (batch, height, width, channel)
B = torch.randn(32*32, 32*32)

## Rearrange an image tensor for mixing across all pixels
channels_last_flat = channels_last.view(
    -1, channels_last.size(1) * channels_last.size(2), channels_last.size(3)
)
channels_first_flat = channels_last_flat.transpose(1, 2)
channels_first_flat_transformed = channels_first_flat @ B.T
channels_last_flat_transformed = channels_first_flat_transformed.transpose(1, 2)
channels_last_transformed = channels_last_flat_transformed.view(*channels_last.shape)
```

改用 `einops`：

```python
height = width = 32

## Rearrange replaces clunky torch view + transpose
channels_first = rearrange(
    channels_last,
    "batch height width channel -> batch channel (height width)"
)
channels_first_transformed = einsum(
    channels_first, B,
    "batch channel pixel_in, pixel_out pixel_in -> batch channel pixel_out"
)
channels_last_transformed = rearrange(
    channels_first_transformed,
    "batch channel (height width) -> batch height width channel",
    height=height, width=width
)
```

或者，如果你想更进一步，可以用 `einx.dot`，即 `einops.einsum` 的 `einx` 对应物，一步完成：

```python
height = width = 32
channels_last_transformed = einx.dot(
    "batch row_in col_in channel, (row_out col_out) (row_in col_in)"
    "-> batch row_out col_out channel",
    channels_last, B,
    col_in=width, col_out=width
)
```

第一种 implementation 可以通过在前后放注释来说明 input 和 output shapes，从而有所改进，但这很笨重，而且容易出错。使用 einsum notation 时，documentation is implementation。

Einsum notation 可以处理任意 input batching dimensions，同时还有一个关键好处：它是 self-documenting。使用 einsum notation 的代码能更清晰地显示 input 和 output tensors 的相关 shapes。对于剩下的 tensors，可以考虑使用 Tensor type hints，例如 `jaxtyping` library，尽管它并不只用于 JAX。

我们会在 assignment 2 中进一步讨论使用 einsum notation 的 performance implications。现在只需知道，它几乎总是比替代方案更好。

#### 3.2.1 Mathematical Notation and Memory Ordering

许多 machine learning papers 在 notation 中使用 row vectors，这与 NumPy 和 PyTorch 默认使用的 row-major memory ordering 很契合。使用 row vectors 时，linear transformation 写作：

```text
y = x W^T
```

其中 row-major `W in R^{d_out x d_in}`，row-vector `x in R^{1 x d_in}`。注意，这让我们可以通过增加 `x` 的最外层 dimension 来 batch inputs，也就是说，可以用 matrix input `X in R^{batch x d_in}` 替代 vector input `x`。

在线性代数中，更常见的是使用 column vectors，其中 linear transformations 写作：

```text
y = W x
```

给定 row-major `W in R^{d_out x d_in}` 和 column-vector `x in R^{d_in}`。在这种设置中，为了 batch input，batch dimension 必须位于 `x` 的最后，因此需要用 matrix `X~ in R^{d_in x batch}` 替代 `x`。

在本作业的数学 notation 中，我们大多使用 column vectors，因为数学通常遵循这种 notation。你需要记住，如果想使用普通 matrix multiplication notation，由于 PyTorch 使用 row-major memory ordering，你必须像 Equation 1 中的 row vector convention 那样使用 transpose 来应用 matrices。如果你使用 `einsum` 做 linear algebra operations，只要正确标注 axes，这就不会成为问题。顺便说，Matlab、Julia 和 Fortran 等其他语言或 linear algebra packages 使用 column-major memory ordering，这意味着 batching dimensions 位于最后，但 Python 及相关 packages 采用了 C 标准的 row-major ordering。

### 3.3 Basic Building Blocks: Linear and Embedding Modules

#### 3.3.1 Parameter Initialization

有效训练 neural networks 通常需要谨慎初始化 model parameters；糟糕的 initialization 可能导致 undesirable behavior，例如 vanishing gradients 或 exploding gradients。Pre-norm transformers 对 initializations 异常稳健，但 initialization 仍会显著影响 training speed 和 convergence。由于本作业已经很长，我们会把细节留到 assignment 3，这里给出一些 approximate initializations，通常应该能良好工作。现在，请使用：

- Linear weights: `N(mu = 0, sigma^2 = 2 / (d_in + d_out))`，截断到 `[-3 sigma, 3 sigma]`。
- Embedding: `N(mu = 0, sigma^2 = 1)`，截断到 `[-3, 3]`。
- RMSNorm: 全 1。

你应使用 `torch.nn.init.trunc_normal_` 初始化 truncated normal weights。

#### 3.3.2 Linear Module

Linear layers 是 Transformers 和 neural nets 的基础 building block。首先，你将实现自己的 `Linear` class，它继承 `torch.nn.Module` 并执行 linear transformation：

```text
y = W x
```

注意，遵循大多数现代 LLMs，我们不包含 bias term。

**Problem (linear): Implementing the linear module (1 point)**

Deliverable: 实现一个继承 `torch.nn.Module` 的 `Linear` class，并执行 linear transformation。你的 implementation 应遵循 PyTorch 内置 `nn.Linear` module 的接口，但没有 bias argument 或 parameter。推荐以下接口：

```python
def __init__(self, in_features, out_features, device=None, dtype=None)
```

构造 linear transformation module。该函数应接受以下参数：

- `in_features: int`：input 的 final dimension。
- `out_features: int`：output 的 final dimension。
- `device: torch.device | None = None`：存放 parameters 的 device。
- `dtype: torch.dtype | None = None`：parameters 的 data type。

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

将 linear transformation 应用于 input。

请确保：

- subclass `nn.Module`
- 调用 superclass constructor
- 将你的 parameter 构造并存储为 `W`，不是 `W^T`，并将其放入 `nn.Parameter`
- 当然，不要使用 `nn.Linear` 或 `nn.functional.linear`

Initialization 请使用上面给出的设置，并用 `torch.nn.init.trunc_normal_` 初始化 weights。

为了测试你的 Linear module，请实现 `[adapters.run_linear]` 中的 test adapter。该 adapter 应将给定 weights 加载到你的 Linear module 中。你可以为此使用 `Module.load_state_dict`。然后运行 `uv run pytest -k test_linear`。

#### 3.3.3 Embedding Module

如上所述，Transformer 的第一层是 embedding layer，它将 integer token IDs 映射到 dimension 为 `d_model` 的 vector space。我们将实现一个自定义 `Embedding` class，继承 `torch.nn.Module`，因此不应使用 `nn.Embedding`。`forward` method 应通过使用 shape 为 `(batch_size, sequence_length)` 的 `torch.LongTensor` token IDs 索引 shape 为 `(vocab_size, d_model)` 的 embedding matrix，为每个 token ID 选择 embedding vector。

**Problem (embedding): Implement the embedding module (1 point)**

Deliverable: 实现继承 `torch.nn.Module` 的 `Embedding` class，并执行 embedding lookup。你的 implementation 应遵循 PyTorch 内置 `nn.Embedding` module 的接口。推荐以下接口：

```python
def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None)
```

构造 embedding module。该函数应接受以下参数：

- `num_embeddings: int`：vocabulary size。
- `embedding_dim: int`：embedding vectors 的 dimension，即 `d_model`。
- `device: torch.device | None = None`：存放 parameters 的 device。
- `dtype: torch.dtype | None = None`：parameters 的 data type。

```python
def forward(self, token_ids: torch.Tensor) -> torch.Tensor
```

查找给定 token IDs 的 embedding vectors。

请确保：

- subclass `nn.Module`
- 调用 superclass constructor
- 将 embedding matrix 初始化为 `nn.Parameter`
- 存储 embedding matrix 时使 `d_model` 位于 final dimension
- 当然，不要使用 `nn.Embedding` 或 `nn.functional.embedding`

同样，使用上面的 initialization 设置，并用 `torch.nn.init.trunc_normal_` 初始化 weights。

为了测试你的 implementation，请实现 `[adapters.run_embedding]` 中的 test adapter。然后运行 `uv run pytest -k test_embedding`。

### 3.4 Pre-Norm Transformer Block

每个 Transformer block 有两个 sub-layers：multi-head self-attention mechanism 和 position-wise feed-forward network，参见 A. Vaswani et al. [2017] Section 3.1。

在原始 Transformer paper 中，模型在每个 sub-layer 周围使用 residual connection，随后进行 layer normalization。这种 architecture 通常称为 "post-norm" Transformer，因为 layer normalization 应用于 sub-layer output。然而，多项工作发现，将 layer normalization 从每个 sub-layer 的 output 移到每个 sub-layer 的 input，并在最终 Transformer block 之后增加一个 layer normalization，可以提升 Transformer training stability [T. Q. Nguyen et al., 2019; R. Xiong et al., 2020]。Figure 2 展示了这种 "pre-norm" Transformer block。每个 Transformer block sub-layer 的 output 随后通过 residual connection 加到 sub-layer input 上 [A. Vaswani et al. [8], Section 5.4]。对 pre-norm 的一种直觉是：从 input embeddings 到 Transformer 最终输出之间存在一条干净的、没有任何 normalization 的 "residual stream"，据称这能改善 gradient flow。Pre-norm Transformer 现在是当今 language models 使用的标准，例如 GPT-3、LLaMA、PaLM 等，因此我们将实现这一变体。我们将逐个介绍并实现 pre-norm Transformer block 的组件。

#### 3.4.1 Root Mean Square Layer Normalization

A. Vaswani et al. [8] 的原始 Transformer implementation 使用 layer normalization [J. L. Ba et al., 2016] 来 normalize activations。遵循 H. Touvron et al. [12]，我们将使用 root mean square layer normalization，即 RMSNorm [B. Zhang et al. [13], equation 4] 作为 layer normalization。给定 activations vector `a in R^{d_model}`，RMSNorm 会按如下方式 rescale 每个 activation `a_i`：

```text
RMSNorm(a_i) = a_i / RMS(a) * g_i
RMS(a) = sqrt((1 / d_model) * sum_i a_i^2 + epsilon)
```

其中 `g_i` 是 learnable "gain" parameter，总共有 `d_model` 个这样的 parameters；`epsilon` 是 hyperparameter，通常固定为 `1e-5`。

在对 input 求平方前，应将其 upcast 到 `torch.float32`，以防 overflow。整体上，你的 `forward` method 应类似：

```python
in_dtype = x.dtype
x = x.to(torch.float32)
# Your code here performing RMSNorm
...
result = ...
# Return the result in the original dtype
return result.to(in_dtype)
```

**Problem (rmsnorm): Root Mean Square Layer Normalization (1 point)**

Deliverable: 将 RMSNorm 实现为 `torch.nn.Module`。推荐以下接口：

```python
def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)
```

构造 RMSNorm module。该函数应接受以下参数：

- `d_model: int`：model 的 hidden dimension。
- `eps: float = 1e-5`：用于 numerical stability 的 epsilon value。
- `device: torch.device | None = None`：存放 parameters 的 device。
- `dtype: torch.dtype | None = None`：parameters 的 data type。

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

处理 shape 为 `(batch_size, sequence_length, d_model)` 的 input tensor，并返回相同 shape 的 tensor。

Note: 如上所述，请记得在执行 normalization 前将 input upcast 到 `torch.float32`，之后再 downcast 到 original dtype。

为了测试你的 implementation，请实现 `[adapters.run_rmsnorm]` 中的 test adapter。然后运行 `uv run pytest -k test_rmsnorm`。

#### 3.4.2 Position-Wise Feed-Forward Network

Figure 3 比较了 SiLU，也称为 Swish，和 ReLU activation functions。

在原始 Transformer paper [A. Vaswani et al. [8], Section 3.3] 中，Transformer feed-forward network 包含两个 linear transformations，中间使用 ReLU activation，`ReLU(x) = max(0, x)`。在该原始 architecture 中，inner feed-forward layer 的 dimensionality 通常是 input dimensionality 的 4 倍。

然而，现代 language models 与该原始设计相比通常包含两个主要变化：使用另一种 activation function，并采用 gating mechanism。具体来说，我们将实现 Llama 3 [A. Grattafiori et al., 2024] 和 Qwen 2.5 [A. Yang et al., 2024] 等 LLMs 采用的 "SwiGLU" activation function。它将 SiLU，常称为 Swish，activation 与一种称为 Gated Linear Unit (GLU) 的 gating mechanism 结合起来。我们还会省略 linear layers 中有时使用的 bias terms，这遵循 PaLM [A. Chowdhery et al., 2022] 和 LLaMA [H. Touvron et al., 2023] 以来大多数现代 LLMs 的做法。

SiLU 或 Swish activation function [D. Hendrycks et al., 2016; S. Elfwing et al., 2017] 定义如下：

```text
SiLU(x) = x * sigmoid(x) = x / (1 + e^{-x})
```

从 Figure 3 可以看到，SiLU activation function 类似 ReLU activation function，但在 0 附近平滑。

Gated Linear Units (GLUs) 最初由 Y. N. Dauphin et al. [19] 定义为：一个 linear transformation 经过 sigmoid function 后，与另一个 linear transformation 做 element-wise product：

```text
GLU(x, W1, W2) = sigmoid(W1 x) * W2 x
```

其中 `*` 表示 element-wise multiplication。Gated Linear Units 被认为能 "reduce the vanishing gradient problem for deep architectures by providing a linear path for the gradients while retaining non-linear capabilities"。

将 SiLU/Swish 和 GLU 放在一起，就得到 SwiGLU，我们会在 feed-forward networks 中使用：

```text
FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1 x) * W3 x)
```

其中 `x in R^{d_model}`，`W1, W3 in R^{d_ff x d_model}`，`W2 in R^{d_model x d_ff}`，标准设置中 `d_ff = 8/3 * d_model`。在具体 implementations 中，为了 hardware efficiency，可以将它 round 到接近的 64 的倍数。

N. Shazeer [20] 首先提出将 SiLU/Swish activation 与 GLUs 结合，并通过实验表明 SwiGLU 在 language modeling tasks 上优于 ReLU 和无 gating 的 SiLU 等 baselines。作业后面你将比较 SwiGLU 和 SiLU。虽然我们已经提到了一些关于这些组件的 heuristic arguments，论文也提供了更多支持证据，但保持 empirical perspective 很重要。Shazeer 论文中有一句现在很有名的话：

> "We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence."

**Problem (positionwise_feedforward): Implement the position-wise feed-forward network (2 points)**

Deliverable: 实现由 SiLU activation function 和 GLU 组成的 SwiGLU feed-forward network。

Note: 在这个特定场景中，为了 numerical stability，可以使用 `torch.sigmoid`。

你的 implementation 应将 `d_ff` 设置为约 `8/3 * d_model`，同时确保 inner feed-forward layer 的 dimensionality 是 64 的倍数，以充分利用硬件。为了使用提供的 tests 测试你的 implementation，需要实现 `[adapters.run_swiglu]` 中的 test adapter。然后运行 `uv run pytest -k test_swiglu` 测试你的 implementation。

#### 3.4.3 Relative Positional Embeddings

为了向模型注入 positional information，我们将实现 Rotary Position Embeddings [J. Su et al., 2021]，通常称为 RoPE。对于位于 token position `i` 的 query token `q(i) = W_q x(i) in R^d`，我们将应用 pairwise rotation matrix `R_i`，得到 `q'(i) = R_i q(i) = R_i W_q x(i)`。这里，`R_i` 会把 embedding elements 的 pair `q(i)_{2k-1:2k}` 当作 2d vectors，按角度 `theta_{i,k} = i / Theta^{(2k-2)/d}` 旋转，其中 `k in {1, ..., d/2}`，`Theta` 是某个常数。因此，可以把 `R_i` 看作一个 size 为 `d x d` 的 block-diagonal matrix，其中 blocks 为 `R_i^k`，`k in {1, ..., d/2}`。

虽然可以构造完整的 `d x d` matrix，但好的 solution 应利用该 matrix 的性质，更高效地实现 transformation。由于我们只关心给定 sequence 内 tokens 的相对旋转，可以跨 layers 和不同 batches 复用为 `cos(theta_{i,k})` 与 `sin(theta_{i,k})` 计算的值。如果想优化，可以使用一个被所有 layers 引用的单个 RoPE module，并在 `init` 期间用 `self.register_buffer(persistent=False)` 创建一个 2d pre-computed buffer 存放 sin 和 cos values，而不是使用 `nn.Parameter`，因为这些固定的 cosine 和 sine values 不应被学习。对 `q(i)` 做的同样 rotation process 也会应用到 `k(j)`，按对应的 `R_j` 进行旋转。注意，这一层没有 learnable parameters。

**Problem (rope): Implement RoPE (2 points)**

Deliverable: 实现一个 `RotaryPositionalEmbedding` class，将 RoPE 应用于 input tensor。

推荐以下接口：

```python
def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None)
```

构造 RoPE module，并在需要时创建 buffers。

- `theta: float`：RoPE 的 `Theta` value。
- `d_k: int`：query 和 key vectors 的 dimension。
- `max_seq_len: int`：将被输入的 maximum sequence length。
- `device: torch.device | None = None`：存放 buffer 的 device。

```python
def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor
```

处理 shape 为 `(..., seq_len, d_k)` 的 input tensor，并返回相同 shape 的 tensor。注意，你应允许 `x` 有任意数量的 batch dimensions。你应假设 token positions 是 shape 为 `(..., seq_len)` 的 tensor，指定 `x` 沿 sequence dimension 的 token positions。

你应使用 token positions 沿 sequence dimension 对可能已 precomputed 的 cos 和 sin tensors 做 slice。

为了测试你的 implementation，请完成 `[adapters.run_rope]` 并确保它通过 `uv run pytest -k test_rope`。

#### 3.4.4 Scaled Dot-Product Attention

现在我们将实现 A. Vaswani et al. [8] Section 3.2.1 中描述的 scaled dot-product attention。作为预备步骤，Attention operation 的定义会使用 softmax。Softmax 是一个 operation，它将 unnormalized scores vector 转换为 normalized distribution：

```text
softmax(v)_i = exp(v_i) / sum_j exp(v_j)
```

注意，对于大值，`exp(v_i)` 可能变成 `inf`，随后 `inf / inf` 会得到 `NaN`。可以注意到 softmax operation 对所有 inputs 加上任意常数 `c` 都不变，从而避免这一问题。我们可以利用这个性质获得 numerical stability：通常从 `v` 的所有元素中减去 `v` 的最大元素，使新的最大元素为 0。现在你将使用这个技巧实现 softmax。

**Problem (softmax): Implement softmax (1 point)**

Deliverable: 编写一个 function，将 softmax operation 应用于 tensor。你的 function 应接受两个参数：一个 tensor 和一个 dimension `i`，并沿 input tensor 的第 `i` 个 dimension 应用 softmax。Output tensor 应具有与 input tensor 相同的 shape，但第 `i` 个 dimension 现在会是一个 normalized probability distribution。请使用从第 `i` 个 dimension 的所有元素中减去该 dimension 最大值的技巧，以避免 numerical stability issues。

为了测试你的 implementation，请完成 `[adapters.run_softmax]`，并确保它通过 `uv run pytest -k test_softmax_matches_pytorch`。

现在我们可以如下数学定义 Attention operation：

```text
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

其中 `Q in R^{n x d_k}`，`K in R^{m x d_k}`，`V in R^{m x d_v}`。这里 `Q`、`K` 和 `V` 都是该 operation 的输入。注意，它们不是 learnable parameters。

**Masking:** 有时对 attention operation 的 output 进行 mask 很方便。Mask 的 shape 应为 `M in {True, False}^{n x m}`，boolean matrix 的每一行 `i` 表示 query `i` 应 attend 哪些 keys。按惯例，且有一点令人困惑，位置 `(i, j)` 处的值为 `True` 表示 query `i` 会 attend key `j`，值为 `False` 表示 query 不会 attend 该 key。换言之，值为 `True` 的 `(i, j)` pair 允许 "information flows"。例如，考虑一个 entries 为 `[[True, True, False]]` 的 `1 x 3` mask matrix。单个 query vector 只 attend 前两个 keys。

计算上，使用 masking 比在 subsequences 上计算 attention 高效得多。可以通过取 pre-softmax values，即 `QK^T / sqrt(d_k)`，并对 mask matrix 中值为 `False` 的 entries 加上 `-inf` 来实现。

**Problem (scaled_dot_product_attention): Implement scaled dot-product attention (5 points)**

Deliverable: 实现 scaled dot-product attention function。你的 implementation 应处理 shape 为 `(batch_size, ..., seq_len, d_k)` 的 keys 和 queries，以及 shape 为 `(batch_size, ..., seq_len, d_v)` 的 values，其中 `...` 表示任意数量的其他 batch-like dimensions，如果有的话。Implementation 应返回 shape 为 `(batch_size, ..., seq_len, d_v)` 的 output。关于 batch-like dimensions，参见 Section 3.2。

你的 implementation 还应支持一个可选的 user-provided boolean mask，shape 为 `(seq_len, seq_len)`。Mask value 为 `True` 的 positions 的 attention probabilities 应共同 sum to 1，mask value 为 `False` 的 positions 的 attention probabilities 应为 0。

为了使用提供的 tests 测试你的 implementation，需要实现 `[adapters.run_scaled_dot_product_attention]` 中的 test adapter。`uv run pytest -k test_scaled_dot_product_attention` 会在 third-order input tensors 上测试你的 implementation，而 `uv run pytest -k test_4d_scaled_dot_product_attention` 会在 fourth-order input tensors 上测试。

#### 3.4.5 Causal Multi-Head Self-Attention

我们将实现 A. Vaswani et al. [8] Section 3.2.2 中描述的 multi-head self-attention。回忆一下，从数学上看，应用 multi-head attention 的 operation 定义如下：

```text
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)
head_i = Attention(Q_i, K_i, V_i)
```

其中 `Q_i`、`K_i`、`V_i` 分别是 `Q`、`K`、`V` 在 embedding dimension 上的第 `i in {1, ..., h}` 个 slice，slice 大小为 `d_k` 或 `d_v`。Attention 是 Section 3.4.4 中定义的 scaled dot-product attention operation。由此可以形成 multi-head self-attention operation：

```text
MultiHeadSelfAttention(x) = W_O MultiHead(W_Q x, W_K x, W_V x)
```

这里的 learnable parameters 是 `W_Q in R^{h d_k x d_model}`，`W_K in R^{h d_k x d_model}`，`W_V in R^{h d_v x d_model}`，以及 `W_O in R^{d_model x h d_v}`。由于在 multi-head attention operation 中 `Q`、`K`、`V` 会被 sliced，可以把 `W_Q`、`W_K` 和 `W_V` 视为沿 output dimension 为每个 head 分离。实现正常后，key、value 和 query projections 总共应通过三次 matrix multiplies 计算。作为 stretch goal，可以尝试把 key、query 和 value projections 合并到单个 weight matrix 中，这样只需要一次 matrix multiply。

**Causal masking**

你的 implementation 应防止模型 attend to sequence 中的 future tokens。换言之，如果模型得到 token sequence `t_1, ..., t_n`，而我们想为 prefix `t_1, ..., t_i` 计算 next-word predictions，其中 `i < n`，模型就不应能访问，也就是 attend to，位置 `t_{i+1}, ..., t_n` 的 token representations，因为 inference 生成文本时它无法访问这些 tokens，而且这些 future tokens 会泄漏 true next word 的身份信息，从而使 language modeling pre-training objective 变得平凡。对于 input token sequence `t_1, ..., t_n`，我们可以 naive 地运行 multi-head self-attention `n` 次，分别对应 sequence 的 `n` 个 unique prefixes，从而防止访问 future tokens。相反，我们会使用 causal attention masking，它允许 token `i` attend 到 sequence 中所有 `j <= i` 的 positions。你可以使用 `torch.triu` 或 broadcasted index comparison 构造这个 mask，并应利用你在 Section 3.4.4 中实现的 scaled dot-product attention 已支持 attention masking 这一事实。

**Applying RoPE**

RoPE 应应用到 query 和 key vectors，而不是 value vectors。此外，head dimension 应作为 batch dimension 处理，因为在 multi-head attention 中，attention 是对每个 head 独立应用的。这意味着完全相同的 RoPE rotation 应应用于每个 head 的 query 和 key vectors。

**Problem (multihead_self_attention): Implement causal multi-head self-attention (5 points)**

Deliverable: 将 causal multi-head self-attention 实现为 `torch.nn.Module`。你的 implementation 至少应接受以下参数：

- `d_model: int`：Transformer block inputs 的 dimensionality。
- `num_heads: int`：multi-head self-attention 使用的 heads 数量。

遵循 A. Vaswani et al. [8]，设置 `d_k = d_v = d_model / h`。为了使用提供的 tests 测试你的 implementation，请实现 `[adapters.run_multihead_self_attention]` 中的 test adapter。然后运行 `uv run pytest -k test_multihead_self_attention`。

### 3.5 The Full Transformer LM

让我们先组装 Transformer block，建议回看 Figure 2。Transformer block 包含两个 sub-layers：一个用于 multihead self attention，另一个用于 SwiGLU feed-forward network。在每个 sub-layer 中，我们先执行 RMSNorm，然后执行 main operation，即 MHA 或 FF，最后加上 residual connection。

具体来说，Transformer block 的前半部分，也就是第一个 sub-layer，应从 input `x` 产生 output `y`，并实现以下更新：

```text
y = x + MultiHeadSelfAttention(RMSNorm(x))
```

**Problem (transformer_block): Implement the Transformer block (3 points)**

按照 Section 3.4 描述并在 Figure 2 中展示的方式，实现 pre-norm Transformer block。你的 Transformer block 至少应接受以下参数：

- `d_model: int`：Transformer block inputs 的 dimensionality。
- `num_heads: int`：multi-head self-attention 使用的 heads 数量。
- `d_ff: int`：position-wise feed-forward inner layer 的 dimensionality。

为了测试你的 implementation，请实现 adapter `[adapters.run_transformer_block]`。然后运行 `uv run pytest -k test_transformer_block` 测试你的 implementation。

Deliverable: 通过提供 tests 的 Transformer block code。

现在我们按 Figure 1 的 high-level diagram 将 blocks 组合起来。按照 Section 3.1.0.1 中对 embedding 的描述，将它输入 `num_layers` 个 Transformer blocks，然后传入 final layer norm 和 LM head，得到 vocabulary 上的 unnormalized distribution，即 logits。

**Problem (transformer_lm): Implementing the Transformer LM (3 points)**

现在把所有组件放在一起。按照 Section 3.1 描述并由 Figure 1 展示的方式，实现 Transformer language model。你的 implementation 至少应接受前面 Transformer block 的所有 construction parameters，以及以下额外参数：

- `vocab_size: int`：vocabulary size，用于确定 token embedding matrix 的 dimensionality。
- `context_length: int`：maximum context length，用于确定 RoPE sin 和 cos buffer 的 dimensionality。
- `num_layers: int`：使用的 Transformer blocks 数量。

为了使用提供的 tests 测试你的 implementation，首先需要实现 `[adapters.run_transformer_lm]` 中的 test adapter。然后运行 `uv run pytest -k test_transformer_lm`。

Deliverable: 通过上述 tests 的 Transformer LM module。

#### Resource accounting

理解 Transformer 各部分如何消耗 compute 和 memory 很有用。我们将进行一些基础的 "FLOPs accounting"。Transformer 中绝大多数 FLOPs 来自 matrix multiplies，因此核心方法很简单：

1. 写出 Transformer forward pass 中所有 matrix multiplies。
2. 将每个 matrix multiply 转换为所需 FLOPs。

第二步中，以下事实很有用：

Rule: 给定 `A in R^{m x n}` 和 `B in R^{n x p}`，matrix-matrix product `AB` 需要 `2mnp` FLOPs。

原因是 `(AB)[i, j] = A[i, :] dot B[:, j]`，该 dot product 需要 `n` 次 additions 和 `n` 次 multiplications，即 `2n` FLOPs。由于 matrix-matrix product `AB` 有 `m x p` 个 entries，总 FLOPs 为 `(2n)(mp) = 2mnp`。

在做下一个问题前，逐一检查你的 Transformer block 和 Transformer LM 的每个 component，并列出所有 matrix multiplies 及其 FLOPs costs，会很有帮助。

**Problem (transformer_accounting): Transformer LM resource accounting (5 points)**

(a) 考虑使用本作业 architecture 的 GPT-2 XL-sized model，其配置如下：

- `vocab_size: 50,257`
- `context_length: 1,024`
- `num_layers: 48`
- `d_model: 1,600`
- `num_heads: 25`
- `d_ff: 4,288`，即最接近 `8/3 * 1,600` 的 64 的倍数

假设我们用该配置构建模型。模型有多少 trainable parameters？如果每个 parameter 使用 single-precision floating point 表示，仅加载该模型需要多少 memory？

Deliverable: 一到两句话回答。

(b) 找出完成 GPT-2 XL-shaped model 的 forward pass 所需的 matrix multiplies。这些 matrix multiplies 总共需要多少 FLOPs？假设 input sequence 有 `context_length` 个 tokens。

Deliverable: 一个 matrix multiplies 列表，包括描述，以及所需 FLOPs 总数。

(c) 基于上面的分析，模型的哪些部分需要最多 FLOPs？

Deliverable: 一到两句话回答。

(d) 用 GPT-2 small，12 layers、768 `d_model`、12 heads；GPT-2 medium，24 layers、1024 `d_model`、16 heads；GPT-2 large，36 layers、1280 `d_model`、20 heads，重复你的分析。随着 model size 增大，Transformer LM 中哪些部分占总 FLOPs 的比例变大或变小？

Deliverable: 对每个 model，提供 model components 及其 associated FLOPs 的 breakdown，表示为 forward pass 所需总 FLOPs 的比例。此外，用一到两句话说明改变 model size 如何改变各 component 的 FLOPs 占比。

(e) 取 GPT-2 XL，并将 context length 增加到 16,384。一次 forward pass 的总 FLOPs 如何变化？Model components 的 FLOPs 相对贡献如何变化？

Deliverable: 一到两句话回答。

## 4 Training a Transformer LM

现在，我们已经有了预处理数据的步骤，即 tokenizer，也有了 model，即 Transformer。剩下的是构建支持 training 的全部代码。这包括：

- Loss：需要定义 loss function，即 cross-entropy。
- Optimizer：需要定义 optimizer 来最小化该 loss，即 AdamW。
- Training loop：需要所有支持性 infrastructure，用于加载数据、保存 checkpoints 和管理 training。

### 4.1 Cross-entropy loss

回忆一下，对于长度为 `m + 1` 的每个 sequence `x`，以及 `i = 1, ..., m`，Transformer language model 定义分布 `p_theta(x_{i+1} | x_{1:i})`。给定由长度为 `m + 1` 的 sequences 组成的 training set `D`，我们定义标准 cross-entropy，即 negative log-likelihood，loss function：

```text
l(theta; D) = (1 / (|D| m)) * sum_{x in D} sum_{i=1}^m -log p_theta(x_{i+1} | x_{1:i})
```

注意，Transformer 的一次 forward pass 会为所有 `i = 1, ..., m` 产生 `p_theta(x_{i+1} | x_{1:i})`。

具体而言，Transformer 为每个 position `i` 计算 logits `o_i in R^{vocab_size}`，从而得到：

```text
p(x_{i+1} | x_{1:i}) = softmax(o_i)[x_{i+1}]
```

Cross-entropy loss 通常相对于 logits vector `o_i in R^{vocab_size}` 和 target `x_{i+1}` 定义。

和 softmax 一样，实现 cross-entropy loss 也需要注意 numerical issues。

**Problem (cross_entropy): Implement cross-entropy (1 point)**

Deliverable: 编写一个 function 计算 cross-entropy loss，它接收 predicted logits `o_i` 和 targets `x_{i+1}`，并计算 cross-entropy `l_i = -log softmax(o_i)[x_{i+1}]`。你的 function 应处理以下事项：

- 为 numerical stability 减去最大元素。
- 尽可能抵消 `log` 和 `exp`。
- 处理任何额外的 batch dimensions，并返回 batch 上的平均值。与 Section 3.2 一样，我们假设 batch-like dimensions 总是位于 vocabulary size dimension 之前。

实现 `[adapters.run_cross_entropy]`，然后运行 `uv run pytest -k test_cross_entropy` 测试你的 implementation。

#### Perplexity

Cross-entropy 足以用于 training，但 evaluation 时我们还希望报告 perplexity。对于长度为 `m` 的 sequence，如果产生 cross-entropy losses `l_1, ..., l_m`，则：

```text
perplexity = exp((1 / m) * sum_i l_i)
```

### 4.2 The SGD Optimizer

现在我们有了 loss function，接下来开始探索 optimizers。最简单的 gradient-based optimizer 是 Stochastic Gradient Descent (SGD)。我们从随机初始化 parameters `theta_0` 开始。然后对于每一步 `t = 0, ..., T - 1`，执行如下更新：

```text
theta_{t+1} <- theta_t - alpha_t * grad L(theta_t; B_t)
```

其中 `B_t` 是从 dataset `D` 中采样的 random batch of data，learning rate `alpha_t` 和 batch size `|B_t|` 是 hyperparameters。

#### 4.2.1 Implementing SGD in PyTorch

为了实现 optimizers，我们会 subclass PyTorch 的 `torch.optim.Optimizer` class。`Optimizer` subclass 必须实现两个 methods：

`def __init__(self, params, ...)` 应初始化 optimizer。这里，`params` 是要优化的 parameters 集合，或 parameter groups，如果用户希望对模型不同部分使用不同 hyperparameters，例如 learning rates。请确保将 `params` 传给 base class 的 `__init__` method，它会存储这些 parameters 供 `step` 使用。你可以根据 optimizer 接受额外 arguments，例如 learning rate 是常见参数，并把它们作为 dictionary 传给 base class constructor，keys 是你为这些参数选择的 names，即 strings。

`def step(self)` 应对 parameters 做一次更新。在 training loop 中，它会在 backward pass 之后被调用，因此你可以访问 last batch 上的 gradients。该 method 应遍历每个 parameter tensor `p`，并原地修改它们，也就是设置 `p.data`，它保存与 parameter 关联的 tensor。更新基于 `p.grad`，如果存在，它表示 loss 关于该 parameter 的 gradient tensor。

PyTorch optimizer API 有一些细节，用例子解释更容易。为了让例子更丰富，我们实现 SGD 的一个小变体：learning rate 在训练过程中衰减，从 initial learning rate `alpha` 开始，随着时间逐渐使用更小步长：

```text
theta_{t+1} = theta_t - alpha / sqrt(t + 1) * grad L(theta_t; B_t)
```

下面展示这个版本的 SGD 如何实现为 PyTorch Optimizer：

```python
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss
```

在 `__init__` 中，我们将 parameters 和 default hyperparameters 传给 base class constructor。Parameters 可能分组，每组有不同 hyperparameters。如果 parameters 只是单个 `torch.nn.Parameter` objects 集合，base constructor 会创建单个 group，并为其分配 default hyperparameters。然后在 `step` 中，我们遍历每个 parameter group，再遍历该 group 中每个 parameter，并应用 Equation 20。这里，我们把 iteration number 作为与每个 parameter 关联的 state 保存：先读取它，在 gradient update 中使用，然后更新它。API 规定用户可能传入 callable `closure`，用于在 optimizer step 前重新计算 loss。我们不需要在将使用的 optimizers 中依赖它，但为了符合 API，会加入它。

为了看到它如何工作，可以使用以下 minimal training loop 示例：

```python
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1)

for t in range(100):
    opt.zero_grad()  # Reset the gradients for all learnable parameters.
    loss = (weights**2).mean() # Compute a scalar loss value.
    print(loss.cpu().item())
    loss.backward() # Run backward pass, which computes gradients.
    opt.step() # Run optimizer step.
```

这是 training loop 的典型结构：每次 iteration 中，我们计算 loss，并执行 optimizer 的一步。当训练 language models 时，learnable parameters 来自 model，在 PyTorch 中 `m.parameters()` 会给出该集合。Loss 会在 sampled batch of data 上计算，但 training loop 的基本结构相同。

**Problem (learning_rate_tuning): Tuning the learning rate (1 point)**

正如我们将看到的，learning rate 是最影响 training 的 hyperparameters 之一。让我们在 toy example 中实际观察这一点。使用上面的 SGD 示例，再分别用三个 learning rate values：`1e1`、`1e2` 和 `1e3` 运行，只跑 10 个 training iterations。对于这些 learning rates，loss 会发生什么？它 decay 更快、更慢，还是 diverge，也就是训练过程中增大？

Deliverable: 用一到两句话描述你观察到的行为。

### 4.3 AdamW

现代 language models 通常使用比 SGD 更复杂的 optimizers。最近使用的大多数 optimizers 都是 Adam optimizer [D. P. Kingma et al., 2015] 的派生。我们将使用 AdamW [I. Loshchilov et al., 2019]，它在近期工作中被广泛采用。AdamW 对 Adam 提出一种修改：通过添加 weight decay，即每次 iteration 将 parameters 拉向 0，改善 regularization，并且这种方式与 gradient update 解耦。我们将按照 I. Loshchilov et al. [23] 的 algorithm 2 实现 AdamW。

AdamW 是 stateful 的：对每个 parameter，它都跟踪 first 和 second moments 的 running estimate。因此，AdamW 用额外 memory 换取更好的 stability 和 convergence。除了 learning rate `alpha`，AdamW 还有一对 hyperparameters `(beta_1, beta_2)`，用于控制 moment estimates 的更新，以及 weight decay rate `lambda`。典型应用会将 `(beta_1, beta_2)` 设置为 `(0.9, 0.999)`，但 LLaMA [H. Touvron et al., 2023] 和 GPT-3 [T. B. Brown et al., 2020] 等 large language models 通常使用 `(0.9, 0.95)`。Algorithm 可以写作如下，其中 `epsilon` 是一个小值，例如 `1e-8`，用于在 `v` 出现极小值时提升 numerical stability：

```text
Algorithm 1: AdamW Optimizer
1 init(theta)                         Initialize learnable parameters
2 m <- 0                              Initial value of the first moment vector; same shape as theta
3 v <- 0                              Initial value of the second moment vector; same shape as theta
4 for t = 1, ..., T do
5     Sample batch of data B_t
6     g <- grad_theta l(theta; B_t)    Compute the gradient of the loss
7     alpha_t <- alpha * sqrt(1 - beta_2^t) / (1 - beta_1^t)
8     theta <- theta - alpha lambda theta
9     m <- beta_1 m + (1 - beta_1) g
10    v <- beta_2 v + (1 - beta_2) g^2
11    theta <- theta - alpha_t * m / (sqrt(v) + epsilon)
12 end for
```

注意 `t` 从 1 开始。现在你将实现这个 optimizer。

**Problem (adamw): Implement AdamW (2 points)**

Deliverable: 将 AdamW optimizer 实现为 `torch.optim.Optimizer` 的 subclass。你的 class 应在 `__init__` 中接收 learning rate `alpha`，以及 `beta`、`epsilon` 和 `lambda` hyperparameters。为了帮助你维护 state，base `Optimizer` class 提供 dictionary `self.state`，它将 `nn.Parameter` objects 映射到一个 dictionary，用于存储该 parameter 需要的任何信息。对于 AdamW 来说，这会是 moment estimates。实现 `[adapters.get_adamw_cls]`，并确保它通过 `uv run pytest -k test_adamw`。

**Problem (adamw_accounting): Resource accounting for training with AdamW (2 points)**

让我们计算运行 AdamW 需要多少 memory 和 compute。假设所有 tensors 都使用 `float32`。

(a) 运行 AdamW 需要多少 peak memory？请根据 parameters、activations、gradients 和 optimizer state 的 memory usage 分解你的答案。用 `batch_size` 和 model hyperparameters，即 `vocab_size`、`context_length`、`num_layers`、`d_model`、`num_heads` 表示答案。假设 `d_ff = 8/3 * d_model`。

为简化计算 activations 的 memory usage，只考虑以下 components：

- Transformer block
- RMSNorm(s)
- Multi-head self-attention sublayer：`QKV` projections、`QK^T` matrix multiply、softmax、values 的 weighted sum、output projection。
- Position-wise feed-forward (SwiGLU)：`W1`、`W2`、gate branch 上的 SiLU、element-wise product、`W3`。
- final RMSNorm
- output embedding
- logits 上的 cross-entropy

Deliverable: 分别给出 parameters、activations、gradients 和 optimizer state 的 algebraic expression，以及 total。

(b) 将你的答案实例化为 GPT-2 XL-shaped model，得到一个只依赖 `batch_size` 的表达式。在 80GB memory 内，maximum batch size 是多少？

Deliverable: 一个形如 `a * batch_size + b` 的 expression，其中 `a` 和 `b` 是数值，以及表示 maximum batch size 的一个数字。

(c) 运行 AdamW 的一步需要多少 FLOPs？

Deliverable: 一个 algebraic expression，并附简要说明。

(d) Model FLOPs utilization (MFU) 定义为 observed throughput，即 tokens per second，与 hardware theoretical peak FLOP throughput 之比 [A. Chowdhery et al., 2022]。NVIDIA H100 GPU 对 "float32"，实际上是 TensorFloat-32，现实中是 "bfloat19" operations，的 theoretical peak 为 495 teraFLOP/s。假设你能达到 50% MFU，在单个 H100 上以 batch size 1024 训练 GPT-2 XL 400K steps 需要多久？遵循 J. Kaplan et al. [25] 和 J. Hoffmann et al. [26]，假设 backward pass 的 FLOPs 是 forward pass 的两倍。

Deliverable: 训练所需小时数，并附简要说明。

### 4.4 Learning rate scheduling

能让 loss 下降最快的 learning rate value 通常会在 training 过程中变化。训练 Transformers 时，通常使用 learning rate schedule：开始使用较大的 learning rate，在初期进行更快更新，然后随着模型训练逐渐 decay 到较小值。本作业中，我们将实现用于训练 LLaMA [H. Touvron et al., 2023] 的 cosine annealing schedule。

Scheduler 只是一个 function，它接收当前 step `t` 和其他相关 parameters，例如 initial 和 final learning rates，并返回 step `t` 中用于 gradient update 的 learning rate。最简单的 schedule 是 constant function，它对任意 `t` 返回相同 learning rate。

Cosine annealing learning rate schedule 接收：(i) 当前 iteration `t`，(ii) maximum learning rate `alpha_max`，(iii) minimum，即 final，learning rate `alpha_min`，(iv) warm-up iterations 数量 `T_w`，以及 (v) cosine annealing 的 final iteration `T_c`。Iteration `t` 的 learning rate 定义如下：

- Warm-up：如果 `t < T_w`，则 `alpha_t = (t / T_w) * alpha_max`。
- Cosine annealing：如果 `T_w <= t <= T_c`，则 `alpha_t = alpha_min + 1/2 * (1 + cos(((t - T_w) / (T_c - T_w)) * pi)) * (alpha_max - alpha_min)`。
- Post-annealing：如果 `t > T_c`，则 `alpha_t = alpha_min`。

**Problem (learning_rate_schedule): Implement cosine learning rate schedule with warmup (1 point)**

编写一个 function，接收 `t`、`alpha_max`、`alpha_min`、`T_w` 和 `T_c`，并按照上面定义的 scheduler 返回 learning rate `alpha_t`。然后实现 `[adapters.get_lr_cosine_schedule]`，并确保它通过 `uv run pytest -k test_get_lr_cosine_schedule`。

### 4.5 Gradient clipping

训练期间，有时会遇到产生 large gradients 的 training examples，这可能使 training 不稳定。为了缓解这一点，实践中常用的一种技术是 gradient clipping。其思想是在每次 backward pass 后、optimizer step 前，对 gradient norm 施加上限。

给定所有 parameters 的 gradient `g`，我们计算其 `l2`-norm `||g||_2`。如果该 norm 小于 maximum value `M`，则保持 `g` 不变；否则，将 `g` 缩放一个 factor：`M / (||g||_2 + epsilon)`，其中加入小的 `epsilon`，例如 `1e-6`，用于 numeric stability。注意，得到的 norm 会略小于 `M`。

**Problem (gradient_clipping): Implement gradient clipping (1 point)**

编写一个 function 实现 gradient clipping。你的 function 应接收 parameters list 和 maximum `l2`-norm。它应原地修改每个 parameter gradient。使用 `epsilon = 1e-6`，即 PyTorch default。然后实现 adapter `[adapters.run_gradient_clipping]`，并确保它通过 `uv run pytest -k test_gradient_clipping`。

## 5 Training loop

现在我们终于要把目前构建的主要组件组合起来：tokenized data、model 和 optimizer。

### 5.1 Data Loader

Tokenized data，例如你在 `tokenizer_experiments` 中准备的数据，是一个 tokens 单序列 `x = (x_1, ..., x_n)`。虽然源数据可能由独立 documents 组成，例如不同 web pages 或 source code files，但常见做法是将它们全部 concatenate 为单个 tokens 序列，并在它们之间加入 delimiter，例如 `<|endoftext|>` token。

Data loader 会把这转换为 batches stream，其中每个 batch 由 `B` 个长度为 `m` 的 sequences，以及对应的 next tokens，也为长度 `m`，组成。例如，对于 `B = 1`、`m = 3`，`([x_2, x_3, x_4], [x_3, x_4, x_5])` 可能是一个 batch。

以这种方式加载数据会从多方面简化 training。首先，任意 `1 <= i <= n - m` 都给出一个有效 training sequence，因此采样 training sequences 很简单。由于所有 training sequences 长度相同，不需要 pad input sequences，这能提升 hardware utilization，也能通过增大 batch size `B` 进一步提升利用率。最后，我们也不需要加载完整 dataset 来采样 training data，因此更容易处理可能无法装入内存的大型 datasets。

**Problem (data_loading): Implement data loading (2 points)**

Deliverable: 编写一个 function，接收 numpy array `x`，即包含 token IDs 的 integer array，一个 `batch_size`、一个 `context_length` 和一个 PyTorch device string，例如 `'cpu'` 或 `'cuda:0'`，并返回一对 tensors：sampled input sequences 和对应的 next-token targets。两个 tensors 的 shape 都应为 `(batch_size, context_length)`，包含 token IDs，并都放在请求的 device 上。为了用提供的 tests 测试你的 implementation，首先需要实现 `[adapters.run_get_batch]` 中的 test adapter。然后运行 `uv run pytest -k test_get_batch`。

#### Low-Resource Tip: Data loading on CPU or Apple Silicon

如果你计划在 CPU 或 Apple Silicon 上训练 LM，需要将 data 移动到正确 device 上；同样，后面也应对 model 使用相同 device。

如果你使用 CPU，可以使用 `'cpu'` device string；在 Apple Silicon，即 M 系列芯片上，可以使用 `'mps'` device string。

关于 MPS，可参阅以下资源：

- `https://docs.pytorch.org/docs/stable/mps.html`
- `https://docs.pytorch.org/docs/stable/notes/mps.html`
- `https://developer.apple.com/documentation/metalperformanceshaders`

如果 dataset 太大，无法加载进内存怎么办？我们可以使用名为 `mmap` 的 Unix system call，它将磁盘上的文件映射到 virtual memory，并在访问该 memory location 时 lazy 地加载文件内容。因此，你可以“假装”整个 dataset 都在内存中。Numpy 通过 `np.memmap` 实现这一点；如果你最初使用 `np.save` 保存 array，也可以对 `np.load` 使用 `mmap_mode='r'` flag。这会返回一个 numpy array-like object，在访问 entries 时按需加载。当 training 期间从 dataset，即 numpy array，采样时，请确保以 memory-mapped mode 加载 dataset，通过 `np.memmap` 或 `np.load` 的 `mmap_mode='r'` flag，具体取决于你如何保存 array。还要确保指定与所加载 array 匹配的 dtype。显式验证 memory-mapped data 看起来正确可能会很有帮助，例如不包含超过预期 vocabulary size 的 values。

### 5.2 Checkpointing

除了加载数据，我们还需要在 training 过程中保存 models。运行 jobs 时，我们常常希望能够恢复中途停止的 training run，例如由于 job timeout、machine failure 等。即使一切顺利，我们也可能希望之后访问 intermediate models，例如 post-hoc 研究 training dynamics，或从不同 training stages 的 models 采样等。

Checkpoint 应包含恢复 training 所需的全部 states。至少，我们当然希望能够恢复 model weights。如果使用 stateful optimizer，例如 AdamW，还需要保存 optimizer state，例如 AdamW 的 moment estimates。最后，为了恢复 learning rate schedule，需要知道停止时的 iteration number。PyTorch 让保存这些内容变得简单：每个 `nn.Module` 都有 `state_dict()` method，返回包含所有 learnable weights 的 dictionary；之后可以用配套 method `load_state_dict()` 恢复这些 weights。任意 `torch.optim.Optimizer` 也是如此。最后，`torch.save(obj, dest)` 可以把 object，例如包含 tensors 以及普通 Python objects 如 integers 的 dictionary，dump 到文件路径或 file-like object，然后可用 `torch.load(src)` 加载回内存。

**Problem (checkpointing): Implement model checkpointing (1 point)**

实现以下两个 functions 来加载和保存 checkpoints：

```python
def save_checkpoint(model, optimizer, iteration, out)
```

该 function 应将 model、optimizer 和 iteration 的全部 state dump 到 file-like object `out`。你可以使用 model 和 optimizer 的 `state_dict` method 获取它们的相关 states，并使用 `torch.save(obj, out)` 将 `obj` dump 到 `out`，PyTorch 支持 path 或 file-like object。典型选择是让 `obj` 成为 dictionary，但只要之后能加载 checkpoint，你可以使用任何格式。

该 function 期望以下参数：

- `model: torch.nn.Module`
- `optimizer: torch.optim.Optimizer`
- `iteration: int`
- `out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`

```python
def load_checkpoint(src, model, optimizer)
```

该 function 应从 `src`，即 path 或 file-like object，加载 checkpoint，然后从该 checkpoint 恢复 model 和 optimizer states。你的 function 应返回 checkpoint 中保存的 iteration number。可以使用 `torch.load(src)` 恢复你在 `save_checkpoint` implementation 中保存的内容，并用 model 和 optimizer 的 `load_state_dict` method 将它们恢复到之前的 states。

该 function 期望以下参数：

- `src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`
- `model: torch.nn.Module`
- `optimizer: torch.optim.Optimizer`

实现 `[adapters.run_save_checkpoint]` 和 `[adapters.run_load_checkpoint]` adapters，并确保它们通过 `uv run pytest -k test_checkpointing`。

### 5.3 Training loop

现在终于到了把你实现的所有组件组合到 main training script 中的时候。让开始不同 hyperparameters 的 training runs 变得容易会很有回报，例如通过 command-line arguments 接收这些参数，因为之后你会多次运行实验，以研究不同选择如何影响 training。

**Problem (training_together): Put it together (4 points)**

Deliverable: 编写一个 script，运行 training loop，在 user-provided input 上训练你的 model。具体来说，我们建议你的 training script 至少支持以下能力：

- 配置和控制各种 model 与 optimizer hyperparameters。
- 使用 `np.memmap` memory-efficient 地加载大型 training 和 validation datasets。
- 将 checkpoints 序列化到 user-provided path。
- 周期性记录 training 和 validation performance，例如输出到 console 或外部服务如 Weights and Biases。

## 6 Generating text

现在我们可以训练 models，最后一块是让模型能够生成文本。回忆一下，language model 接收一个长度为 `sequence_length` 的 integer sequence，可能 batched，并产生一个 size 为 `(sequence_length, vocab_size)` 的 matrix，其中 sequence 的每个元素都是预测该 position 之后 next token 的 probability distribution。现在我们将编写几个 functions，将其变为生成新 sequences 的 sampling scheme。

### Softmax

按照标准惯例，language model 的 output 是 final linear layer 的输出，即 "logits"，因此我们需要通过前面 Equation 10 中看到的 softmax operation 将其转换为 normalized probability。

### Decoding

为了从 model 生成文本，也就是 decode，我们会向 model 提供 prefix tokens 序列，即 "prompt"，并要求它产生 vocabulary 上的 probability distribution，用来预测 sequence 中的 next token。然后，我们从该 vocabulary items 上的分布中 sample，以确定 next output token。

具体而言，decoding process 的一步应接收 sequence `x_{1...t}`，并通过以下等式返回 token `x_{t+1}`：

```text
P(x_{t+1} = i | x_{1...t}) = exp(v_i) / sum_j exp(v_j)
v = TransformerLM(x_{1...t})_t in R^{vocab_size}
```

其中 TransformerLM 是我们的 model，它接收长度为 `sequence_length` 的 sequence，并产生 size 为 `(sequence_length, vocab_size)` 的 matrix。由于我们寻找第 `t` 个 position 的 next token prediction，所以取该 matrix 的最后一个元素。

这样就得到一个 basic decoder：反复从这些 one-step conditionals 中 sampling，并将之前生成的 output token append 到下一 decoding timestep 的 input 中，直到生成 end-of-sequence token `<|endoftext|>`，或达到用户指定的 maximum number of tokens。

### Decoder tricks

我们将使用 small models 做实验，而 small models 有时会生成 very low-quality texts。两个简单的 decoder tricks 可以帮助改善这些问题。首先是 temperature scaling：我们用 temperature parameter `tau` 修改 softmax，新的 softmax 为：

```text
softmax(v, tau)_i = exp(v_i / tau) / sum_j exp(v_j / tau)
```

注意，当 `tau -> 0` 时，`v` 中最大的元素会占主导，softmax output 会变成集中在最大元素上的 one-hot vector。

第二个 trick 是 nucleus sampling 或 top-p sampling。它通过截断 low-probability tokens 来修改 sampling distribution。令 `q` 为从 temperature-scaled softmax 得到的 size 为 `vocab_size` 的 probability distribution。带 hyperparameter `p` 的 nucleus sampling 根据以下等式产生 next token：

```text
P(x_{t+1} = i | q) = q_i / sum_{j in V(p)} q_j, if i in V(p)
P(x_{t+1} = i | q) = 0, otherwise
```

其中 `V(p)` 是最小的 indices 集合，使得 `sum_{j in V(p)} q_j >= p`。可以先按 magnitude 对 probability distribution `q` 排序，再选择最大的 vocabulary elements，直到达到目标 `p` level，从而容易地计算该 quantity。

**Problem (decoding): Decoding (3 points)**

Deliverable: 实现一个 function，从你的 language model 中 decode。我们建议支持以下功能：

- 为 user-provided prompt 生成 completions，也就是接收某个 `x_{1...t}`，并 sample completion，直到遇到 `<|endoftext|>` token。
- 允许用户控制 maximum number of generated tokens。
- 给定 desired temperature value，在 sampling 前对 predicted next-token distributions 应用 softmax temperature scaling。
- 给定 user-specified threshold value，支持 Top-p sampling [A. Holtzman et al., 2020]，也称为 nucleus sampling。

## 7 Experiments

现在是把所有内容组合起来，并在 pretraining dataset 上训练 small language models 的时候。

### 7.1 How to Run Experiments and Deliverables

理解 Transformer architecture components 背后理由的最佳方式，是亲自修改它并运行。Hands-on experience 无可替代。

为此，能快速、一致地做实验，并记录做过的事情，非常重要。为了快速实验，我们会在 small-scale model，大约 17M total parameters，和简单 dataset TinyStories 上运行许多 experiments。为了保持一致，你将系统性地 ablate components 并改变 hyperparameters。为了保留记录，我们要求你提交 experiments log 和每个 experiment 对应的 learning curves。

为了能提交 loss curves，请确保周期性评估 validation losses，并记录 steps 数和 wall-clock times。Weights and Biases 等 logging infrastructure 可能有帮助。

**Problem (experiment_log): Experiment logging (3 points)**

对于你的 training 和 evaluation code，创建 experiment tracking infrastructure，使你能相对于 gradient steps 和 wall-clock time 追踪 experiments 和 loss curves。

Deliverable: 用于 experiments 的 logging infrastructure code，以及本 section 后续 assignment problems 的 experiment log，即记录所有你尝试过事项的 document。

### 7.2 TinyStories

我们将从一个非常简单的 dataset 开始，即 TinyStories [R. Eldan et al. [1]]。在该数据集上，models 会训练得很快，而且我们能看到一些有趣行为。获取该 dataset 的说明见 Section 1。下面是该 dataset 的一个示例。

**Example (tinystories_example): One example from TinyStories**

从前有一个名叫 Ben 的小男孩。Ben 喜欢探索身边的世界。他看到了许多令人惊奇的东西，比如商店里展示的美丽花瓶。一天，Ben 在商店里走着，发现了一个非常特别的花瓶。Ben 看到它时非常惊讶！他说：“哇，那真是一个很棒的花瓶！我可以买它吗？”店主微笑着说：“当然可以。你可以把它带回家，给所有朋友看看它有多棒！”于是 Ben 把花瓶带回家，并为它感到非常自豪！他把朋友们叫来，向他们展示这个很棒的花瓶。所有朋友都觉得花瓶很美，也不敢相信 Ben 这么幸运。Ben 就是这样在商店里找到了一个很棒的花瓶！

#### 7.2.1 Hyperparameter tuning

我们会给出一些非常基础的 hyperparameters 作为起点，并要求你找到其他表现良好的设置。

**Vocab size 10000.** 典型 vocabulary sizes 是数万到数十万。你应改变这个值，观察 vocabulary 和 model behavior 如何变化。

**Context length 256.** TinyStories 这样的简单 datasets 可能不需要长 sequence lengths，但后面的 OpenWebText data 可能需要改变该值。尝试改变它，并观察它对 per-iteration runtime 和 final perplexity 的影响。

**d_model 512.** 这比许多 small Transformer papers 中使用的 768 dimensions 略小，但会更快。

**d_ff 1344.** 这大约是 `8/3 d_model`，同时是 64 的倍数，有利于 GPU performance。

**RoPE theta parameter Theta 10000.**

**Number of layers and heads 4 layers, 16 heads.** 合起来，这会得到约 17M non-embedding parameters，是一个相当小的 Transformer。

**Total tokens processed 327,680,000.** 你的 `batch size * total step count * context length` 应大致等于这个值。

你应通过 trial and error 为以下其他 hyperparameters 找到好的 defaults：learning rate、learning rate warmup、AdamW 的其他 hyperparameters，即 `beta_1`、`beta_2`、`epsilon`，以及 weight decay。可以在 D. P. Kingma et al. [22] 中找到这些 hyperparameters 的一些典型选择。

#### 7.2.2 Putting it together

现在你可以把所有内容组合起来：获得 trained BPE tokenizer，tokenize training dataset，并在你写的 training loop 中运行它。重要提示：如果你的 implementation 正确且高效，上述 hyperparameters 应该在 1 个 B200 GPU 上产生约 20 到 30 分钟的 runtime。如果你的 runtime 长很多，请检查 dataloading、checkpointing 或 validation loss code 是否成为 bottleneck，并确认 implementation 已正确 batched。

#### 7.2.3 Tips and tricks for debugging model architectures

我们强烈建议你熟悉 IDE 内置 debugger，例如 VSCode/Zed。相比使用 print statements，它能节省时间。如果你使用 text editor，可以使用类似 `ipdb` 的工具。调试 model architectures 时还有一些良好实践：

- 开发任意 neural net architecture 时，一个常见第一步是 overfit 到单个 minibatch。如果 implementation 正确，应能很快把 training loss 拉到接近 0。
- 在多个 model components 中设置 debug breakpoints，并检查 intermediate tensors 的 shapes，确保它们符合预期。
- 监控 activations、model weights 和 gradients 的 norms，确保它们没有 exploding 或 vanishing。

**Problem (learning_rate): Tune the learning rate (2 B200 hrs) (3 points)**

Learning rate 是最重要的 hyperparameters 之一。基于你训练的 base model，回答以下问题：

(a) 对 learning rates 执行 hyperparameter sweep，并报告 final losses。如果 optimizer diverges，请说明 divergence。

Deliverable: 与多个 learning rates 关联的 learning curves。解释你的 hyperparameter search strategy。

Deliverable: 一个在 TinyStories 上 validation loss，即 per-token，至多为 1.45 的 model。

#### Low-Resource Tip: Train for a few steps on CPU or Apple Silicon

如果你在 `cpu` 或 `mps` 上运行，应将 total tokens processed 降低到 40,000,000，这足以产生相当流畅的文本。你也可以将 target validation loss 从 1.45 提高到 2.00。

在一台 M4 Max 芯片、36 GB RAM 的机器上，使用 tuned learning rate 运行我们的 solution code 时，我们采用 `batch size * total step count * context length = 32 * 5000 * 256 = 40,960,000` tokens。它在 cpu 上耗时 1 小时 22 分钟，在 mps 上耗时 36 分钟。在 step 5000，我们达到 validation loss 1.80。

一些额外提示：

- 使用 `N` training steps 时，建议调整 cosine learning rate decay schedule，使其 decay 在恰好 step `N` 时结束，也就是达到 minimum learning rate。
- 使用 `mps` 时，不要使用 TF32 kernels，即不要像在 cuda devices 上那样设置 `torch.set_float32_matmul_precision('high')`。我们尝试在 mps 上启用 TF32 kernels，torch version 2.9.0，发现 backend 有时会使用 silently broken kernels，导致 training 不稳定。
- 可以通过 `torch.compile` 对 model 做 JIT-compiling 来加速 training。具体来说：
  - 在 cpu 上，用 `model = torch.compile(model)` compile model。
  - 在 mps 上，可以用 `model = torch.compile(model, backend="aot_eager")` 一定程度优化 backward pass。
  - 截至 torch version 2.9.0，mps 不支持 Inductor compilation。

(b) 经验说法是，最佳 learning rate 位于 "edge of stability"。研究 learning rates 开始 diverge 的点与你的最佳 learning rate 之间的关系。

Deliverable: 随 learning rate 增大的 learning curves，至少包含一个 divergent run，并分析它与 convergence rates 的关系。

现在我们改变 batch size，看看 training 会发生什么。Batch sizes 很重要，它们让我们通过更大的 matrix multiplies 更高效地使用 GPUs。但我们是否总是希望 batch sizes 很大？让我们运行一些 experiments 找答案。

**Problem (batch_size_experiment): Batch size variations (1 B200 hr) (1 point)**

将 batch size 从 1 一直变化到 GPU memory limit。中间也尝试至少几个 batch sizes，包括 64 和 128 等典型值。

Deliverable: 不同 batch sizes runs 的 learning curves。必要时应重新优化 learning rates。

Deliverable: 用几句话讨论你关于 batch sizes 及其对 training 影响的发现。

现在有了 decoder，我们可以生成文本。我们会从 model 生成文本，并观察质量。作为参考，你的 outputs 至少应达到下面示例的水平。

**Example (ts_generate_example): Sample output from a TinyStories language model**

从前，有一个名叫 Lily 的漂亮女孩。她喜欢吃口香糖，尤其是那种大的黑色口香糖。一天，Lily 的妈妈让她帮忙做晚饭。Lily 非常兴奋！她喜欢帮妈妈。Lily 的妈妈为晚饭做了一大锅汤。Lily 很开心，说：“谢谢你，妈妈！我爱你。”她帮妈妈把汤倒进一个大碗里。晚饭后，Lily 的妈妈做了一些美味的汤。Lily 很喜欢！她说：“谢谢你，妈妈！这汤真好喝！”她妈妈微笑着说：“我很高兴你喜欢，Lily。”她们完成了烹饪，并继续一起做饭。结束。

#### Low-Resource Tip: Generate text on CPU or Apple Silicon

如果你使用的是 processed 40M tokens 的 low-resource configuration，应能看到仍然像英语、但不如上例流畅的 generations。例如，我们从一个在 40M tokens 上训练的 TinyStories language model 得到的 sample output 如下：

> Once upon a time, there was a little girl named Sue. Sue had a tooth that she loved very much. It was his best head. One day, Sue went for a walk and met a ladybug! They became good friends and played on the path together.
>
> “Hey, Polly! Let’s go out!” said Tim. Sue looked at the sky and saw that it was difficult to find a way to dance shining. She smiled and agreed to help the talking!“
>
> As Sue watched the sky moved, what it was. She

下面是精确的问题陈述和要求：

**Problem (generate): Generate text (1 point)**

使用你的 decoder 和 trained checkpoint，报告你的 model 生成的文本。你可能需要调整 decoder parameters，例如 temperature、top-p 等，以得到 fluent outputs。

Deliverable: 至少 256 tokens 的 text dump，或直到第一个 `<|endoftext|>` token，并简要评论该 output 的 fluency，以及至少两个影响该 output 好坏的因素。

### 7.3 Ablations and architecture modification

理解 Transformer 的最佳方式，是实际修改它并观察其行为。现在我们做几个简单的 ablations 和 modifications。

#### Ablation 1: layer normalization

人们常说 layer normalization 对 Transformer training 的 stability 很重要。但也许我们想冒险一下。让我们从每个 Transformer block 中移除 RMSNorm，看看会发生什么。

**Problem (layer_norm_ablation): Remove RMSNorm and train (0.5 B200 hrs) (1 point)**

移除 Transformer 中所有 RMSNorms 并训练。在之前的 optimal learning rate 下会发生什么？能否通过使用较低 learning rate 获得 stability？

Deliverable: 一条移除 RMSNorms 并训练时的 learning curve，以及一条最佳 learning rate 的 learning curve。

Deliverable: 用几句话评论 RMSNorm 的影响。

现在我们研究另一个乍看有些任意的 layer normalization 选择。Pre-norm Transformer blocks 定义为：

```text
z = x + MultiHeadSelfAttention(RMSNorm(x))
y = z + FFN(RMSNorm(z))
```

这是对原始 Transformer architecture 少数几个已经形成 "consensus" 的修改之一。原始 Transformer 使用 post-norm approach：

```text
z = RMSNorm(x + MultiHeadSelfAttention(x))
y = RMSNorm(z + FFN(z))
```

让我们回到 post-norm approach，看看会发生什么。

**Problem (pre_norm_ablation): Implement post-norm and train (0.5 B200 hrs) (1 point)**

将你的 pre-norm Transformer implementation 修改为 post-norm one。使用 post-norm model 训练，并观察发生什么。

Deliverable: 一条 post-norm Transformer 的 learning curve，并与 pre-norm 进行比较。

我们看到，layer normalization 会对 Transformer 的 behavior 产生重大影响，甚至 layer normalization 的位置也很重要。

#### Ablation 2: position embeddings

接下来我们研究 position embeddings 对 model performance 的影响。具体来说，我们将比较 base model，即使用 RoPE，和完全不包含 position embeddings，即 NoPE。事实证明，decoder-only transformers，也就是像我们实现的这种带 causal mask 的 transformers，在理论上可以在不显式提供 position embeddings 的情况下推断 relative 或 absolute position information [Y.-H. H. Tsai et al., 2019; A. Kazemnejad et al., 2023]。现在我们将 empirically 测试 NoPE 与 RoPE 的表现对比。

**Problem (no_pos_emb): Implement NoPE (0.5 B200 hrs) (1 point)**

修改你的带 RoPE 的 Transformer implementation，完全移除 position embedding information，并观察会发生什么。

Deliverable: 比较 RoPE 和 NoPE performance 的 learning curve。

#### Ablation 3: SwiGLU vs. SiLU

接下来，我们遵循 N. Shazeer [20]，通过比较 SwiGLU feed-forward networks 与使用 SiLU activations 但没有 gated linear unit (GLU) 的 feed-forward networks，测试 feed-forward network 中 gating 的重要性：

```text
FFN_SiLU(x) = W2 SiLU(W1 x)
```

回忆一下，在我们的 SwiGLU implementation 中，inner feed-forward layer 的 dimensionality 约为 `d_ff = 8/3 d_model`，同时确保 `d_ff mod 64 = 0`，以利用 GPU tensor cores。在这个 ablation baseline 中，你的 `FFN_SiLU` implementation 应改设 `d_ff = 4 * d_model`，以近似匹配默认 SwiGLU feed-forward network 的 parameter count，因为 SwiGLU 有三个 weight matrices，而不是两个。

**Problem (swiglu_ablation): SwiGLU vs. SiLU (0.5 B200 hrs) (1 point)**

Deliverable: 一条比较 SwiGLU 和 SiLU feed-forward networks performance 的 learning curve，二者应具有近似匹配的 parameter counts。

Deliverable: 用几句话讨论你的发现。

#### Low-Resource Tip: Online students with limited GPU resources should test modifications on TinyStories

在作业剩余部分，我们将转向更大规模、更嘈杂的 web dataset，即 OpenWebText，并实验 architecture modifications，且可选地提交到课程 leaderboard。

在 OpenWebText 上将 LM 训练到流畅需要很长时间，因此我们建议 GPU access 有限的 online students 继续在 TinyStories 上测试 modifications，使用 validation loss 作为 metric 来评估 performance。

### 7.4 Running on OpenWebText

现在我们转向一个从 web crawl 创建的更标准 pretraining dataset。OpenWebText [A. Gokaslan et al., 2019] 的一个小 sample 也以单个 text file 提供；如何访问该文件见 Section 1。

下面是 OpenWebText 的一个示例。注意文本更真实、更复杂、更多样。你可能想浏览 training dataset，以了解 web-scraped corpus 的 training data 是什么样。

**Example (owt_example): One example from OWT**

Baseball Prospectus 技术总监 Harry Pavlidis 在雇用 Jonathan Judge 时冒了一个险。

Pavlidis 知道，正如 Alan Schwarz 在 *The Numbers Game* 中所写，“美国文化中没有哪个角落比棒球运动员的表现更精确地被计数、更热情地被量化。” 只需点击几下，你就可以发现 Noah Syndergaard 的 fastball 在到达本垒板途中每分钟旋转超过 2,100 次，Nelson Cruz 在 2016 年 qualified hitters 中拥有最高的平均 exit velocity，以及无数其他像从电子游戏或科幻小说中撕出来的细节。不断上涨的数据海洋赋予棒球文化中一个日益重要的角色以力量：analytical hobbyist。

这种 empowerment 也带来了额外审视：不仅审视 measurements，也审视背后的人和 publications。对于 Baseball Prospectus，Pavlidis 很了解 quantitative imperfection 所伴随的 backlash。他也知道，该网站的 catching metrics 需要重做，而这需要一个有学识的头脑，也就是能处理 complex statistical modeling problems 的人，来完成这项工作。

“He freaks us out.” Harry Pavlidis

Pavlidis 有一种预感，基于 Judge 的写作以及他们在一次网站赞助的 ballpark event 上的互动，Judge “got it”。[…]

Note: 你可能需要为这个 experiment 重新 tune hyperparameters，例如 learning rate 或 batch size。

**Problem (main_experiment): Experiment on OWT (2 B200 hrs) (2 points)**

使用与 TinyStories 相同的 model architecture 和 total training iterations，在 OpenWebText 上训练你的 language model。这个 model 表现如何？

Deliverable: 你的 language model 在 OpenWebText 上的 learning curve。描述它与 TinyStories 上 losses 的差异；我们应如何解释这些 losses？

Deliverable: 以与 TinyStories outputs 相同的格式，给出 OpenWebText LM 生成的文本。该文本的 fluency 如何？为什么即使使用相同 model 和 compute budget，output quality 仍比 TinyStories 更差？

### 7.5 Your own modification + leaderboard

恭喜你做到这里。你快完成了。现在你将尝试改进 Transformer architecture，并看看你的 hyperparameters 和 architecture 与班上其他学生相比如何。

#### Rules for the leaderboard

除了以下规则，没有其他限制：

**Runtime:** 你的 submission 最多可以在 B200 上运行 45 分钟。如果你使用 SLURM 或 Modal，可能需要在 submission script 中强制执行这一点。

**Data:** 你只能使用我们提供的 OpenWebText training dataset。

除此之外，你可以自由尝试任何方法。

如果你想找一些 implementation ideas，可以查看以下资源：

- State-of-the-art open-source LLM families，例如 Llama 3 [A. Grattafiori et al., 2024] 或 Qwen 2.5 [A. Yang et al., 2024]。
- NanoGPT speedrun repository，即 `github.com/KellerJordan/modded-nanogpt`。其中 community members 发布了许多有趣 modifications，用于 "speedrunning" small-scale language model pretraining。例如，一个可追溯到原始 Transformer paper 的常见修改是 tie input 和 output embeddings 的 weights，参见 A. Vaswani et al. [8] Section 3.4 和 A. Chowdhery et al. [16] Section 2。如果尝试 weight tying，可能需要降低 embedding/LM head init 的 standard deviation。

在尝试完整 45-minute run 前，你会希望先在 OpenWebText 的小子集或 TinyStories 上测试这些修改。

需要提示的是，我们确实注意到，某些在这个 leaderboard 中表现良好的 modifications 可能无法泛化到 larger-scale pretraining。我们会在课程的 scaling laws unit 中进一步探讨这一点。

**Problem (leaderboard): Leaderboard (10 B200 hrs) (6 points)**

你将在上述 leaderboard rules 下训练一个 model，目标是在 0.75 B200-hours 内最小化你的 language model 的 validation loss。

Deliverable: 记录到的 final validation loss，一条清晰显示 wall-clock-time x-axis 小于 45 minutes 的 associated learning curve，以及对你所做工作的描述。我们期望 leaderboard submission 至少超过 naive baseline，即 loss 5.0。提交到 leaderboard：`github.com/stanford-cs336/assignment1-basics-leaderboard`。

## Bibliography

[1] R. Eldan and Y. Li, "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?." 2023.

[2] A. Gokaslan, V. Cohen, E. Pavlick, and S. Tellex, "OpenWebText corpus." 2019.

[3] R. Sennrich, B. Haddow, and A. Birch, "Neural Machine Translation of Rare Words with Subword Units," in Proc. of ACL, 2016.

[4] C. Wang, K. Cho, and J. Gu, "Neural Machine Translation with Byte-Level Subwords." 2019.

[5] P. Gage, "A new algorithm for data compression," C Users Journal, vol. 12, no. 2, pp. 23-38, Feb. 1994.

[6] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, "Language Models are Unsupervised Multitask Learners." 2019.

[7] A. Radford, K. Narasimhan, T. Salimans, and I. Sutskever, "Improving Language Understanding by Generative Pre-Training." 2018.

[8] A. Vaswani et al., "Attention is All you Need," in Proc. of NeurIPS, 2017.

[9] T. Q. Nguyen and J. Salazar, "Transformers without Tears: Improving the Normalization of Self-Attention," in Proc. of IWSWLT, 2019.

[10] R. Xiong et al., "On Layer Normalization in the Transformer Architecture," in Proc. of ICML, 2020.

[11] J. L. Ba, J. R. Kiros, and G. E. Hinton, "Layer Normalization." 2016.

[12] H. Touvron et al., "LLaMA: Open and Efficient Foundation Language Models." 2023.

[13] B. Zhang and R. Sennrich, "Root Mean Square Layer Normalization," in Proc. of NeurIPS, 2019.

[14] A. Grattafiori et al., "The Llama 3 Herd of Models." [Online]. Available: https://arxiv.org/abs/2407.21783

[15] A. Yang et al., "Qwen2.5 Technical Report," arXiv preprint arXiv:2412.15115, 2024.

[16] A. Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways." 2022.

[17] D. Hendrycks and K. Gimpel, "Bridging Nonlinearities and Stochastic Regularizers with Gaussian Error Linear Units." 2016.

[18] S. Elfwing, E. Uchibe, and K. Doya, "Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning." [Online]. Available: https://arxiv.org/abs/1702.03118

[19] Y. N. Dauphin, A. Fan, M. Auli, and D. Grangier, "Language Modeling with Gated Convolutional Networks." [Online]. Available: https://arxiv.org/abs/1612.08083

[20] N. Shazeer, "GLU Variants Improve Transformer." 2020.

[21] J. Su, Y. Lu, S. Pan, B. Wen, and Y. Liu, "RoFormer: Enhanced Transformer with Rotary Position Embedding." 2021.

[22] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in Proc. of ICLR, 2015.

[23] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," in Proc. of ICLR, 2019.

[24] T. B. Brown et al., "Language Models are Few-Shot Learners," in Proc. of NeurIPS, 2020.

[25] J. Kaplan et al., "Scaling Laws for Neural Language Models." 2020.

[26] J. Hoffmann et al., "Training Compute-Optimal Large Language Models." 2022.

[27] A. Holtzman, J. Buys, L. Du, M. Forbes, and Y. Choi, "The Curious Case of Neural Text Degeneration," in Proc. of ICLR, 2020.

[28] Y.-H. H. Tsai, S. Bai, M. Yamada, L.-P. Morency, and R. Salakhutdinov, "Transformer Dissection: An Unified Understanding for Transformer`s Attention via the Lens of Kernel," in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), K. Inui, J. Jiang, V. Ng, and X. Wan, Eds., Hong Kong, China: Association for Computational Linguistics, Nov. 2019, pp. 4344-4353. doi: 10.18653/v1/D19-1443.

[29] A. Kazemnejad, I. Padhi, K. Natesan, P. Das, and S. Reddy, "The Impact of Positional Encoding on Length Generalization in Transformers," in Thirty-seventh Conference on Neural Information Processing Systems, 2023. [Online]. Available: https://openreview.net/forum?id=Drrl2gcjzl
