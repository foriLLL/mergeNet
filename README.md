# 行交织冲突消解器

本项目包含行交织类型的冲突消解器的模型及训练代码。

## 训练模型

如需自行训练模型，首先需要下载预训练模型，然后准备数据集，最后调用训练代码进行训练。

### 下载预训练模型

新建 bert 文件夹，下载预训练模型。

```bash
$ mkdir bert && cd bert
$ git lfs install
$ git clone https://huggingface.co/huggingface/CodeBERTa-small-v1
```

### 数据处理

首先对数据进行处理，模型训练需要处理数据集，数据集以 JSON 格式存储，每个文件包含一个 JSON 数组，数组中的每个元素是一个 JSON 对象，包含以下字段：

- `ours`：当前分支的修改
- `theirs`：目标分支的修改
- `base`：共同父节点版本的修改
- `resolve`：冲突消解后的修改

示例：
```json
[
    {
        "ours": [
            "    setDetailNode(categoryTree);",
            "",
            "    // Load last selected category in TreeView.",
            "    categoryTree.setSelectedCategoryById(preferences.getInt(SELECTED_CATEGORY, DEFAULT_CATEGORY));",
            "    TreeItem treeItem = (TreeItem) categoryTree.getSelectionModel().getSelectedItem();",
            "    setSelectedCategory((Category) treeItem.getValue());"
        ],
        "theirs": [
            "    setDetailNode(categoryTreeBox);",
            "    // Sets initial shown CategoryPane.",
            "    setMasterNode(this.categories.get(INITIAL_CATEGORY).getCategoryPane());",
            "    setDividerPosition(DIVIDER_POSITION);"
        ],
        "base": [
            "    setDetailNode(categoryTree);",
            "    // Sets initial shown CategoryPane.",
            "    setMasterNode(this.categories.get(INITIAL_CATEGORY).getCategoryPane());",
            "    setDividerPosition(DIVIDER_POSITION);"
        ],
        "resolve": [
            "    setDetailNode(categoryTreeBox);",
            "    // Load last selected category in TreeView.",
            "    categoryTree.setSelectedCategoryById(preferences.getInt(SELECTED_CATEGORY, DEFAULT_CATEGORY));",
            "    TreeItem treeItem = (TreeItem) categoryTree.getSelectionModel().getSelectedItem();",
            "    setSelectedCategory((Category) treeItem.getValue());"
        ]
    }
]
```

### 调用 tokenizer
数据集格式处理正确后，调用预训练模型的 tokenizer，将数据集转换为模型可接受的格式。

修改 `data_process.py` 中的 `bert_path` 为预训练模型的路径，`data_path` 为数据集的路径，`output_path` 为输出数据集的路径。

接着运行 `data_process.py` ，得到输出数据集即可。

### 训练模型

在 `train.py` 中修改 `params` 中的下列参数，然后运行 `train.py` 即可。

```py
params['codeBERT'] = './bert/CodeBERTa-small-v1'
params['save_path'] = './output/finalModel4MergeBertData.pt'
params['dataset_path'] = './output/tokenized_output'                # tokenized dataset path
```

最后得到的模型保存在 `save_path` 下。