# 测试说明

`tests/` 覆盖公共入口、训练适配、奖励、评测和数据处理；`tests/harness/` 覆盖统一交互契约、SFT 流水线及 Eval v2。默认测试发现包含这两部分，并采用独立模块导入以避免同名测试文件冲突。

```bash
uv run shopping-grpo smoke
uv run pytest
```

迁移专项检查：

```bash
uv run pytest tests/test_package_layout.py tests/test_naming_compatibility.py
```

内置 ShopSimulator 的测试继续保留在环境目录，准备环境依赖后单独运行：

```bash
uv run pytest environments/ShopSimulator/shop_env/tests
```

完整训练与采样测试需要相应可选依赖及 Linux / WSL；`fcntl` 是 Linux 采样文件锁依赖。当前整理的基线、已通过检查和既有失败见 [迁移验证记录](../docs/migration.md)。
