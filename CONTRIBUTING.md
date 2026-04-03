# 贡献指南

## 开发环境

```bash
python -m pip install -e ".[dev]"
```

## 质量检查

```bash
ruff check .
mypy src/x2md
pytest -q
```

## 提交规范

- 保持变更最小化、可读、可回滚
- 不提交大文件（PDF/音视频/模型目录）
- 不在日志/输出中打印任何密钥或敏感信息
