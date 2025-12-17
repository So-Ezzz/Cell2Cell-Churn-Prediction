SHELL := /bin/bash
# ========== 基本配置 ==========
ENV_NAME = cell2cell
PYTHON = python
MAIN = main.py

# ========== Conda 路径 ==========
CONDA_BASE := $(shell conda info --base)
CONDA_ACTIVATE := source $(CONDA_BASE)/etc/profile.d/conda.sh

# ========== 默认目标 ==========
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make env        Create conda environment"
	@echo "  make install    Install dependencies"
	@echo "  make run        Run main.py"
	@echo "  make clean      Remove conda environment"

# ========== 创建环境 ==========
.PHONY: env
env:
	@echo ">>> Creating conda environment: $(ENV_NAME)"
	@$(CONDA_ACTIVATE) && \
	conda create -y -n $(ENV_NAME) python=3.11

# ========== 安装依赖 ==========
.PHONY: install
install:
	@echo ">>> Installing dependencies"
	@$(CONDA_ACTIVATE) && \
	conda activate $(ENV_NAME) && \
	pip install -r requirements.txt

# ========== 运行主程序 ==========
.PHONY: run
run:
	@echo ">>> Running main.py"
	@$(CONDA_ACTIVATE) && \
	conda activate $(ENV_NAME) && \
	$(PYTHON) $(MAIN)

# ========== 一键复现 ==========
.PHONY: all
all: env install run

# ========== 清理环境 ==========
.PHONY: clean
clean:
	@echo ">>> Removing conda environment: $(ENV_NAME)"
	@$(CONDA_ACTIVATE) && \
	conda remove -y -n $(ENV_NAME) --all