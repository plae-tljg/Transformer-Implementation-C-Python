# 编译器设置
CC = gcc
CFLAGS = -Wall -Wextra

# 项目根目录
ROOT_DIR := $(shell pwd)

# 查找所有的源文件和头文件
SRC_FILES := $(shell find $(ROOT_DIR) -name "*.c")
HEADER_FILES := $(shell find $(ROOT_DIR) -name "*.h")

# 添加 build 目录
BUILD_DIR = $(ROOT_DIR)/build

# 修改目标文件路径
OBJ_FILES := $(patsubst %.c,$(BUILD_DIR)/%.o,$(SRC_FILES))

# 可执行文件名
TARGET = $(BUILD_DIR)/my_program

# 获取所有include目录
INCLUDE_DIRS := $(shell find $(ROOT_DIR) -type d -name "include")
INCLUDE_FLAGS := $(addprefix -I,$(INCLUDE_DIRS))

# 默认目标
all: $(BUILD_DIR) $(TARGET)

# 创建 build 目录
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# 链接目标文件生成可执行文件
$(TARGET): $(OBJ_FILES)
	$(CC) $(OBJ_FILES) -o $@ -lm

# 编译规则 - 需要创建对应的目录结构
$(BUILD_DIR)/%.o: %.c $(HEADER_FILES)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDE_FLAGS) -c $< -o $@

# 清理编译产物
clean:
	rm -rf $(BUILD_DIR)

# 显示编译信息
info:
	@echo "Source files:"
	@echo $(SRC_FILES)
	@echo "\nHeader files:"
	@echo $(HEADER_FILES)
	@echo "\nInclude directories:"
	@echo $(INCLUDE_DIRS)

.PHONY: all clean info