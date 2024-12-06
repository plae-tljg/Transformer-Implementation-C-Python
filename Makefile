# 编译器设置
CC = clang

# 目录设置
SRC_ROOT = src
OBJ_DIR = obj
BIN_DIR = bin

# 获取所有源文件目录（排除include目录和educational目录）
SRC_DIRS = $(shell find $(SRC_ROOT) -type d -not -path "*/include" -not -path "*/educational*")

# 获取所有头文件目录
# 1. 项目根目录的include
# 2. 每个模块目录下的include（排除educational目录）
INCLUDE_DIRS = include \
               $(shell find $(SRC_ROOT) -type d -name "include" -not -path "*/educational*")

# 生成-I参数
INCLUDE_FLAGS = $(addprefix -I,$(INCLUDE_DIRS))

# 编译选项
CFLAGS = -Wall -Wextra -g
CFLAGS += $(INCLUDE_FLAGS)
LDFLAGS = -lm

# 获取所有源文件（排除educational目录）
SRCS = $(foreach dir,$(SRC_DIRS),$(wildcard $(dir)/*.c))

# 生成目标文件列表
OBJS = $(patsubst $(SRC_ROOT)/%.c,$(OBJ_DIR)/%.o,$(SRCS))

# 获取所有模块名称（排除educational目录）
MODULES = $(notdir $(shell find $(SRC_ROOT) -mindepth 1 -maxdepth 1 -type d -not -path "*/educational"))

# 根据TARGET是否定义设置源文件路径
ifdef TARGET
    TARGET_SRC = $(shell find $(SRC_ROOT) -name "$(TARGET).c")
    ifeq ($(TARGET_SRC),)
        $(error Source file for target $(TARGET) not found)
    endif
    TARGET_OBJ = $(patsubst $(SRC_ROOT)/%.c,$(OBJ_DIR)/%.o,$(TARGET_SRC))
endif

# 默认目标
all: dirs $(TARGET)

# 创建必要的目录
dirs:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(BIN_DIR)
	@for dir in $(SRC_DIRS); do \
		mkdir -p $(OBJ_DIR)/$${dir#$(SRC_ROOT)/}; \
	done

# 编译可执行文件
$(TARGET): $(TARGET_OBJ)
	$(CC) $(TARGET_OBJ) -o $(BIN_DIR)/$@ $(LDFLAGS)

# 编译所有源文件
build_all: dirs $(OBJS)
	$(CC) $(OBJS) -o $(BIN_DIR)/program $(LDFLAGS)

# 编译规则
$(OBJ_DIR)/%.o: $(SRC_ROOT)/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

# 清理目标
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# 显示编译信息
info:
	@echo "Source directories:"
	@echo "$(SRC_DIRS)" | tr ' ' '\n'
	@echo "\nInclude directories:"
	@echo "$(INCLUDE_DIRS)" | tr ' ' '\n'
	@echo "\nSource files:"
	@echo "$(SRCS)" | tr ' ' '\n'
	@echo "\nObject files:"
	@echo "$(OBJS)" | tr ' ' '\n'
	@echo "\nModules:"
	@echo "$(MODULES)" | tr ' ' '\n'

# 为每个模块创建单独的目标
define make-module-target
.PHONY: $(1)
$(1): dirs $$(filter $(OBJ_DIR)/$(1)/%.o,$$(OBJS))
	@echo "Building module $(1)"
endef

$(foreach module,$(MODULES),$(eval $(call make-module-target,$(module))))

# 运行程序
run: $(TARGET)
	./$(BIN_DIR)/$(TARGET)

# 编译并运行目标程序
run_target: clean all run

.PHONY: all clean run dirs build_all info run_target $(MODULES)