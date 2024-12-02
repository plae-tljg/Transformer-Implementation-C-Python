# 编译器设置
CC = clang

# 编译选项
CFLAGS = -Wall -Wextra -I./include -g -lm

# 目录设置
SRC_DIR = src
OBJ_DIR = obj

# 获取所有源文件
SRCS = $(wildcard $(SRC_DIR)/*.c) $(wildcard $(SRC_DIR)/*/*.c)
# 生成目标文件列表
OBJS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))

# 根据TARGET是否定义设置源文件路径
ifdef TARGET
    SRC = $(SRC_DIR)/$(TARGET)/$(TARGET).c
    OBJ = $(OBJ_DIR)/$(TARGET).o
endif

# 默认目标
all: dirs $(TARGET)

# 创建目标目录
dirs:
	mkdir -p $(OBJ_DIR)

# 编译可执行文件
$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $@ $(CFLAGS)

# 编译源文件
$(OBJ): $(SRC)
	$(CC) $(CFLAGS) -c $< -o $@

# 编译所有源文件
build_all: dirs $(OBJS)
	$(CC) $(OBJS) -o program $(CFLAGS)

# 运行程序
run: $(TARGET)
	./$(TARGET)

# 编译并运行目标程序
run_target: clean all
	@echo "Compiling and running $(TARGET)..."
	@./$(TARGET)

# 清理目标
clean:
	rm -rf $(OBJ_DIR) $(TARGET) program

# 修改编译源文件的规则
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: all clean run dirs build_all 