# Makefile for vulkan_infer — Vulkan/Linux MoE inference engine
CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE -Ideps
LDFLAGS = -ldl -luring -lcblas -lblas -lpthread -lm

GLSLC = glslc

INFER_TARGET = infer
INFER_SRC = infer.c vk_compute.c io_ring.c weights.c linear_attn.c full_attn.c deps/volk.c

# Shader sources and SPIR-V outputs
SHADER_DIR = shaders
SHADER_SRCS = $(wildcard $(SHADER_DIR)/*.comp)
SHADER_SPVS = $(SHADER_SRCS:.comp=.spv)

.PHONY: all clean shaders

all: shaders $(INFER_TARGET)

shaders: $(SHADER_SPVS)

$(SHADER_DIR)/%.spv: $(SHADER_DIR)/%.comp
	$(GLSLC) --target-env=vulkan1.3 -O $< -o $@

$(INFER_TARGET): $(INFER_SRC) vk_compute.h io_ring.h
	$(CC) $(CFLAGS) $(INFER_SRC) -o $(INFER_TARGET) $(LDFLAGS)

test_vk: test_vk.c vk_compute.c deps/volk.c shaders/test_add.spv
	$(CC) $(CFLAGS) test_vk.c vk_compute.c deps/volk.c -o test_vk -ldl -lpthread -lm

test_io: test_io.c io_ring.c
	$(CC) $(CFLAGS) test_io.c io_ring.c -o test_io -luring -lm

test_matvec: test_matvec.c vk_compute.c deps/volk.c shaders/dequant_matvec_4bit.spv
	$(CC) $(CFLAGS) test_matvec.c vk_compute.c deps/volk.c -o test_matvec $(LDFLAGS)

clean:
	rm -f $(INFER_TARGET) test_vk test_io test_matvec $(SHADER_DIR)/*.spv
