PACKAGE_NAME := QhpiHmxProbePackage
WORK := build
SRC_DIR := src
INCLUDE_DIR := include
LIBRARY_NAME := libQnn$(PACKAGE_NAME).so

QNN_INCLUDE ?= $(QAIRT_SDK_ROOT)/include/QNN
QNN_TARGET_LIB ?= $(QAIRT_SDK_ROOT)/lib/aarch64-android
QNN_HOST_LIB ?= $(QAIRT_SDK_ROOT)/lib/x86_64-linux-clang
HEXAGON_TOOLS_VERSION ?= 19.0.07

HEXAGON_TOOLS := $(HEXAGON_SDK_ROOT)/tools/HEXAGON_Tools/$(HEXAGON_TOOLS_VERSION)/Tools
NATIVE_INCLUDE := $(HEXAGON_TOOLS)/libnative/include
NATIVE_LIB := $(HEXAGON_TOOLS)/libnative/lib
HEXAGON_CXX := $(HEXAGON_TOOLS)/bin/hexagon-clang++
ANDROID_CXX := $(ANDROID_NDK_ROOT)/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++
X86_CXX ?= clang++-14

COMMON_FLAGS := -std=c++17 -fPIC -Wall -Wextra -Werror -Wno-missing-field-initializers -Wno-unused-parameter \
	-I$(QNN_INCLUDE) -I$(INCLUDE_DIR) \
	-DTHIS_PKG_NAME=$(PACKAGE_NAME) \
	-DQNN_API='__attribute__((visibility("default")))' \
	-D__QAIC_HEADER_EXPORT='__attribute__((visibility("default")))' $(EXTRA_FLAGS)

SOURCES := $(SRC_DIR)/QhpiHmxProbePackageInterface.cpp $(SRC_DIR)/U8S8HmxMatMul.cpp \
	$(SRC_DIR)/MixedResourceHmxHvxHmx.cpp $(SRC_DIR)/SequentialHmxHvxHmx.cpp
KERNEL_HEADER := $(INCLUDE_DIR)/HmxInt8Tile.hpp

.PHONY: all htp_x86 htp_v79 htp_aarch64 emulation simulator clean

all: htp_x86 htp_v79 htp_aarch64

htp_x86: $(WORK)/x86_64-linux-clang/$(LIBRARY_NAME)

htp_v79: $(WORK)/hexagon-v79/$(LIBRARY_NAME)

htp_aarch64: $(WORK)/aarch64-android/$(LIBRARY_NAME)

emulation: $(WORK)/x86_64-linux-clang/hmx_int8_tile_emulation

simulator: $(WORK)/hexagon-v79/hmx_int8_tile_simulator

$(WORK)/x86_64-linux-clang:
	mkdir -p $@

$(WORK)/hexagon-v79:
	mkdir -p $@

$(WORK)/aarch64-android:
	mkdir -p $@

$(WORK)/x86_64-linux-clang/$(LIBRARY_NAME): $(SOURCES) $(KERNEL_HEADER) | $(WORK)/x86_64-linux-clang
	$(X86_CXX) $(COMMON_FLAGS) -I$(NATIVE_INCLUDE) -D__HVXDBL__ -DUSE_OS_LINUX -DREFERENCE_OP \
		-shared -o $@ $(SOURCES) -L$(QNN_HOST_LIB) -lQnnHtp

$(WORK)/x86_64-linux-clang/hmx_int8_tile_emulation: tests/HmxInt8TileEmulation.cpp include/HmxInt8Tile.hpp | $(WORK)/x86_64-linux-clang
	$(X86_CXX) -std=c++17 -Wall -Wextra -Werror -Wno-unused-parameter -D__HVXDBL__ -I$(INCLUDE_DIR) -I$(NATIVE_INCLUDE) \
		-o $@ $< -Wl,--whole-archive -L$(NATIVE_LIB) -lnative -Wl,--no-whole-archive -lpthread

$(WORK)/hexagon-v79/hmx_int8_tile_simulator: tests/HmxInt8TileSimulator.cpp include/HmxInt8Tile.hpp | $(WORK)/hexagon-v79
	$(HEXAGON_CXX) -std=c++17 -O2 -mv79 -mhmx -I$(INCLUDE_DIR) -o $@ $<

$(WORK)/hexagon-v79/$(LIBRARY_NAME): $(SOURCES) $(KERNEL_HEADER) | $(WORK)/hexagon-v79
	$(HEXAGON_CXX) $(COMMON_FLAGS) -O2 -mv79 -mhvx -mhvx-length=128B -mhmx -DUSE_OS_QURT -DPREPARE_DISABLED \
		-I$(HEXAGON_SDK_ROOT)/rtos/qurt/computev79/include/qurt \
		-I$(HEXAGON_SDK_ROOT)/rtos/qurt/computev79/include/posix \
		-I$(HEXAGON_SDK_ROOT)/incs -I$(HEXAGON_SDK_ROOT)/incs/stddef \
		-shared -o $@ $(SOURCES)

$(WORK)/aarch64-android/$(LIBRARY_NAME): $(SOURCES) $(KERNEL_HEADER) | $(WORK)/aarch64-android
	$(ANDROID_CXX) $(COMMON_FLAGS) -D__HVXDBL__ -DUSE_OS_LINUX -DANDROID -DREFERENCE_OP \
		--target=aarch64-none-linux-android21 \
		--sysroot=$(ANDROID_NDK_ROOT)/toolchains/llvm/prebuilt/linux-x86_64/sysroot \
		-stdlib=libc++ -static-libstdc++ -I$(NATIVE_INCLUDE) \
		-shared -o $@ $(SOURCES) -L$(QNN_TARGET_LIB) -lQnnHtp -lQnnHtpPrepare

clean:
	rm -rf $(WORK)
