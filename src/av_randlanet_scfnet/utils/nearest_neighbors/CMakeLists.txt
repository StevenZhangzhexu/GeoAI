# Set the cross-compilation toolchain file for MinGW
set(CMAKE_TOOLCHAIN_FILE /path/to/mingw-w64-toolchain-file.cmake)

# Specify the target platform
set(CMAKE_SYSTEM_NAME Windows)

# Specify the target architecture
set(CMAKE_SYSTEM_PROCESSOR x86_64)  # Or whichever architecture you're targeting

# Set the output directory for Windows binaries
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/win_lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/win_lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/win_bin)

# Add your source files
add_library(your_library SHARED ${your_sources})
