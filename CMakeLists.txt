option(USE_OPENMP "Enable OpenMP parallelism" ON)

cmake_minimum_required(VERSION 3.28)

# === Auto-select Homebrew GCC on macOS ===
if(APPLE AND USE_OPENMP)
  file(GLOB HOMEBREW_GPP /opt/homebrew/bin/g++-*)
  if(HOMEBREW_GPP STREQUAL "")
    message(FATAL_ERROR "❌ Homebrew GCC not found! Please run: brew install gcc")
  endif()
  list(SORT HOMEBREW_GPP)
  list(REVERSE HOMEBREW_GPP)
  list(GET HOMEBREW_GPP 0 HOMEBREW_GPP_BIN)
  get_filename_component(HOMEBREW_GPP_VER ${HOMEBREW_GPP_BIN} NAME)
  string(REPLACE "g++-" "" HOMEBREW_GPP_VER_NUM ${HOMEBREW_GPP_VER})

  set(CMAKE_C_COMPILER "/opt/homebrew/bin/gcc-${HOMEBREW_GPP_VER_NUM}" CACHE STRING "" FORCE)
  set(CMAKE_CXX_COMPILER "/opt/homebrew/bin/g++-${HOMEBREW_GPP_VER_NUM}" CACHE STRING "" FORCE)

  message(STATUS "✅ Using Homebrew GCC ${HOMEBREW_GPP_VER_NUM} (path: ${HOMEBREW_GPP_BIN})")
endif()


project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

# If using Visual Studio generators, always target v10.x of the Windows SDK.
# Doing this avoids lookups that could fall back to very old versions, e.g. by finding
# outdated registry entries.
# ref: https://cmake.org/cmake/help/latest/variable/CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION.html
if(CMAKE_GENERATOR MATCHES "Visual Studio")
    set(CMAKE_SYSTEM_VERSION 10.0 CACHE INTERNAL "target Windows SDK version" FORCE)
endif()


if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "7.1")
    message(FATAL_ERROR "❌ GCC >= 7.1 is required for full C++17 support (you have ${CMAKE_CXX_COMPILER_VERSION})")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "6.0")
    message(FATAL_ERROR "❌ Clang >= 6.0 is required for full C++17 support (you have ${CMAKE_CXX_COMPILER_VERSION}")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10.0")
    message(FATAL_ERROR "❌ AppleClang >= 10.0 is required for full C++17 support (you have ${CMAKE_CXX_COMPILER_VERSION}")
  endif()
elseif(MSVC)
  if(MSVC_VERSION LESS 1910)
    message(FATAL_ERROR "❌ MSVC >= 19.10 (Visual Studio 2017) is required. (you have ${CMAKE_CXX_COMPILER_VERSION}")
  endif()
else()
  message(WARNING "⚠️ Unknown compiler: ${CMAKE_CXX_COMPILER_ID} — proceed at your own risk.")
endif()

# Set OpenMP
if(USE_OPENMP)
  # Help CMake find keg-only libomp if needed
  if(APPLE)
    execute_process(
      COMMAND brew --prefix libomp
      OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    list(APPEND CMAKE_PREFIX_PATH "${HOMEBREW_LIBOMP_PREFIX}")
  endif()

  find_package(OpenMP)
  if(OpenMP_FOUND)
    message(STATUS "✅ OpenMP found, parallel compilation will be enabled")
  else()
    message(WARNING "⚠️ OpenMP not found. Parallelism will be disabled!")
  endif()
endif()


# Set optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options(
    -O3
    -ffast-math
    -Wno-unused-function
    -Wno-maybe-uninitialized
    -std=c++17
    #-march=native # <-- Uncomment to enable native optimizations
    #-funroll-loops # <-- Uncomment to enable loop unrolling
    #-ftree-vectorize # <-- Uncomment to enable vectorization
  )
elseif(MSVC)
  add_compile_options(/O2)
  if(USE_OPENMP)
    add_compile_options(/openmp)
  endif()
endif()

# Find packages
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

macro(add_cython_module name)
  string(TOLOWER "${name}" module_name)
  set(PYX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/similaripy/cython_code/${module_name}.pyx)
  set(CPP_PATH ${CMAKE_CURRENT_BINARY_DIR}/${module_name}.cpp)

  add_custom_command(
    OUTPUT ${CPP_PATH}
    COMMAND Python::Interpreter -m cython
            ${PYX_PATH}
            --cplus
            --output-file ${CPP_PATH}
    DEPENDS ${PYX_PATH}
    VERBATIM
  )

  python_add_library(${module_name} MODULE ${CPP_PATH} WITH_SOABI)
  set_target_properties(${module_name} PROPERTIES LINKER_LANGUAGE CXX)
  target_compile_features(${module_name} PRIVATE cxx_std_17)

  if(OpenMP_FOUND)
    target_link_libraries(${module_name} PRIVATE OpenMP::OpenMP_CXX)
    set_target_properties(${module_name} PROPERTIES INSTALL_RPATH "${HOMEBREW_LIBOMP_PREFIX}/lib")
  endif()

  target_include_directories(${module_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/similaripy/cython_code)

  install(TARGETS ${module_name} DESTINATION similaripy/cython_code)
endmacro()

# Log environment summary
message(STATUS "🧠 Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "🐍 Python path: ${Python_EXECUTABLE}")
message(STATUS "🐍 Python version: ${Python_VERSION}")

# Cython modules
message(STATUS "🐍 Cythonizing .pyx files to C++")
foreach(module IN ITEMS s_plus normalization)
  add_cython_module(${module})
endforeach()

