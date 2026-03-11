cmake_minimum_required(VERSION 3.19)
project(fwmc-viewer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
endif()

# GLFW (disable extras we don't need)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(vendor/glfw)

# GLAD (generated OpenGL 3.3 Core loader)
add_library(glad STATIC vendor/glad/src/gl.c)
target_include_directories(glad PUBLIC vendor/glad/include)

# Dear ImGui
add_library(imgui STATIC
  vendor/imgui-src/imgui.cpp
  vendor/imgui-src/imgui_draw.cpp
  vendor/imgui-src/imgui_widgets.cpp
  vendor/imgui-src/imgui_tables.cpp
  vendor/imgui-src/backends/imgui_impl_glfw.cpp
  vendor/imgui-src/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC
  vendor/imgui-src
  vendor/imgui-src/backends
)
target_link_libraries(imgui PUBLIC glfw glad)
target_compile_definitions(imgui PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLAD2)

# Brain model headers (for brain_sdf.h, voxel_grid.h, neural_field.h)
set(BRAIN_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../brain-model")

# --- Optional: MuJoCo body simulation (NeuroMechFly) ---
option(FWMC_BODY_SIM "Build with MuJoCo body simulation" ON)

if(FWMC_BODY_SIM)
  # Find MuJoCo (pip-installed or manually specified).
  if(NOT MUJOCO_DIR)
    execute_process(
      COMMAND python -c "import mujoco, pathlib; print(pathlib.Path(mujoco.__file__).parent)"
      OUTPUT_VARIABLE _MUJOCO_PIP_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(_MUJOCO_PIP_DIR AND EXISTS "${_MUJOCO_PIP_DIR}/include/mujoco/mujoco.h")
      set(MUJOCO_DIR "${_MUJOCO_PIP_DIR}" CACHE PATH "MuJoCo root directory")
      message(STATUS "Found MuJoCo via pip: ${MUJOCO_DIR}")
    endif()
  endif()

  if(MUJOCO_DIR)
    set(MUJOCO_INCLUDE_DIR "${MUJOCO_DIR}/include")

    if(WIN32)
      set(MUJOCO_DLL "${MUJOCO_DIR}/mujoco.dll")
      # Find or generate import lib.
      set(MUJOCO_LIB "${MUJOCO_DIR}/mujoco.lib")
      if(NOT EXISTS "${MUJOCO_LIB}")
        set(MUJOCO_LIB "${MUJOCO_DIR}/lib/mujoco.lib")
      endif()
      if(NOT EXISTS "${MUJOCO_LIB}")
        set(MUJOCO_LIB "${CMAKE_BINARY_DIR}/mujoco.lib")
      endif()
      if(NOT EXISTS "${MUJOCO_LIB}")
        set(MUJOCO_LIB "${CMAKE_CURRENT_BINARY_DIR}/mujoco.lib")
      endif()
      # Check sibling build directories (flygame-build may have generated it).
      if(NOT EXISTS "${MUJOCO_LIB}")
        set(MUJOCO_LIB "${CMAKE_BINARY_DIR}/../flygame-build/mujoco.lib")
      endif()
      if(NOT EXISTS "${MUJOCO_LIB}")
        # Auto-generate import lib from DLL.
        find_program(_DUMPBIN dumpbin)
        find_program(_LIB lib)
        if(_DUMPBIN AND _LIB)
          execute_process(COMMAND "${_DUMPBIN}" /exports "${MUJOCO_DLL}" OUTPUT_VARIABLE _DUMP)
          set(_DEF "LIBRARY mujoco\nEXPORTS\n")
          string(REGEX MATCHALL "[0-9]+[ \t]+[0-9A-Fa-f]+[ \t]+[0-9A-Fa-f]+[ \t]+([a-zA-Z_][a-zA-Z0-9_]+)" _MATCHES "${_DUMP}")
          foreach(_M ${_MATCHES})
            string(REGEX REPLACE ".*[ \t]" "" _SYM "${_M}")
            string(APPEND _DEF "    ${_SYM}\n")
          endforeach()
          file(WRITE "${CMAKE_BINARY_DIR}/mujoco.def" "${_DEF}")
          execute_process(
            COMMAND "${_LIB}" "/def:${CMAKE_BINARY_DIR}/mujoco.def"
                    "/out:${CMAKE_BINARY_DIR}/mujoco.lib" /machine:x64
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
          )
          set(MUJOCO_LIB "${CMAKE_BINARY_DIR}/mujoco.lib")
        endif()
      endif()

      if(EXISTS "${MUJOCO_LIB}")
        add_library(mujoco::mujoco SHARED IMPORTED GLOBAL)
        set_target_properties(mujoco::mujoco PROPERTIES
          IMPORTED_LOCATION "${MUJOCO_DLL}"
          IMPORTED_IMPLIB "${MUJOCO_LIB}"
          INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_INCLUDE_DIR}"
        )
        set(MUJOCO_FOUND TRUE)
        message(STATUS "MuJoCo: ${MUJOCO_DIR} (lib: ${MUJOCO_LIB})")
      else()
        message(WARNING "MuJoCo DLL found but no import lib. Body sim disabled.")
        set(MUJOCO_FOUND FALSE)
      endif()
    else()
      find_library(MUJOCO_LIB NAMES mujoco
        PATHS "${MUJOCO_DIR}" "${MUJOCO_DIR}/lib" NO_DEFAULT_PATH)
      if(MUJOCO_LIB)
        add_library(mujoco::mujoco SHARED IMPORTED GLOBAL)
        set_target_properties(mujoco::mujoco PROPERTIES
          IMPORTED_LOCATION "${MUJOCO_LIB}"
          INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_INCLUDE_DIR}"
        )
        set(MUJOCO_FOUND TRUE)
      else()
        set(MUJOCO_FOUND FALSE)
      endif()
    endif()
  else()
    message(STATUS "MuJoCo not found. Body sim disabled. Set -DMUJOCO_DIR=<path>")
    set(MUJOCO_FOUND FALSE)
  endif()
endif()

# flygame headers (CPG, walk data) for the body sim.
set(NMFLY_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../flygame/cpp/src")

# Viewer executable
add_executable(fwmc-viewer viewer.cc)
if(MSVC)
  set_target_properties(fwmc-viewer PROPERTIES LINK_FLAGS "/SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup")
endif()
target_compile_definitions(fwmc-viewer PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(fwmc-viewer PRIVATE
  ${BRAIN_MODEL_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
)
target_link_libraries(fwmc-viewer PRIVATE imgui glad glfw)

# OpenMP for parallel neuron dynamics in the viewer
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(fwmc-viewer PRIVATE OpenMP::OpenMP_CXX)
endif()

if(FWMC_BODY_SIM AND MUJOCO_FOUND)
  target_compile_definitions(fwmc-viewer PRIVATE FWMC_BODY_SIM=1)
  target_link_libraries(fwmc-viewer PRIVATE mujoco::mujoco)
  target_include_directories(fwmc-viewer PRIVATE ${NMFLY_SRC_DIR})
  message(STATUS "Body sim enabled (NMF headers: ${NMFLY_SRC_DIR})")

  # Copy MuJoCo DLL next to viewer exe.
  if(WIN32)
    add_custom_command(TARGET fwmc-viewer POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${MUJOCO_DLL}" "$<TARGET_FILE_DIR:fwmc-viewer>"
    )
  endif()

  # Copy NMF model data (XML + mesh STLs) next to exe so the viewer finds it at runtime.
  set(NMF_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../flygame/cpp/data")
  if(EXISTS "${NMF_DATA_DIR}/nmf_complete.xml")
    add_custom_command(TARGET fwmc-viewer POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${NMF_DATA_DIR}" "$<TARGET_FILE_DIR:fwmc-viewer>/data"
    )
  endif()
endif()

# Copy viewer data files (regions.toml, brain specs) next to exe.
add_custom_command(TARGET fwmc-viewer POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:fwmc-viewer>/data"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/regions.toml" "$<TARGET_FILE_DIR:fwmc-viewer>/data/"
)
set(EXAMPLES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../examples")
if(EXISTS "${EXAMPLES_DIR}/drosophila_full.brain")
  add_custom_command(TARGET fwmc-viewer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${EXAMPLES_DIR}/drosophila_full.brain" "$<TARGET_FILE_DIR:fwmc-viewer>/data/"
  )
endif()

if(WIN32)
  target_link_libraries(fwmc-viewer PRIVATE opengl32 ws2_32)
elseif(APPLE)
  find_library(OPENGL_LIB OpenGL)
  target_link_libraries(fwmc-viewer PRIVATE ${OPENGL_LIB})
else()
  target_link_libraries(fwmc-viewer PRIVATE GL dl)
endif()
