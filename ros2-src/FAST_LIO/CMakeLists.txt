cmake_minimum_required(VERSION 3.8)
project(fast_lio)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

# ─── Multi-processor detection ────────────────────────────────────────
message("Current CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processor number: ${N}")
  if(N GREATER 4)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=3)
    message("core for MP: 3")
  elseif(N GREATER 3)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=2)
    message("core for MP: 2")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

# ─── Metal GPU backend (Apple Silicon / macOS) ───────────────────────
set(USE_METAL OFF)
if(APPLE)
  find_library(METAL_FRAMEWORK Metal)
  find_library(FOUNDATION_FRAMEWORK Foundation)
  if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
    set(USE_METAL ON)
    message(STATUS "Metal GPU backend: ENABLED")

    set(METAL_SHADER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/include/compute/metal/kernels.metal")
    set(METAL_AIR "${CMAKE_CURRENT_BINARY_DIR}/kernels.air")
    set(METAL_LIB "${CMAKE_CURRENT_BINARY_DIR}/kernels.metallib")
    add_custom_command(
      OUTPUT ${METAL_AIR}
      COMMAND xcrun -sdk macosx metal -c ${METAL_SHADER_SRC} -o ${METAL_AIR}
      DEPENDS ${METAL_SHADER_SRC}
      COMMENT "Compiling Metal shaders → kernels.air"
    )
    add_custom_command(
      OUTPUT ${METAL_LIB}
      COMMAND xcrun -sdk macosx metallib ${METAL_AIR} -o ${METAL_LIB}
      DEPENDS ${METAL_AIR}
      COMMENT "Linking Metal library → kernels.metallib"
    )
    add_custom_target(metal_shaders DEPENDS ${METAL_LIB})
  else()
    message(STATUS "Metal GPU backend: DISABLED (Metal framework not found)")
  endif()
else()
  message(STATUS "Metal GPU backend: DISABLED (non-Apple platform)")
endif()

# ─── CUDA GPU backend (NVIDIA GPUs) ──────────────────────────────────
set(USE_CUDA OFF)
if(NOT USE_METAL)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(USE_CUDA ON)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    message(STATUS "CUDA GPU backend: ENABLED (${CMAKE_CUDA_COMPILER})")
  else()
    message(STATUS "CUDA GPU backend: DISABLED (no CUDA compiler found)")
  endif()
else()
  message(STATUS "CUDA GPU backend: DISABLED (Metal backend active)")
endif()

# ─── ROS 2 dependencies ──────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(rosidl_default_generators REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)

message(STATUS "Eigen: ${EIGEN3_INCLUDE_DIR}")

# ─── Message generation ──────────────────────────────────────────────
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Pose6D.msg"
  DEPENDENCIES geometry_msgs
)

# ─── Include directories ─────────────────────────────────────────────
include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  include
)

# ─── Build target ────────────────────────────────────────────────────
if(USE_METAL)
  set_source_files_properties(include/compute/metal_backend.mm PROPERTIES
    LANGUAGE OBJCXX
    COMPILE_FLAGS "-fobjc-arc -std=c++17"
  )
  add_executable(fastlio_mapping
    src/laserMapping.cpp
    include/ikd-Tree/ikd_Tree.cpp
    src/preprocess.cpp
    include/compute/metal_backend.mm
  )
  add_dependencies(fastlio_mapping metal_shaders)
  target_compile_definitions(fastlio_mapping PRIVATE HAS_METAL USE_GPU_BACKEND)
  target_link_libraries(fastlio_mapping
    ${PCL_LIBRARIES}
    ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK}
  )
  target_compile_definitions(fastlio_mapping PRIVATE
    METALLIB_PATH="${CMAKE_CURRENT_BINARY_DIR}/kernels.metallib"
  )
elseif(USE_CUDA)
  add_executable(fastlio_mapping
    src/laserMapping.cpp
    include/ikd-Tree/ikd_Tree.cpp
    src/preprocess.cpp
    include/compute/cuda/cuda_backend.cu
    include/compute/cuda/kernels.cu
  )
  target_compile_definitions(fastlio_mapping PRIVATE HAS_CUDA USE_GPU_BACKEND)
  target_link_libraries(fastlio_mapping ${PCL_LIBRARIES})
  target_include_directories(fastlio_mapping PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_target_properties(fastlio_mapping PROPERTIES CUDA_ARCHITECTURES "60;70;75;80;86;89;90")
  endif()
  set_target_properties(fastlio_mapping PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )
else()
  add_executable(fastlio_mapping
    src/laserMapping.cpp
    include/ikd-Tree/ikd_Tree.cpp
    src/preprocess.cpp
  )
  target_link_libraries(fastlio_mapping ${PCL_LIBRARIES})
endif()

# ─── ROS 2 ament dependencies ────────────────────────────────────────
ament_target_dependencies(fastlio_mapping
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  tf2_ros
  tf2_eigen
  pcl_conversions
  livox_ros_driver2
)

# Link the generated Pose6D message typesupport
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} rosidl_typesupport_cpp)
target_link_libraries(fastlio_mapping "${cpp_typesupport_target}")

# ─── Install ─────────────────────────────────────────────────────────
install(TARGETS fastlio_mapping
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch rviz_cfg
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
