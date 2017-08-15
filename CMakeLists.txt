##
#  CMake script for the program:
##

# Set the name of the src file
SET(TARGET "practice5")
# Set the name of the input msh file
SET(MSH_FILE "traction_rod2")

# Set the name of the input parameter file
SET(PARAMETER_FILE "parameter_update")

SET(TARGET_SRC
  ${TARGET}.cpp
  )
# if needed, copy files to the build directory
file(COPY ./msh/${MSH_FILE}.msh DESTINATION ./)
file(COPY ./${PARAMETER_FILE}.txt DESTINATION ./)  
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)


PROJECT(${TARGET})
include_directories("D:/cmake/boost/boost-1.64.0.0/include" "D:/cmake/dealii-8.5.0/include" "D:/cmake/dealii-8.5.0/build/include" ${CMAKE_CURRENT_SOURCE_DIR})
set(LIBBASE "D:/cmake/dealii-8.5.0/build/source")
If (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
link_directories(
D:/cmake/boost/boost_serialization-vc140.1.64.0.0/lib/native/address-model-32/lib
"${LIBBASE}/base/obj_base_release.dir/Debug"
${LIBBASE}/numerics/obj_numerics_release.dir/Debug
${LIBBASE}/multigrid/obj_multigrid_release.dir/Debug
${LIBBASE}/dofs/obj_dofs_release.dir/Debug
${LIBBASE}/fe/obj_fe_release.dir/Debug
${LIBBASE}/grid/obj_grid_release.dir/Debug
${LIBBASE}/hp/obj_hp_release.dir/Debug
D:/cmake/dealii-8.5.0/build/DEBUGSTATIC
${LIBBASE}/algorithms/obj_algorithms_release.dir/Debug
${LIBBASE}/distributed/obj_distributed_release.dir/Debug
D:/cmake/dealii-8.5.0/bundled/muparser_v2_2_4/muparser-2.2.5/lib
)
message("Debug build")
elseif(${CMAKE_BUILD_TYPE} STREQUAL "Release")
link_directories(
D:/cmake/boost/boost_serialization-vc140.1.64.0.0/lib/native/address-model-32/lib
${LIBBASE}/base/obj_base_release.dir/Release
${LIBBASE}/numerics/obj_numerics_release.dir/Release
${LIBBASE}/multigrid/obj_multigrid_release.dir/Release
${LIBBASE}/dofs/obj_dofs_release.dir/Release
${LIBBASE}/fe/obj_fe_release.dir/Release
${LIBBASE}/grid/obj_grid_release.dir/Release
${LIBBASE}/hp/obj_hp_release.dir/Release
${LIBBASE}/lac/obj_lac_release.dir/Release
${LIBBASE}/algorithms/obj_algorithms_release.dir/Release
${LIBBASE}/distributed/obj_distributed_release.dir/Release
D:/cmake/dealii-8.5.0/bundled/muparser_v2_2_4/muparser-2.2.5/lib
)
message("release build")
else()
MESSAGE(FATAL_ERROR
      "Unknown build type ${CMAKE_BUILD_TYPE}"
      )
endif() 
add_executable(${TARGET} ${TARGET_SRC})
target_link_libraries(${TARGET} obj_base_release.lib obj_numerics_release.lib obj_multigrid_release.lib obj_dofs_release.lib obj_fe_release.lib obj_grid_release.lib obj_hp_release.lib 
obj_lac_release.lib obj_algorithms_release.lib obj_distributed_release.lib muparser_static.lib)
