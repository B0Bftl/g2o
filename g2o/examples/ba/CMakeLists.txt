add_executable(ba_demo
  ba_demo.cpp
)

add_executable(ba_demo_block
    ba_demo_block.cpp)

include_directories(${CHOLMOD_INCLUDE_DIR})

target_link_libraries(ba_demo core solver_cholmod types_sba )

target_link_libraries(ba_demo_block core solver_cholmod types_sba )



